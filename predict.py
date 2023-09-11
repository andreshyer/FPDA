from os import listdir
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer
from numpy import array, loadtxt, argsort, mean, pi, sqrt
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from matplotlib import pyplot as plt

from utils.drop_profiles import new_drop_profile
from utils.drop_images import new_drop_image
from utils.keras_CNN import default_predict


def trad_predict(y):
    file, output_dir = y

    # Load drop profile
    drop_profile = loadtxt(file, delimiter=",")
    true_x, true_z = drop_profile[:, 0], drop_profile[:, 1]
    true_max_x = true_x.max()

    def error(bond_number_i):

        # Predict Bo
        _, predicted_drop_profile_i = new_drop_profile(bond_number=bond_number_i[0],
                                                       max_worthington_number=1, delta_s=1e-3)
        predicted_x_i, predicted_z_i = predicted_drop_profile_i[:, 0], predicted_drop_profile_i[:, 1]
        predicted_max_x_i = predicted_x_i.max()

        # Fit function
        f_i = UnivariateSpline(predicted_z_i, predicted_x_i, s=0)

        # Scale drop profile
        t_x_i, t_z_i = true_x * (predicted_max_x_i / true_max_x), true_z * (predicted_max_x_i / true_max_x)

        # Calculate error between profiles
        error_mse = sum(abs(t_x_i - f_i(t_z_i)))
        return error_mse

    bond_number = minimize(error, array([0.35]), bounds=[(0.1, 0.4)], method='L-BFGS-B', options={'maxiter': 100}).x[0]
    bond_number = round(bond_number, 3)

    if output_dir:
        _, predicted_drop_profile = new_drop_profile(bond_number=bond_number, max_worthington_number=1, delta_s=1e-3)
        predicted_x, predicted_z = predicted_drop_profile[:, 0], predicted_drop_profile[:, 1]
        predicted_max_x = predicted_x.max()

        # Fit function
        f = UnivariateSpline(predicted_z, predicted_x, s=0)

        t_x, t_z = true_x * (predicted_max_x / true_max_x), true_z * (predicted_max_x / true_max_x)

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        _, drop_image = new_drop_image(drop_profile, img_size_pix=256, rotation=0, drop_scale=1, noise=0,
                                       rel_capillary_height=0, above_apex=False, delta_s=1 - 3, shift=False)

        ax[0].imshow(drop_image, cmap="gray")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].plot(predicted_x, predicted_z, label="Theoretical")
        ax[1].plot(t_x, t_z, label="Experimental")
        # ax[1].legend()
        ax[1].set_aspect("equal")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("z")

        prediction_error = t_x - f(t_z)
        indexes = argsort(t_z)

        t_z = t_z[indexes]
        prediction_error = prediction_error[indexes]

        z_bin = []
        prediction_error_bin = []
        bin_index = int(0.01 * len(t_z))

        zi_bin = []
        for i, zi in enumerate(t_z):
            zi_bin.append(zi)
            if i % bin_index == 0:
                z_bin.append(mean(zi_bin))
                zi_bin = []

        prediction_i_error_bin = []
        for i, error_i in enumerate(prediction_error):
            prediction_i_error_bin.append(error_i)
            if i % bin_index == 0:
                prediction_error_bin.append(mean(prediction_i_error_bin))
                prediction_i_error_bin = []

        ax[2].scatter(z_bin, prediction_error_bin)
        ax[2].set_ylabel("Residual")
        ax[2].set_xlabel("z")

        plt.tight_layout()
        plt.savefig(output_dir / f"{file.stem}_{bond_number}.png")

    return bond_number


def traditional(root_dir_path, output_dir=None, return_results=False):
    # Process from image maps directly
    root_dir = Path(root_dir_path)
    files = listdir(root_dir)
    cores = cpu_count()

    results = []
    batch = []
    i = 0
    for file in tqdm(files, total=len(files), desc="Traditional"):
        batch.append([root_dir / file, output_dir])
        i += 1
        if i == cores:
            # Process in parallel
            with Pool(cores) as p:
                r = p.map(trad_predict, batch)
                if return_results:
                    results.extend(list(zip(batch, r)))
            batch = []
            i = 0
    if batch:
        with Pool(cores) as p:
            r = p.map(trad_predict, batch)
            if return_results:
                results.extend(list(zip(batch, r)))

    if return_results:
        return results


def predict_other_parameters(root_dir_path, parameter, return_results=True):
    files = listdir(root_dir_path)
    results = []
    for file in tqdm(files, total=len(files), desc="Traditional"):

        drop_profile = loadtxt(root_dir_path / file, delimiter=",")

        # Calculate drop parameter
        volume = 0
        area = 0
        max_drop_radius = 0
        cap_diameter = 0

        for i in range(1, len(drop_profile)):
            row = drop_profile[i]
            x, z = row
            x0, z0 = drop_profile[i - 1]

            delta_z = abs(z - z0)
            delta_x = abs(x - x0)
            delta_s = sqrt(delta_x ** 2 + delta_z ** 2)

            volume += pi * (x ** 2) * delta_z
            area += 2 * pi * x * delta_s
            cap_diameter = 2 * x

            if x > max_drop_radius:
                max_drop_radius = x

        volume = volume / (pi * cap_diameter * (max_drop_radius ** 2))
        area = area / (pi * cap_diameter * max_drop_radius)
        cap_diameter = cap_diameter / max_drop_radius

        if return_results:

            if parameter == "cap_diameter":
                results.append([file, cap_diameter])

            elif parameter == "volume":
                results.append([file, volume])

            elif parameter == "area":
                results.append([file, area])

            else:
                raise ValueError(f"Parameter: {parameter} not defined")

    if return_results:
        return results


def cnn(root_dir_path, parameter="bond_number", return_results=True):
    # Number of files to process at once
    batch_size = 2000
    root_dir = Path(root_dir_path)
    files = listdir(root_dir)

    results = []
    batch = []
    i = 0
    for file in tqdm(files, desc="CNN"):
        batch.append(root_dir / file)
        i += 1
        if i == batch_size:
            r = default_predict(img_files=batch, parameter=parameter).tolist()
            if return_results:
                results.extend(list(zip(batch, r)))
            batch = []
            i = 0
    if batch:
        r = default_predict(img_files=batch, parameter=parameter).tolist()
        if return_results:
            results.extend(list(zip(batch, r)))

    if return_results:
        return results


if __name__ == "__main__":
    # For timing metrics

    # Predict Bo from traditional method
    start_time = default_timer()
    traditional("data/drop_profiles", return_results=False)
    end_time = default_timer()
    traditional_time = end_time - start_time

    # Predict Bo from CNN method
    start_time = default_timer()
    cnn("data/drop_images/control", return_results=False)
    end_time = default_timer()
    cnn_time = end_time - start_time

    print(f"Time needed for traditional method: {traditional_time}")
    print(f"Time needed for CNN method: {cnn_time}")
