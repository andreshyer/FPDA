
from os import listdir
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer
from numpy import array, loadtxt, where
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from matplotlib import pyplot as plt

from utils.drop_profiles import new_drop_profile
from utils.keras_CNN import default_predict


def trad_predict(y):

    file, output_dir = y

    # Load drop profile
    drop_profile = loadtxt(file, delimiter=",")
    true_x, true_z = drop_profile[:, 0], drop_profile[:, 1]
    true_max_x = true_x.max()

    def error(Bo):

        # Predict Bo
        _, predicted_drop_profile = new_drop_profile(bond_number=Bo[0], max_worthington_number=1, delta_s=1e-3)
        pred_x, pred_z = predicted_drop_profile[:, 0], predicted_drop_profile[:, 1]
        pred_max_x = pred_x.max()

        # Fit function
        f = UnivariateSpline(pred_z, pred_x, s=0)

        # Scale drop profile
        t_x, t_z = true_x * (pred_max_x / true_max_x), true_z * (pred_max_x / true_max_x)

        # Calculate error between profiles
        error_mse = sum(abs(t_x - f(t_z)))
        return error_mse

    bond_number = minimize(error, array([0.35]), bounds=[(0.1, 0.4)], method='L-BFGS-B', options={'maxiter': 100}).x[0]
    bond_number = round(bond_number, 3)

    if output_dir:
        _, predicted_drop_profile = new_drop_profile(bond_number=bond_number, max_worthington_number=1, delta_s=1e-3)
        pred_x, pred_z = predicted_drop_profile[:, 0], predicted_drop_profile[:, 1]
        pred_max_x = pred_x.max()

        # Fit function
        f = UnivariateSpline(pred_z, pred_x, s=0)

        t_x, t_z = true_x * (pred_max_x / true_max_x), true_z * (pred_max_x / true_max_x)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(pred_x, pred_z, label="Theoretical")
        ax[0].plot(t_x, t_z, label="Experimental")
        ax[0].legend()
        ax[0].set_aspect("equal")
        ax[0].set_title(bond_number)

        error = t_x - f(t_z)
        ax[1].scatter(t_z, error)
        ax[1].set_title("Residual")

        plt.tight_layout()
        plt.savefig(output_dir / f"{file.stem}_{bond_number}.png")

    return bond_number


def traditional(root_dir_path, output_dir=None, keep_results=False):
    # Process from image maps directly
    root_dir = Path(root_dir_path)
    files = listdir(root_dir)
    cores = cpu_count()
    N = len(files)

    if keep_results:
        results = []

    batch = []
    i = 0
    for file in tqdm(files, total=N, desc="Traditional"):
        batch.append([root_dir / file, output_dir])
        i += 1
        if i == cores:
            # Process in parallel
            with Pool(cores) as p:
                r = p.map(trad_predict, batch)
                if keep_results:
                    results.extend(list(zip(batch, r)))
            batch = []
            i = 0
    if batch:
        with Pool(cores) as p:
            r = p.map(trad_predict, batch)
            if keep_results:
                results.extend(list(zip(batch, r)))

    if keep_results:
        return results


def cnn(root_dir_path, keep_results=False):
    # Number of files to process at once
    batch_size = 2000
    root_dir = Path(root_dir_path)
    files = listdir(root_dir)

    if keep_results:
        results = []

    batch = []
    i = 0
    for file in tqdm(files, desc="CNN"):
        batch.append(root_dir / file)
        i += 1
        if i == batch_size:
            r = default_predict(img_files=batch, parameter="bond_number").tolist()
            if keep_results:
                results.extend(list(zip(batch, r)))
            batch = []
            i = 0
    if batch:
        r = default_predict(img_files=batch, parameter="bond_number").tolist()
        if keep_results:
            results.extend(list(zip(batch, r)))

    if keep_results:
        return results


if __name__ == "__main__":
    
    # Predict Bo from traditional method
    start_time = default_timer()
    traditional("data/drop_profiles")
    end_time = default_timer()
    traditional_time = end_time - start_time

    # Predict Bo from CNN method
    start_time = default_timer()
    cnn("data/drop_images/control")
    end_time = default_timer()
    cnn_time = end_time - start_time

    print(f"Time needed for traditional method: {traditional_time}")
    print(f"Time needed for CNN method: {cnn_time}")
