
from os import listdir
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer
from random import choice
from numpy import loadtxt
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from utils.drop_profiles import new_drop_profile
from utils.keras_CNN import default_predict


def trad_predict(drop_profile):

    # Load drop profile
    drop_profile = loadtxt(drop_profile, delimiter=",")
    true_x, true_z = drop_profile[:, 0], drop_profile[:, 1]

    def error(Bo):

        # Predict Bo
        _, predicted_drop_profile = new_drop_profile(bond_number=Bo, max_worthington_number=1, delta_s=1e-3)
        pred_x, pred_z = predicted_drop_profile[:, 0], predicted_drop_profile[:, 1]

        # Fit function
        f = UnivariateSpline(pred_z, pred_x, s=0)

        # Calculate error between profiles
        error = sum(abs(true_x - f(true_z)))
        return error

    bond_number = minimize(error, 0.2, bounds=[(0.1, 0.35)], method='L-BFGS-B', options={'maxiter': 100}).x[0]
    return bond_number


def traditional():
    # Process from image maps directly
    root_dir = Path("data/drop_profiles")
    files = listdir(root_dir)
    cores = cpu_count()
    N = int(1e6)
    batch = []
    i = 0
    for _ in tqdm(range(N), desc="Traditional"):
        # Randomly choose drop point map
        drop_profile = root_dir / choice(files)
        batch.append(drop_profile)
        i += 1
        if i == cores:
            # Process in parallel
            with Pool(cores) as p:
                p.map(trad_predict, batch)
            batch = []
            i = 0


def cnn():
    # Number of files to process at once
    batch_size = 2000
    root_dir = Path("data/drop_images/control")
    files = listdir(root_dir)
    batch = []
    i = 0
    for file in tqdm(files, desc="CNN"):
        batch.append(root_dir / file)
        i += 1
        if i == batch_size:
            default_predict(img_files=batch, parameter="bond_number")
            batch = []
            i = 0



if __name__ == "__main__":
    
    # # Predict Bo from traditional method
    start_time = default_timer()
    traditional()
    end_time = default_timer()
    traditional_time = end_time - start_time

    # Predict Bo from CNN method
    # start_time = default_timer()
    # cnn()
    # end_time = default_timer()
    # cnn_time = end_time - start_time

    # print(f"Time needed for traditional method: {traditional_time}")
    # print(f"Time needed for CNN method: {cnn_time}")
