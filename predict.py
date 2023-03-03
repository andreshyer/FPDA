from pathlib import Path

from numpy import where, mean, arange, sqrt, argsort, unique
from matplotlib import pyplot as plt
from matplotlib import use
from scipy.interpolate import CubicSpline

from utils.drop_profiles import new_drop_profile
from utils.drop_images import extract_drop_profile, new_drop_image

import cv2

use('TkAgg')


def calc_true_bo(drop_profile):

    # Error function to calculate maximum distance between points
    def calc_error(b):

        # Draw new drop profile
        _, test_drop_profile = new_drop_profile(bond_number=b, max_worthington_number=1, delta_s=1e-3)
        test_drop_profile = test_drop_profile / test_drop_profile[:, 0].max()

        # Ensure that drops have the same height
        idx, _ = where(test_drop_profile > drop_profile[:, 1].max())
        try:
            cutoff_index = min(idx)
        except ValueError:
            return 1e6
        test_drop_profile = test_drop_profile[:cutoff_index]

        # Drop duplicate x in test drop
        x_values, unique_indices = unique(test_drop_profile[:, 1], return_index=True)
        sorted_data = test_drop_profile[unique_indices, :]

        # Sort based on x
        sort_indices = argsort(test_drop_profile[:, 1])
        sorted_data = test_drop_profile[sort_indices]

        # Resample the fitted data onto the same grid as the real data
        f = CubicSpline(test_drop_profile[:, 1], test_drop_profile[:, 0])
        y_fitted_resampled = f(drop_profile[:, 1])

        # Calculate the maximum difference between the real and fitted data points
        error = mean(abs(drop_profile[:, 0] - y_fitted_resampled))

        return error
    
    # Range to find bond number
    br = [0.1, 0.5]

    # Calculate what bond number give the correct value
    max_N = 100
    counter = 0
    while abs(br[1] - br[0]) > 1e-4:
        b0_error = calc_error(br[0])
        b1_error = calc_error(br[1])
        
        half_distance = (br[1] + br[0]) / 2
        if b1_error > b0_error:
            br[1] = half_distance
        else:
            br[0] = half_distance

        counter += 1
        if counter >= max_N:
            break

    return mean(br)


def main():
    for img in Path("temp_drop").iterdir():
        
        if "EtOH" in img.name:
            pass
        else:
            pass
            
        drop_profile = extract_drop_profile(img, thres=100)
        bond_number = calc_true_bo(drop_profile=drop_profile)

        # print(drop_profile)
        # _, drop_profile = new_drop_profile(bond_number=0.15, max_worthington_number=0.7, delta_s=1e-3)
        _, drop_image = new_drop_image(drop_profile=drop_profile, img_size_pix=128, rotation=0, drop_scale=0.98, noise=0, rel_capillary_height=0, above_apex=False, delta_s=None)

        # print(drop_image)

        cv2.imwrite('dev.png', drop_image)
        break

        # _, calc_drop_profile = new_drop_profile(bond_number=bond_number, max_worthington_number=0.7, delta_s=1e-3)
        # calc_drop_profile = calc_drop_profile / calc_drop_profile[:, 0].max()
        # idx, _ = where(calc_drop_profile > drop_profile[:, 1].max())
        # cutoff_index = min(idx)
        # calc_drop_profile = calc_drop_profile[:cutoff_index]

        # fig, axs = plt.subplots()
        
        # axs.scatter(drop_profile[:, 0], drop_profile[:, 1])
        # axs.scatter(calc_drop_profile[:, 0], calc_drop_profile[:, 1])
        # axs.set_aspect("equal")

        # plt.title(f"{bond_number}: {img.name}")
        # plt.show()
        # plt.close()


if __name__ == "__main__":
    main()

