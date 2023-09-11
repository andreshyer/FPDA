from pathlib import Path

from cv2 import imwrite

from matplotlib import use
from numpy import loadtxt, mean, std, linspace

from utils.drop_images import new_drop_image
from predict import traditional, cnn, predict_other_parameters

use('TkAgg')


def csv_to_images():
    root_dir = Path("drops/real_drop_profiles")
    output_dir = Path("drops/processed")
    rotations = linspace(0, 5, 5)

    for file_i in root_dir.iterdir():
        data = loadtxt(file_i, delimiter=",")
        for rot in rotations:
            img = new_drop_image(data, 128, rot, 1, 0, 0, False, delta_s=1e-3)
            _, drop_image = img
            new_file = file_i.stem + f"_{rot}.png"
            imwrite(str(output_dir / new_file), drop_image)


if __name__ == "__main__":
    csv_to_images()

    true_bond_numbers = traditional("drops/real_drop_profiles",
                                    Path("drops/converged_traditional_Bo_predictions"),
                                    return_results=True)
    predicted_bond_numbers = cnn("drops/processed", return_results=True)

    for f in true_bond_numbers:
        file = f[0][0]
        file_name = file.stem
        true_Bo = float(f[1])

        predicted_Bo = []
        for g in predicted_bond_numbers:
            predicted_file = g[0]
            predicted_file_name = predicted_file.stem

            if predicted_file_name.startswith(file_name):
                predicted_Bo.append(float(g[1]))

        print(f"({file_name}) Actual Bo: {true_Bo} | Predicted Bo: {mean(predicted_Bo)} | "
              f"Error Bo: {abs(mean(predicted_Bo) - true_Bo)} | STD Bo: {std(predicted_Bo)}")

    other_parameters = ["cap_diameter", "volume", "area"]
    for parameter in other_parameters:
        true_values = predict_other_parameters(Path("drops/real_drop_profiles"), parameter=parameter, return_results=True)
        predicted_values = cnn("drops/processed", parameter=parameter, return_results=True)

        for f in true_values:
            file_name = f[0].split(".")[0]
            true_Bo = float(f[1])

            predicted_Bo = []
            for g in predicted_values:
                predicted_file = g[0]
                predicted_file_name = predicted_file.stem

                if predicted_file_name.startswith(file_name):
                    predicted_Bo.append(float(g[1]))

            print(f"({file_name}) Actual {parameter}: {true_Bo} | Predicted {parameter}: {mean(predicted_Bo)} | "
                  f"Error {parameter}: {abs(mean(predicted_Bo) - true_Bo)} | STD {parameter}: {std(predicted_Bo)}")
