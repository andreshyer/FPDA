from pathlib import Path
from os import mkdir, listdir
from os.path import exists
from json import load, dump
from time import sleep
from shutil import rmtree
from random import shuffle, uniform, choice

from tqdm import tqdm
from numpy import savetxt, loadtxt
from cv2 import imwrite
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib import use

from utils.drop_profiles import new_drop_profile
from utils.drop_images import new_drop_image
from utils.keras_CNN import train, test
from utils.backends.misc import parameters_to_name, name_to_parameters

# Choose what backend to use matplotlib with
use('TkAgg')
    

def generate_data(c):
    
    # Make new directory to store all data
    root_directory = Path(c["directory"])
    mkdir(root_directory)

    # Generate new drop profiles
    drop_profile_directory = root_directory / "drop_profiles"
    mkdir(drop_profile_directory)
    for _ in tqdm(range(int(c["drop_profiles"]["N"])), desc="Generating Drop Profiles"):
        drop_parameters, drop_profile = new_drop_profile(bond_number=uniform(*c["drop_profiles"]["bond_number_range"]), 
                                                         max_worthington_number=uniform(c["drop_profiles"]["min_worthington_number"], 1), 
                                                         delta_s=c["drop_profiles"]["delta_s"])
        file = parameters_to_name(drop_parameters, suffix=".csv")
        savetxt(drop_profile_directory / file, drop_profile, delimiter=",")

    # Create images for models
    drop_profiles_paths = listdir(drop_profile_directory)
    drop_img_directory = root_directory / "drop_images"
    mkdir(drop_img_directory)
    for model in c["models"]: 

        model_img_directory = drop_img_directory / model["name"]
        mkdir(model_img_directory)

        for _ in tqdm(range(int(c["image_parameters"]["N"])), desc=f"Generating Drop Images for {model['name']}"):

            # Grab random drop profile
            drop_profile = choice(drop_profiles_paths)
            drop_profile = drop_profile_directory / drop_profile
            drop_parameters = name_to_parameters(drop_profile)
            drop_profile = loadtxt(drop_profile, delimiter=",")

            # Randomly select further parameters for image manipulations
            drop_parameters["rotation"] = uniform(*c["image_parameters"]["rotation_range_degrees"])
            drop_parameters["drop_scale"] = uniform(*c["image_parameters"]["drop_scale_range"])
            drop_parameters["rel_capillary_height"] = uniform(*model["rel_capillary_height_range"])
            if uniform(0, 1) <= model["prob_capillary_zero"]:
                drop_parameters["rel_capillary_height"] = 0

            # Generate drop images, calculating the relative drop radius in the process
            relative_drop_radius, drop_image = new_drop_image(drop_profile=drop_profile, 
                                                              img_size_pix=c["image_parameters"]["img_size_pix"],
                                                              rotation=drop_parameters["rotation"],
                                                              drop_scale=drop_parameters["drop_scale"],
                                                              noise=model["noise"],
                                                              rel_capillary_height=drop_parameters["rel_capillary_height"],
                                                              above_apex=model["only_keep_drop_above_apex"],
                                                              delta_s=c["drop_profiles"]["delta_s"]
            )
            drop_parameters["drop_radius"] = relative_drop_radius
            file = parameters_to_name(drop_parameters, suffix=".png")
            imwrite(str(model_img_directory / file), drop_image)

    # Define split percents
    train_percent = c["model_parameters"]["train_percent"]
    val_percent = c["model_parameters"]["val_percent"]
    test_percent = 1 - train_percent - val_percent
    N = c["image_parameters"]["N"]
    train_percent, val_percent, test_percent = int(train_percent * N), int(val_percent * N), int(test_percent * N)

    # Split datasets
    root_model_directory = root_directory / "models"
    mkdir(root_model_directory)
    for model in tqdm(c["models"], desc="Splitting datasets into train/val/test sets"): 
        model_directory = root_model_directory / model["name"]
        mkdir(model_directory)

        # Gather img paths for a particular model
        model_img_directory = drop_img_directory / model["name"]
        img_paths = []
        for img_path in model_img_directory.iterdir():
            img_paths.append(img_path)

        # Split
        shuffle(img_paths)
        train_imgs = img_paths[:train_percent]
        val_imgs = img_paths[train_percent:train_percent+val_percent]
        test_imgs = img_paths[train_percent+val_percent:]
        
        # Save Split
        model_split = dict(
            train=train_imgs,
            val=val_imgs,
            test=test_imgs,
        )
        with open(model_directory / "split.json", "w") as f:
            dump(model_split, f, default=str, indent=4)

    # Train models
    y_keys = ["bond_number", "volume", "area", "cap_diameter" , "drop_radius"]
    for model in c["models"]: 

        model_path=root_model_directory / model["name"]

        # Train model, collecting y scaler and loss vs epochs
        keras_model, history_data, y_scaler = train(model_path=model_path, 
                                                    y_keys=y_keys,
                                                    epochs=c["model_parameters"]["epochs"],
                                                    batch_size=c["model_parameters"]["batch_size"]
                                                    )

        # Save data
        with open(model_path / "y_scaler.save", "wb") as f:
            joblib_dump(y_scaler, f)
        with open(model_path / "history.json", "w") as f:
            dump(history_data, f)
        keras_model.save(model_path / "model")

    # Test models
    for model in c["models"]:

        model_path=root_model_directory / model["name"]
        
        # Open data
        keras_model = load_model(model_path / "model")
        with open(model_path / "y_scaler.save", "rb") as f:
            y_scaler = joblib_load(f)
        with open(model_path / "split.json", "r") as f:
            y_files = load(f)["test"]

        # Generate Predicted vs Actual data
        normalized_pva, pva = test(model=keras_model, y_keys=y_keys, y_scaler=y_scaler, y_files=y_files, batch_size=c["model_parameters"]["batch_size"])

        # Save pva data
        
        # Save individual PVA plots
        model_pva_path = model_path / "pva_graphs"
        mkdir(model_pva_path)
        for i, key in enumerate(y_keys):
            y = []
            y_pred = []
            y_norm = []
            y_pred_norm = []

            for j in range(len(pva)):
                y.append(pva[j][0][i])
                y_pred.append(pva[j][1][i])
                y_norm.append(normalized_pva[j][0][i])
                y_pred_norm.append(normalized_pva[j][1][i])

            plt.scatter(y, y_pred)
            plt.savefig(model_pva_path / f"{key}_pva.png")
            plt.close()

            plt.scatter(y_norm, y_pred_norm)
            plt.savefig(model_pva_path / f"{key}_pva_normalized.png")
            plt.close()


if __name__ == "__main__":

    # # Test config files with small N, verify config file is correctly configured
    # print("\n--------------------------------------------------------------------------------------")
    # print("Running test on config file with small N to verify config file is correctly configured.")
    # print("--------------------------------------------------------------------------------------\n")
    # sleep(2)
    # config = Path("config.json")
    # with open(config, "r") as f:
    #     config = load(f)
    # temp_config = config.copy()
    # temp_config["drop_profiles"]["N"] = 10
    # temp_config["image_parameters"]["N"] = 10
    # temp_config["model_parameters"]["batch_size"] = 1
    # temp_config["model_parameters"]["epochs"] = 1
    # generate_data(temp_config)
    # rmtree(temp_config["directory"])

    # Generate data on real config file
    print("\n--------------------------------------------------------------------------------------")
    print("Generating data on real configuration file.")
    print("--------------------------------------------------------------------------------------\n")
    sleep(2)
    config = Path("config.json")
    with open(config, "r") as f:
        config = load(f)
    generate_data(config)
