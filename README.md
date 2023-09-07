# FPDA

## Fast Pendant Drop Analysis

This is a repo that is designed for predicting the following parameters,
given a drop profile or drop image of a pendant drop. 

`["bond_number", "volume", "area", "cap_diameter", "drop_radius"]`

Each of these parameters have been made non-dimensionless by `$R_0$` and `$D_n$`,
which is the radius of curvature at the drop apex and the diameter of the capillary tip
respectfully. The dimensionless properties that are predicted are scaled as follows.

`$Bond Number = \frac{\Delta \rho g R_0^2}{\gamma}$`

`$Volume = \frac{V_d}{\pi D_n R_0^2}$`

`$Area = \frac{A_d}{\pi D_n R_0}$`

`$Capillary Diameter = \frac{D_n}{R_0}$`

`$Radius of Drop at Apex = \frac{R_0}{c}$`

## Predicting

An example of how to predict properties can be seen in `predict.py`,
although the current prediction script will not work without data.
But, it does show the structure of predicting using the code provided.

## Full dataset

This repo is meant to be a minimum example to allowing for predicting drop properties,
with pre-trained models. The full dataset can be accessed one 
[Dropbox](https://www.dropbox.com/scl/fo/et7ujrw9nbjwn87ts4qao/h?rlkey=dynzo8riobe2k9hf8q9df76cl&dl=0), 
where the data is compressed. 
The data folder in this repo should be replaced with the decompressed data folder from Dropbox.
Alternatively, all the data can be generated from scratch using `generate_data.py`,
which also trains new models from the generated data. 
If data is being generated from scratch, 
properties can be modified in `config.json`.

## Environment

The easiest way to get a working environment is to use anaconda or miniconda,
running the following in the `environment` directory using the terminal (Linux/macOS) 
or Anaconda Prompt (Windows).

`conda env create -f drop.yml`

This will create a conda environment that is named `drop`. 
These are the full requirements listed in `drop.yml`.

```yml
name: drop
channels:
  - defaults
dependencies:
  - python<=3.11
  - tqdm
  - numpy
  - matplotlib
  - scipy
  - scikit-learn
  - pip
  - pip:
      - tensorflow
      - opencv-python
```
