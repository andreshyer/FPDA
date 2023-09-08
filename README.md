# FPDA

## Fast Pendant Drop Analysis

This is a repo that is designed for predicting the following parameters,
given a drop profile or drop image of a pendant drop. 

`["bond_number", "volume", "area", "cap_diameter", "drop_radius"]`

Each of these parameters have been made dimensionless by `$R_0$` and `$D_n$`,
which is the radius of curvature at the drop apex and the diameter of the capillary tip
respectively. The dimensionless properties that are predicted are scaled as follows.

$$\text{Bond Number} = \frac{\Delta \rho g R_0^2}{\gamma}$$

$$\text{Volume} = \frac{V_d}{\pi D_n R_0^2}$$

$$\text{Area} = \frac{A_d}{\pi D_n R_0}$$

$$\text{Capillary Diameter} = \frac{D_n}{R_0}$$

The radius of the drop at the apex can also be predicted.

$$\text{Radius of Drop at Apex} = \frac{R_0 \text{ (m)}}{c \text{ (m)}} = \frac{R_0 \text{ (pix)}}{c \text{ (pix)}}$$

Where $c$ is the length scale, and can be calculated from any known reference point.

## Predicting

An example of how to predict properties from experimental drop profiles can be seen in `validate.py`.
The full generated dataset is predicted in`predict.py`,
although the full dataset must be downloaded first.

## Full dataset

This repo is meant to be a minimal example to allow for predicting drop properties,
with pre-trained models. The full dataset can be accessed on 
[Dropbox](https://www.dropbox.com/scl/fi/ykettma07e3ag53ywnso4/data.zip?rlkey=86ppnivp2xdr6jal4aqypnh1g&dl=0), 
where the data is compressed. 
The data folder in this repo should be replaced with the decompressed data folder from Dropbox.
Alternatively, all the data can be generated from scratch using `generate_data.py`,
which also trains new models from the generated data. 
If data is being generated from scratch, 
properties can be modified in `config.json`.

## Environment

The easiest way to get a working environment is to use Anaconda or Miniconda,
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
  - scikit-learn=1.2.0
  - pip
  - pip:
      - tensorflow
      - opencv-python
```
