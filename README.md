# SpectFbCalc
### Tools for the calculation of radiative feedbacks and sensitivities, both broad-band and spectrally-resolved

## Description
This software is designed to calculate, analyse and visualize radiance anomalies and radiative feedbacks using synthetic kernels and variations in climate model parameters. 

SpectFbCalc is a Python-based tool designed for calculating radiative anomalies and climate feedbacks using synthetic kernels and climate model outputs. The software facilitates the analysis of radiative biases and explores the impact of parameter tuning on climate model performance and sensitivity.

## Motivation
1. Analyze how variations in individual climate variables influence radiative biases and climate feedbacks. 
2. Improve the tuning of climate models and enhance the understanding of biases in simulated radiances.

## Features
- Kernel-based radiative feedback computation: uses both broadband and spectral kernels
- Compatibility with CMIP6 standard CMOR output and specific output from EC-Earth3/4 simulations
- Configurable setup: allows defining paths, kernels, and datasets using a YAML configuration file
- Data handling and standardization: functions for reading and standardizing datasets
- Modular architecture: core calculations are implemented in reusable functions

## Repository Structure
```
SpectFbCalc/
│-- ClimTools/                  # External library for climate data analysis
│-- spectfbcalc/
│   ├── spectfbcalc_lib.py      # Core library with fundamental functions
│   ├── config.yaml             # Configuration file for data and parameters
│   ├── __init__.py             # Module initialization
│-- environment.yml              # Conda environment file for dependencies
│-- test_fbcalc.ipynb      # Jupyter notebook with test examples
│-- install.sh                   # Installation script
│-- README.md                    # Project documentation
```

## Getting Started
### Install Conda
Install Conda (Miniconda/Mamba recommended): [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
### Clone the Repository
```bash
git clone https://github.com/fedef17/SpectFbCalc/
cd SpectFbCalc
```
### Set up the Environment
```bash
conda env create -f environment.yml
conda activate spectfbcalc
```
### Run the install script
```bash
bash install.sh
```

## Usage
### Running the Notebook
This tool can be used in Jupyter Notebook or directly in VS Code.
To execute the code interactively:
1. Activate the environment:
```bash
conda activate spectfbcalc
```
2. Open Jupyter Notebook and run:
```bash
jupyter notebook
```
3. Navigate to **`test_spectfbcalc.ipynb`** and execute the cells to see an example usage.
### Initial Setup in the Notebook
Before running the core functions for calculating anomalies and feedbacks, create a **`config.yaml/`** file in the **`spectfbcalc/`** folder by copying **`config_example.yaml/`** and modifying paths and configurations based on your purpose: 
```bash
cp spectfbcalc/config_example.yaml spectfbcalc/config.yaml
```
Then initialize the following parameters:
```bash
config = "path/to/config/file/config.yaml"
kernel = ""
standard_dict = sfc.standard_names
```
### The spectfbcalc_lib
The **`spectfbcalc_lib`** contains several functions essential for data processing and radiative feedback calculations: 
- Kernel handling: functions to load and manage radiative kernels. 
- Dataset reading: function for reading and loading datasets and ensuring compatibility with the required format. 
- Variable name standardization: a function to standardize variable names based on a predefined dictionary. **It is recommended to check the dataset beforehand to identify which variables need renaming when prompted by the program**. 
- Core functions for radiance anomaly calculation: these functions compute radiance anomalies under both clear-sky and all-sky conditions. If the datasets are already pre-processed, they can be used directly; otherwise, wrapped functions are available to handle data reading, opening and standardization automatically. 
- Feedback calculations: includes functions for computing the following radiative feedbacks: Planck, Albedo, Lapse-rate, Water-vapor and Cloud

### What the user can do 
The user can choose what to compute according to its need:
- individual radiance variations (using each core or wrapper version).
- all radiance variations at once (using the **`calc_anoms_wrapper`** or **`calc_anoms`** function), but not that due to clouds because the calculation for cloud feedback is indirect and all other feedbacks are needed to obtain it.
- all feedback at once without the cloud feedback (using the **`calc_fb_wrapper`** or **`calc_fb`** function) or with the cloud feedback (using the **`feedbac_cloud_wrapper`** or **`feedback_cloud`** function)

### Configuration File (config.yaml)
The software is designed to handle data from both EC-Earth3 and EC-Earth4. Currently, it supports two kernel types:
1. **ERA5-based** kernels (Soden et al., 2008) 
2. **Huang** kernels (Huang et al., 2023)
Both are broadband kernels, but spectral kernels will be supported in future versions.
#### File Paths
Paths should be specified appropriately based on how datasets are organized: 
- Use * for standard wildcards. 
- Use **/* if data is organized into subdirectories. 
#### Special File Path Cases 
- pl paths: Used when pressure level variables are stored in separate files because for the calculation of some feedback, it is necessary to loada atmospheric temperature dataset with pressure levels ('plev') as a coordinate.  
- experiment_dataset2: Defined when additional datasets are required for comparison. 
#### Climatology Settings 
- use_climatology = True: to compute the monthly averaged climatology to calculate the anomaly.
- use_climatology = False: to coompute the 21-years running mean climatology to calculate the anomaly.
#### Output dimension
- use_ds_climatology = True: The anomaly is computed as a single averaged value over the time dimension. 
- use_ds_climatology = False: The anomaly is computed for each available year in the dataset.

## Future Goals 
- Implementation of advanced visualization tools for spectral anomalies and radiative feedbacks.
- Support for additional simulation configurations.
- Development of automated testing for validation.
- Improved documentation (potentially with Sphinx).