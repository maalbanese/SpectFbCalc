# EC-Earth Model performance validation thanks to radiance anomalies analysis and radiative feedbacks calculation 
## Description
This software is designed to calculate, analyse and visualize radiance anomalies and radiative feedbacks using synthetic kernels and variations in climate model parameters. 
(It aims to assist in validating the EC-Earth climate model, particularly through comparison with IASI measurements and using the σ-IASI radiative transfer model.)
SpectFbCalc is a Python-based tool designed for calculating radiative anomalies and feedbacks using synthetic kernels and climate model outputs. The software facilitates the analysis of radiative biases and explores the impact of parameter tuning on climate model performance, particularly within the EC-Earth climate model.

## Motivation
The project aims to trace spectral anomalies in radiances back to their contributing atmospheric variables.  
By leveraging outputs from the EC-Earth model this work seeks to: 
1. Reverse-engineer the simulator's results.
2. Analyze how variations in individual climate variables influence spectral biases. 
3. Improve the tuning of climate models and enhance the understanding of biases in simulated radiances.
Ultimately, the tool provide insights into improving the tuning of climate models and understanding biases in simulated radiances.

## Features
- Kernel-based radiative feedback computation: uses both broadband and spectral kernels togheter with anomalies to analyze radiative feedbacks and radiance anomalies.
- Data handling and standardization: functions for reading and standardizing datasets.
- Modular architecture: core calculations are implemented in reusable functions for flexibility and maintainability.
- Compatibility with EC-Earth simulations: designed to assess radiative anomalies from EC-Earth experiments and it handles multiple datasets from different EC-Earth versions.
- Configurable setup: allows defining paths, kernels, and datasets using a YAML configuration file.
- Collaborative development workflow: a structured git branch-based system for version control and contributions.

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
### Configuration File (config.yaml)
The software is designed to handle data from both EC-Earth3 and EC-Earth4. Currently, it supports two kernel types:
1. **ERA5-based** kernels (Soden et al., 2008) 
2. **Huang** kernels (Huang et al., 2023)
Both are broadband kernels, but spectral kernels will be supported in future versions.
#### File Paths
Paths should be specified appropriately based on how datasets are organized: 
- Use * for standard wildcards. 
- Use */ * if data is organized into subdirectories. 
#### Special File Path Cases 
- pl paths: Used when pressure level variables are stored in separate files. 
- experiment_dataset2: Defined when additional datasets are required for comparison. 
#### Climatology Settings 
- use_ds_climatology = True: The anomaly is computed as a single averaged value over the time dimension. 
- use_ds_climatology = False: The anomaly is computed for each available year in the dataset.

## Future Goals 
- Implementation of advanced visualization tools for spectral anomalies and radiative feedbacks.
- Support for additional simulation configurations.
- Development of automated testing for validation.
- Improved documentation (potentially with Sphinx).