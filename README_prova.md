# EC-Earth Model performance validation thanks to radiance anomalies analysis and radiative feedbacks calculation 
## Description
This software is designed to calculate, analyse and visualize radiance anomalies and radiative feedbacks using synthetic kernels and variations in climate model parameters. 
(It aims to assist in validating the EC-Earth climate model, particularly through comparison with IASI measurements and using the σ-IASI radiative transfer model.)
SpectFbCalc is a Python-based tool designed for calculating radiative anomalies and feedbacks using synthetic kernels and climate model outputs. The software facilitates the analysis of radiative biases and explores the impact of parameter tuning on climate model performance, particularly within the EC-Earth climate model.

## Motivation
will ultimately provide insights into improving the tuning of climate models and understanding biases in simulated radiances.
The project aims to trace spectral anomalies in radiances back to their contributing atmospheric variables.  
By leveraging outputs from the EC-Earth model this work seeks to: 
1. Reverse-engineer the simulator's results.
2. Analyze how variations in individual climate variables influence spectral biases. 
3. Improve the tuning of climate models and enhance the understanding of biases in simulated radiances.

## Features
- Kernel-based radiative feedback computation: Uses both broadband and spectral kernels togheter with anomalies to analyze radiative feedbacks and radiance anomalies.
- Data handling and standardization: Functions for reading and standardizing datasets.
- Modular architecture: Core calculations are implemented in reusable functions for flexibility and maintainability.
- Compatibility with EC-Earth simulations: Designed to assess radiative anomalies from EC-Earth experiments. 
- Collaborative development workflow: A structured git branch-based system for version control and contributions.

## Repository Structure
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


## Future Goals 
- Implementation of advanced visualization tools for spectral anomalies and radiative feedbacks.
- Support for additional simulation configurations.
- Development of automated testing for validation.
- Improved documentation (potentially with Sphinx).