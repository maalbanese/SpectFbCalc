# SpectFbCalc
[![Documentation Status](https://readthedocs.org/projects/spectfbcalc/badge/?version=latest)](#) *Tools for the calculation of radiative feedbacks and sensitivities, both broad-band and spectrally-resolved.*

SpectFbCalc is a Python-based tool designed for calculating radiative anomalies and climate feedbacks using synthetic kernels and climate model outputs. It facilitates the analysis of radiative biases and climate feedbacks and explores the impact of parameter tuning on climate model performance and sensitivity.

## Features
- **Kernel-based computation:** Supports both broadband (Huang, ERA5) and spectral kernels.
- **CMIP Compatibility:** Works seamlessly with standard CMIP CMOR output.
- **Configurable Setup:** Fully driven by a YAML configuration file for easy experimental setups.
- **Dask Integration:** Preserves lazy evaluation for handling large model outputs efficiently.

## Documentation
For full installation instructions, usage tutorials, theoretical framework, and API reference, please visit our **[ReadTheDocs Documentation](#)**.

## Quick Install
```bash
git clone [https://github.com/fedef17/SpectFbCalc/](https://github.com/fedef17/SpectFbCalc/)
cd SpectFbCalc
conda env create -f environment.yml
conda activate spectfbcalc
bash install.sh
```