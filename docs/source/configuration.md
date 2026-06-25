# Configuration File (config.yaml)
The software is designed to handle data from CMIP6/CMIP7 models. It supports three kernel types:
1. **Huang** kernels [^1]
2. **ERA5-based** kernels [^2]
3. **SPECTRAL** kernels [^3]

Where the first two are broadband.

## File Paths
Paths should be specified appropriately based on how datasets are organized: 
- Use * for standard wildcards. 
- Use **/* if data is organized into subdirectories. 

## Paramaters
- **`anomaly_method`** : the tool allows different methods for calculating anomalies and handling climatology. Options are *climatology* (monthly averaged) or *running_mean*.
- **`use_atm_mask`** : if **`True`**, applies the stratospheric mask (Reichler algorithm).
- **`save_pattern`** : if **`True`**, saves the full spatial anomaly patterns alongside the global means.
- **`num_year_regr`** : number of years to group into chunks for the linear regression feedback calculation.
- **`time_range_clim`** / **`time_range_exp`** : restricts the analysis to a specific temporal window. Leave **`time_range_exp`** empty to automatically match the reference dataset's length.

[^1]: Huang, H., & Huang, Y. (2023). Radiative sensitivity quantified by a new set of radiation flux kernels  based on the ECMWF Reanalysis v5 (ERA5). Earth System Science Data, 15(7), 3001–3021. https://doi.org/10.5194/essd-15-3001-2023
[^2]: Soden, B. J., Held, I. M., Colman, R., Shell, K. M., Kiehl, J. T., & Shields, C. A. (2008). Quantifying Climate Feedbacks Using Radiative Kernels. Journal of Climate, 21(14), 3504–3520. https://doi.org/https://doi.org/10.1175/2007JCLI2110.1
[^3]: Della Fera, S., Fabiano, F., Raspollini, P., Ridolfi, M., Von Hardenberg, J., & Cortesi, U. (2025). Reproducing and Attributing IASI Radiance Trends with EC-Earth Climate Model Simulations. Journal of Climate, 38(23), 6943-6959. https://doi.org/10.1175/JCLI-D-25-0034.1 