# Framework architecture and preprocessing
## Principal components (*class*)

### Kernel
Class used for loading and standardising external radiative kernels. 
- Supported kernel types: HUANG [^1], ERA5 [^2], and SPECTRAL [^3].
- Surface pressure management when HUANG kernel are used **`recompute_dp`**. If the experiment contains surface pressure, the class recalculates the atmospheric layers based on it.

### Experiment
Class that maps and manages climate model datasets (both piControl control simulations and those with 4xCO2 forcing).
- Input data: organises NetCDF files using pattern dictionaries (**`file_dict`**).
- Lazy loading: utilises **`xr.open_mfdataset`** and configurable temporal chunking. 

It processes the dataset internally in three successive stages:
1. **`self.ds`**: remapped data and checks on the necessary variables (e.g. albedo calculation and the logarithm of specific humidity, *hus_log*).
2. **`self.ds_clim`**: reference climatology or running mean. 
3. **`self.ds_anom`**: anomalies ready for feedback calculation.


## Preprocessing function
The **`preprocess_data`** function coordinates all the steps required to ensure consistency between the model data and the radiative kernels.
Pipeline: 
1. Configuration and initialization: loading the configuration file containing all the paths and metadata to be used and mapping any non-standard variables where necessary. 
2. Horizontal remapping (to obtain a unique grid): using **`remap`** (custom functions based on **`climtools`**) or **`remap_cdo`** (which instead utilises CDO), the model data are interpolated into the target kernel’s horizontal geographic grid.   
3. Variable checking **`check_vars`**: 
    - **`check_alb`**: if alb variable is missing, it is calculated dynamically as *alb = rsds/rsus*.
    - **`check_hus_log`**: conversion of specific humidity to a logarithmic scale (**`hus_log`**) if required by the specific kernel (e.g. Huang).
    - **`Net_TOA`**: automatic calculation of the all-sky (*Net*) and clear-sky (*Net0*) TOA net radiative fluxes.
4. Vertical interpolation **`vertical_interp`**: conversion of pressure coordinates from Pascals to hPa (if necessary, by using **`check_vertical`**) followed by interpolation into the kernel’s vertical pressure levels (*plev*).
5.  Anomaly calculation: generation of anomaly datasets (*ds_anom*) by subtracting the control climatology. 

Users of the tool can then decide how to calculate the anomalies choosing the method:
 - *climatology*: uses monthly averaged climatology (optimised using flox if available, otherwise via lazy expansion with **`expand_clim`**);
 - *running_mean*: uses 21-years running mean climatology; `cit Zelinka for 21 years ???`


## Utilities and diagnostic support functions
There are some advanced functions implemented in addition to the principal pipeline:
1. Stratospheric masking (**`mask_strato`**): it generates a mask for atmospheric temperature data by implementing the Reichler algorithm. [^4]
2. Water vapour normalisation factors (**`Kq_fact`**): it calculates normalisation coefficient to normalize the water vapor kernel, which usually corresponds to a change in specific humidity due to an increase of atm temp by 1 K, keeping RH constant.

The entire process preserves dask’s lazy evaluation until the very last possible moment. The diagnostic functions such as **`check_lazy_loading`** monitor the computational state by tracking the theoretical size in megabytes and the number of chunks generated. Data are calculated explicitly (**`.compute()`**) only in the following cases:
- Saving remapped intermediate files to disk (**`save_remapped = True`**).
- Calculating fixed climatology matrices (*ds_clim*).
- Extracting coordinate vectors and surface pressure for loops on geographical nodes.
While these preprocessing modules are essential for standardizing datasets, SpectFbCalc is designed with a modular architecture that allows users to bypass these steps and interface directly with core functions if their inputs are already harmonized. 

[^1]: Huang, Y., Y. Xia, and X. Tan (2017), On the pattern of CO2 radiative forcing and poleward energy transport, J. Geophys. Res. Atmos., 122, 10,578–10,593. https://doi.org/10.1002/2017JD027221 
[^2]: Huang, H., & Huang, Y. (2023). Radiative sensitivity quantified by a new set of radiation flux kernels  based on the ECMWF Reanalysis v5 (ERA5). Earth System Science Data, 15(7), 3001–3021. https://doi.org/10.5194/essd-15-3001-2023
[^3]: Della Fera, S., Fabiano, F., Raspollini, P., Ridolfi, M., Von Hardenberg, J., & Cortesi, U. (2025). Reproducing and Attributing IASI Radiance Trends with EC-Earth Climate Model Simulations. Journal of Climate, 38(23), 6943-6959. https://doi.org/10.1175/JCLI-D-25-0034.1 
[^4]: Reichler, T., M. Dameris, and R. Sausen (2003), Determining the tropopause height from gridded data, Geophys. Res. Lett., 30, 20. https://doi.org/10.1029/2003GL018240