#!/usr/bin/python
# -*- coding: utf-8 -*-

### Our library!

##### Package imports

from logging import config
import sys
import os
import glob
import re

import dask.delayed
import numpy as np
import xarray as xr
import pandas as pd

from climtools import climtools_lib as ctl
from matplotlib import pyplot as plt
import matplotlib.cbook as cbook
from scipy import stats
import pickle
import dask.array as da
import yaml
from difflib import get_close_matches
import dask
import psutil



######################################################################
### Functions
dask.config.set(scheduler='single-threaded')

def mytestfunction():
    print('test!')
    return

###### INPUT/OUTPUT SECTION: load kernels, load data ######
def load_spectral_kernel(cart_k: str, cart_out: str, version="v3"):
    """
    Loads and preprocesses spectral kernels for further analysis.

    Spectral kernels are expected as monthly climatologies split into
    individual NetCDF files (01–12), for clear-sky and all-sky conditions.
    The function reconstructs a monthly kernel with dimension `month`
    (NOT `time`), to ensure compatibility with downstream calls such as
    `anoms.groupby('time.month') * kernel`.

    Parameters
    ----------
    cart_k : str
        Base path containing spectral kernel subdirectories:
        - clear_sky_fluxes/
        - all_sky_fluxes/

    cart_out : str
        Output directory for pickled kernel objects.

    version : str, optional
        Kernel version string (default: "v3").

    Returns
    -------
    allkers : dict
        Dictionary with keys (tip, variable), where:
        - tip      ∈ {"clr", "cld"}
        - variable ∈ {"t", "ts", "wv_lw"}

        Each value is an xarray DataArray with dimension `month`.
    """

    import os
    import pickle
    import xarray as xr

    # mapping: filename tag → (output tag, subdirectory)
    tips = {
        "clear": ("clr", "clear_sky_fluxes"),
        "cld":   ("cld", "all_sky_fluxes"),
    }

    # variable name mapping: nc_name → (out_name, has_lev)
    vnams = {
        "temp_jac": ("t", True),
        "ts_jac":   ("ts", False),
        "wv_jac":   ("wv_lw", True),
    }

    allkers = {}
    vlevs = None

    for tip_raw, (tip_out, subdir) in tips.items():

        sky_dir = os.path.join(cart_k, subdir)
        ds_months = []

        # --- load monthly files ---
        for month in range(1, 13):
            fname = f"spectral_fluxes_kernel_longwave_{month:02d}_{tip_raw}_{version}.nc"
            fpath = os.path.join(sky_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing spectral kernel file: {fpath}")
            ds = xr.open_dataset(fpath)
            # explicitly tag the month (temporary time-like dimension)
            ds = ds.expand_dims(time=[month])
            ds_months.append(ds)

        # --- concatenate months ---
        kernels = xr.concat(ds_months, dim="time")
        if kernels.sizes.get("time", 0) != 12:
            raise ValueError("Spectral kernel must have exactly 12 months")

        # 🔑 CRUCIAL STEP:
        # convert time → month so downstream groupby('time.month') works
        kernels = (
            kernels
            .assign_coords(month=("time", kernels["time"].values))
            .swap_dims({"time": "month"})
            .drop_vars("time")
        )

        # --- extract kernels ---
        for vna_local, (vna_out, has_lev) in vnams.items():
            ker = kernels[vna_local]
            if has_lev:
                ker = ker.rename({"lev": "player"})
            allkers[(tip_out, vna_out)] = ker

        # --- pressure levels (once is enough) ---
        if vlevs is None and "lev" in kernels.coords:
            vlevs = kernels["lev"].rename({"lev": "player"})

    # --- save outputs ---
    with open(os.path.join(cart_out, "allkers_SPECTRAL.p"), "wb") as f:
        pickle.dump(allkers, f)
    with open(os.path.join(cart_out, "vlevs_SPECTRAL.p"), "wb") as f:
        pickle.dump(vlevs, f)
    with open(os.path.join(cart_out, "cose_SPECTRAL.p"), "wb") as f:
        pickle.dump(100 * vlevs.player, f)

    return allkers

def load_kernel_ERA5(cart_k, cart_out, finam):
    """
    Loads and preprocesses ERA5 kernels for further analysis.

    This function reads NetCDF files containing ERA5 kernels for various variables and conditions 
    (clear-sky and all-sky), renames the coordinates for compatibility with the Xarray data model, 
    and saves the preprocessed results as pickle files.

    Parameters:
    -----------
    cart_k : str
        Path to the directory containing the ERA5 kernel NetCDF files.
        Files should follow the naming format `ERA5_kernel_{variable}_TOA.nc`.

    cart_out : str
        Path to the directory where preprocessed files (pressure levels, kernels, and metadata) 
        will be saved as pickle files.

    Returns:
    --------
    allkers : dict
        A dictionary containing the preprocessed kernels. The dictionary keys are tuples of the form `(tip, variable)`, where:
        - `tip`: Atmospheric condition ('clr' for clear-sky, 'cld' for all-sky).
        - `variable`: Name of the variable (`'t'` for temperature, `'ts'` for surface temperature, `'wv_lw'`, `'wv_sw'`, `'alb'`).

    Saved Files:
    ------------
    - **`vlevs_ERA5.p`**: Pickle file containing the pressure levels (`player`).
    - **`k_ERA5.p`**: Pickle file containing the ERA5 kernel for the variable 't' under all-sky conditions.
    - **`cose_ERA5.p`**: Pickle file containing the pressure levels scaled to hPa.
    - **`allkers_ERA5.p`**: Pickle file containing all preprocessed kernels.

    Notes:
    ------
    - The NetCDF kernel files must be organized as `ERA5_kernel_{variable}_TOA.nc` and contain 
      the fields `TOA_clr` and `TOA_all` for clear-sky and all-sky conditions, respectively.
    - This function uses the Xarray library to handle datasets and Pickle to save processed data.

    """
    vnams = ['ta_dp', 'ts', 'wv_lw_dp', 'wv_sw_dp', 'alb']
    tips = ['clr', 'cld']
    allkers = dict()
    for tip in tips:
        for vna in vnams:
            ker = xr.load_dataset(cart_k+ finam.format(vna))
            ker=ker.rename({'latitude': 'lat', 'longitude': 'lon'})
                
            if vna=='ta_dp':
                ker=ker.rename({'level': 'player'})
                vna='t'
            if vna=='wv_lw_dp':
                ker=ker.rename({'level': 'player'})
                vna='wv_lw'
            if vna=='wv_sw_dp':
                ker=ker.rename({'level': 'player'})
                vna='wv_sw'
            if tip=='clr':
                stef=ker.TOA_clr
            else:
                stef=ker.TOA_all
            allkers[(tip, vna)] = stef.assign_coords(month = np.arange(1, 13))

    
    vlevs = xr.load_dataset( cart_k+'dp_era5.nc')
    vlevs=vlevs.rename({'level': 'player', 'latitude': 'lat', 'longitude': 'lon'})
    cose = 100*vlevs.player
    pickle.dump(vlevs, open(cart_out + 'vlevs_ERA5.p', 'wb'))
    pickle.dump(cose, open(cart_out + 'cose_ERA5.p', 'wb'))
    return allkers

def load_kernel_HUANG(cart_k, cart_out, finam):
    """
    Loads and processes climate kernel datasets (from HUANG 2017), and saves specific datasets to pickle files.

    Parameters:
    -----------
    cart_k : str
        Path template to the kernel dataset files. 
        Placeholders should be formatted as `{}` to allow string formatting.
        
    cart_out : str
        Path template to save the outputs. 

    Returns:
    --------
    allkers : dict
        A dictionary containing the loaded and processed kernels.
    
    Additional Outputs:
    -------------------
    The function also saves three objects as pickle files in a predefined output directory:
      - `vlevs.p`: The vertical levels data from the 'dp.nc' file.
      - `k.p`: The longwave kernel data corresponding to cloudy-sky temperature ('cld', 't').
      - `cose.p`: A scaled version (100x) of the 'player' variable from the vertical levels data.
    """
    vnams = ['t', 'ts', 'wv_lw', 'wv_sw', 'alb']
    tips = ['clr', 'cld']
    allkers = dict()

    for tip in tips:
        for vna in vnams:
            file_path = cart_k + finam.format(vna, tip)

            if not os.path.exists(file_path):
                print("ERRORE: Il file non esiste ->", file_path)
            else:
                ker = xr.load_dataset(file_path)

            allkers[(tip, vna)] = ker.assign_coords(month = np.arange(1, 13))
            if vna in ('ts', 't', 'wv_lw'):
                allkers[(tip, vna)]=allkers[(tip, vna)].lwkernel
            else:
                allkers[(tip, vna)]=allkers[(tip, vna)].swkernel

    vlevs = xr.load_dataset( cart_k + 'dp.nc')  
    pickle.dump(vlevs, open(cart_out + 'vlevs_HUANG.p', 'wb'))
    cose = 100*vlevs.player
    pickle.dump(cose, open(cart_out + 'cose_HUANG.p', 'wb'))

    return allkers

def load_kernel_wrapper(ker, config_file: str):
    """
    Loads and processes climate kernel datasets, and saves specific datasets to pickle files.

    Parameters:
    -----------

    ker (str): 
        The name of the kernel to load (e.g., 'ERA5' or 'HUANG').

    cart_k : str
        Path template to the kernel dataset files. 
        Placeholders should be formatted as `{}` to allow string formatting.
        
    cart_out : str
        Path template to save the outputs. 
    
    Returns:
    --------
    allkers : dict
        A dictionary containing the preprocessed kernels. The dictionary keys are tuples of the form `(tip, variable)`, where:
        - `tip`: Atmospheric condition ('clr' for clear-sky, 'cld' for all-sky).
        - `variable`: Name of the variable (`'t'` for temperature, `'ts'` for surface temperature, `'wv_lw'`, `'wv_sw'`, `'alb'`).

    Saved Files:
    ------------
    - **`vlevs_ker.p`**: Pickle file containing the pressure levels (`player`).
    - **`cose_ker.p`**: Pickle file containing the pressure levels scaled to hPa.
    - **`allkers_ker.p`**: Pickle file containing all preprocessed kernels.

    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 

    if ker == 'ERA5':
        cart_k  = config['kernels']['era5']['path_input']
        cart_out = config['kernels']['era5']['path_output']
        finam   = config['kernels']['era5']['filename_template']
        allkers = load_kernel(ker, cart_k, cart_out, finam)

    elif ker == 'HUANG':
        cart_k  = config['kernels']['huang']['path_input']
        cart_out = config['kernels']['huang']['path_output']
        finam   = config['kernels']['huang']['filename_template']
        allkers = load_kernel(ker, cart_k, cart_out, finam)

    elif ker == 'SPECTRAL':
        cart_k  = config['kernels']['spect']['path_input']
        cart_out = config['kernels']['spect']['path_output']
        # NOTA: niente finam
        allkers = load_kernel(ker, cart_k, cart_out)

    else:
        raise ValueError(f"Unknown kernel type: {ker}")

    return allkers

def load_kernel(ker, cart_k, cart_out, finam=None):
    """
    Selects and loads radiative kernels from different sources.

    This function acts as a unified interface for loading and preprocessing
    radiative kernels from multiple datasets: ERA5, HUANG, and SPECTRAL.
    It dispatches the loading task to the corresponding specialized routine
    based on the selected kernel type, and returns the processed kernel
    dictionary.

    Parameters:
    -----------
    ker : str
        Identifier of the kernel dataset to load.
        Supported values:
        - `'ERA5'`     : Loads kernels using `load_kernel_ERA5`.
        - `'HUANG'`    : Loads kernels using `load_kernel_HUANG`.
        - `'SPECTRAL'` : Loads spectrally resolved kernels using
                         `load_spectral_kernel`.

    cart_k : str
        Path to the directory containing the kernel NetCDF files for the
        selected dataset.

    cart_out : str
        Path to the directory where preprocessed kernels, metadata,
        and auxiliary files will be saved by the underlying loader.

    finam : str
        Filename pattern used to locate the kernel files. It must include
        a formatting placeholder (e.g., `'ERA5_kernel_{}_TOA.nc'`,
        `'spectral_kernel_ste_{}.nc'`) that will be filled with the
        variable name or kernel type.

    Returns:
    --------
    allkers : dict
        A dictionary containing the preprocessed kernels produced by the
        selected loader. The structure of the dictionary depends on the
        kernel dataset:
        - ERA5     → keys of the form `(tip, variable)`
        - HUANG    → structure defined in `load_kernel_HUANG`
        - SPECTRAL → keys of the form `(tip, variable)` with spectral
                      frequency selection applied

    Notes:
    ------
    - This function does not perform preprocessing directly; all variable
      renaming, frequency selection, and file saving are handled inside
      the specific loader.
    - If an unsupported kernel name is provided, the function will fail
      silently unless additional validation is added.
    """
    if ker == 'ERA5':
        return load_kernel_ERA5(cart_k, cart_out, finam)

    elif ker == 'HUANG':
        return load_kernel_HUANG(cart_k, cart_out, finam)

    elif ker == 'SPECTRAL':
        return load_spectral_kernel(cart_k, cart_out)

    else:
        raise ValueError(f"Unsupported kernel type: {ker}")

###### LOAD AND CHECK DATA
def read_data(config_file: str, variable_mapping_file: str = "configvariable.yml") -> xr.Dataset:
    """
    Reads the configuration from the YAML file, opens the NetCDF file specified in the config,
    and standardizes the variable names in the dataset.
    
    Parameters:
    -----------
    config_file : str
        The path to the YAML configuration file.
    
    variable_mapping_file : str
        Path to the YAML file that contains renaming and computation rules.
    
    Returns:
    --------
    ds : xarray.Dataset
        The dataset with standardized variable names.
    """

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 

    file_path1 = config['file_paths'].get('experiment_dataset', None)
    file_path2 = config['file_paths'].get('experiment_dataset2', None)
    file_pathpl = config['file_paths'].get('experiment_dataset_pl', None)
    time_chunk = config.get('time_chunk', None)
    dataset_type = config.get('dataset_type', None)


    if not file_path1:
        raise ValueError("Error: The 'experiment_dataset' path is not specified in the configuration file")

    
    ds_list = [xr.open_mfdataset(file_path1, combine='by_coords', use_cftime=True, chunks={'time': time_chunk})]
 
    if file_path2 and file_path2.strip():
        ds_list.append(xr.open_mfdataset(file_path2, combine='by_coords', use_cftime=True, chunks={'time': time_chunk}))

    if file_pathpl and file_pathpl.strip():
        ds_list.append(xr.open_mfdataset(file_pathpl, combine='by_coords', use_cftime=True,  chunks={'time': time_chunk}))

    # Merge dataset
    ds = xr.merge(ds_list, compat="override")

    ds = standardize_names(ds, dataset_type, variable_mapping_file)

    return ds

def load_variable_mapping(configvar_file, dataset_type):
    """Load variable mappings for the specified dataset type from YAML."""
    with open(configvar_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get(dataset_type, {})

def safe_eval(expr, ds):
    """Safely evaluate a string expression using variables from the xarray dataset."""
    local_dict = {var: ds[var] for var in ds.variables}
    try:
        return eval(expr, {"__builtins__": {}}, local_dict)
    except Exception as e:
        print(f"Failed to evaluate expression '{expr}': {e}")
        return None

def standardize_names(ds, dataset_type="ece3", configvar_file="configvariable.yml"):
    """
    Standardizes and computes variable names in the dataset using a config file.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be standardized.
    dataset_type : str
        Either 'ece3', 'ece4', etc. depending on the mapping config.
    config_file : str
        Path to the YAML file with variable mappings.

    Returns
    -------
    xarray.Dataset
        Dataset with renamed and computed variables.
    """
    mapping = load_variable_mapping(configvar_file, dataset_type)
    rename_map = mapping.get("rename_map", {}) or {}
    compute_map = mapping.get("compute_map", {}) or {}

    # Apply renaming
    existing_renames = {old: new for old, new in rename_map.items() if old in ds.variables}
    ds = ds.rename(existing_renames)
    if existing_renames:
        print(f"Renamed variables: {existing_renames}")
    else:
        print("No variables needed to be renamed.")

     # Apply computed variables
    for new_var, expr in compute_map.items():
        if new_var not in ds:
            result = safe_eval(expr, ds)
            if result is not None:
                ds[new_var] = result
                print(f"Computed variable '{new_var}' using expression: {expr}")
            else:
                print(f"Failed to compute variable '{new_var}'")

    return ds

def check_data(ds, piok):
    if len(ds["time"]) != len(piok["time"]):
        raise ValueError("Error: The 'time' columns in 'ds' and 'piok' must have the same length. To fix use variable 'time_range' of the function")
    return

def preproc(ds):
    ds = ds.assign_coords(lat = ds.lat.round(4))
    if 'lat_bnds' in ds:
        ds['lat_bnds'] = ds['lat_bnds'].round(4)
    
    return ds

######################################################################################
#### Aux functions
def ref_clim(config_file: str, allvars, ker, variable_mapping_file: str, allkers=None):
    """
    Computes the reference climatology using the provided configuration, variables, and kernel data.

    Parameters:
    -----------
    config_file : str
        The path to the YAML configuration file.
    allvars : str
        The variable to process (e.g., 'alb', 'rsus').
    ker : dict
        The preprocessed kernels.
    variable_mapping_file : str
        Path to the YAML file that contains renaming and computation rules.

    Returns:
    --------
    piok : xarray.DataArray
        The computed climatology (PI).
    """

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 

    dataset_type = config.get('dataset_type', None)
    filin_pi = config['file_paths'].get('reference_dataset', None)
    filin_pi_pl = config['file_paths'].get('reference_dataset_pl', None)
    time_chunk = config.get('time_chunk', None)
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    
    if not filin_pi:
        raise ValueError("Error: the 'reference_dataset' path is not specified in the configuration file.")

    ds_list = [xr.open_mfdataset(filin_pi, combine='by_coords', compat='no_conflicts', use_cftime=True, preprocess = preproc, chunks={'time': time_chunk})]
    
    if filin_pi_pl and filin_pi_pl.strip():
        ds_list.append(xr.open_mfdataset(filin_pi_pl, combine='by_coords', use_cftime=True, chunks={'time': time_chunk}))

    ds_ref = xr.merge(ds_list, compat="override")

    ds_ref = standardize_names(ds_ref, dataset_type, variable_mapping_file)

    time_range_clim = config.get("time_range", {})
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None

    if allkers is None:  
        allkers = load_kernel_wrapper(ker, config)
    else:
        print("Using pre-loaded kernels.")

    if isinstance(allvars, str):
        allvars = [allvars]

    piok = {} 
    for vnams in allvars:
        clim_method = "climatology" if vnams in ['rsut', 'rlut', 'rsutcs', 'rlutcs'] else method
        piok[vnams] = climatology(ds_ref, allkers, vnams, time_range_clim, clim_method, time_chunk)

    return piok

def climatology(filin_pi:str,  allkers, allvars:str, time_range=None, method=None, time_chunk=12):
    """
    Computes the preindustrial (PI) climatology or running mean for a given variable or set of variables.
    The function handles the loading and processing of kernels (either HUANG or ERA5) and calculates the PI climatology
    or running mean depending on the specified parameters. The output can be used for anomaly calculations
    or climate diagnostics.
    Parameters:
    -----------
    filin_pi : str
        Template path for the preindustrial data NetCDF files, with a placeholder for the variable name.
        Example: `'/path/to/files/{}_data.nc'`.
    cart_k : str
        Path to the directory containing kernel dataset files.
    allkers  : dict
        Dictionary containing radiative kernels for different conditions (e.g., 'clr', 'cld').
    allvars : str
        The variable name(s) to process. For example, `'alb'` for albedo or specific flux variables
        (e.g., `'rsus'`, `'rsds'`).
    method : {"climatology", "running_m", "climatology_mean", "running_m_mean"}, default "climatology"
        Method for anomaly computation:
        - "climatology"            : computes the mean climatology over the entire time period
        - "running_m"              : computes a running mean (e.g., 252-month moving average) over the selected time period.
        - "climatology_mean"       : computes the mean climatology over the entire time period
        - "running_m_mean"         : computes a running mean (e.g., 252-month moving average) over the selected time period.
    time_chunk : int, optional (default=12)
        Time chunk size for processing data with Xarray. Optimizes memory usage for large datasets.
    Returns:
    --------
    piok : xarray.DataArray
        The computed PI climatology or running mean of the specified variable(s), regridded to match the kernel's spatial grid.
    Notes:
    ------
    - For albedo ('alb'), the function computes it as `rsus / rsds` using the provided PI files for surface upward
      (`rsus`) and downward (`rsds`) shortwave radiation.
    - If `use_climatology` is False, the function computes a running mean for the selected time period (e.g., years 2540-2689).
    - Kernels are loaded or preprocessed from `cart_k` and stored in `cart_out`. Supported kernels are HUANG and ERA5.
    """
    if ('cld', 't') in allkers:
        k = allkers[('cld', 't')]
    else:
        print(f"Key ('cld', 't') not found in allkers")
        k = None  
    pimean = dict()

    if allvars == 'alb':
        allvar = ['rsus', 'rsds']
        
        for vnam in allvar:
            if isinstance(filin_pi, str):  # 1: path ai file
                filist = glob.glob(filin_pi.format(vnam))
                filist.sort()
                var = xr.open_mfdataset(filist, chunks={'time': time_chunk}, use_cftime=True)
            elif isinstance(filin_pi, xr.Dataset):  # 2: dataset già caricato
                var = filin_pi[vnam]
            else:
                raise ValueError("filin_pi must to be a string path or an xarray.Dataset")

            if time_range is not None:
                var = var.sel(time=slice(time_range['start'], time_range['end']))

            if method in ["climatology", "climatology_mean"]:
                var_mean = var.groupby('time.month').mean()
                var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)
                pimean[vnam] = var_mean
            else:
                pimean[vnam] = ctl.regrid_dataset(var, k.lat, k.lon)

        piok = pimean['rsus'] / pimean['rsds']
        if method in ["running_m", "running_m_mean"]:
            piok = ctl.running_mean(piok, 252)

    else:
        if isinstance(filin_pi, str):  # 1: path ai file
            filist = glob.glob(filin_pi.format(allvars))
            filist.sort()
            var = xr.open_mfdataset(filist, chunks={'time': time_chunk}, use_cftime=True)
        elif isinstance(filin_pi, xr.Dataset):  # 2: dataset già caricato
            var = filin_pi[allvars]
        else:
            raise ValueError("filin_pi must to be a string path or an xarray.Dataset")

        if time_range is not None:
            var = var.sel(time=slice(time_range['start'], time_range['end']))

        #if time_range is not None:
            #var = var.sel(time=slice(time_range[0], time_range[1]))

        if method in ["climatology", "climatology_mean"]:
            var_mean = var.groupby('time.month').mean()
            var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)
            piok = var_mean
        else:
            piok = ctl.regrid_dataset(var, k.lat, k.lon)
            piok = ctl.running_mean(piok, 252)

    return piok

##tropopause computation (Reichler 2003) 
def mask_atm(var):
    """
    Generates a mask for atmospheric temperature data based on the lapse rate threshold.
    as in (Reichler 2003) 

    Parameters:
    -----------
    var: xarray.DataArray
    Atmospheric temperature dataset with pressure levels ('plev') as a coordinate. 

    Returns:
    --------
    mask : xarray.DataArray
        A mask array where:
        - Values are 1 where the lapse rate (`laps1`) is less than or equal to -2 K/km.
        - Values are NaN elsewhere.
    """
    A=(var.plev/var)*(9.81/1005)
    laps1=(var.diff(dim='plev'))*A  #derivata sulla verticale = laspe-rate

    laps1=laps1.where(laps1<=-2)
    mask = laps1/laps1
    return mask

### Mask for surf pressure
def mask_pres(surf_pressure, cart_out:str, allkers, config_file=None):
    """
    Computes a "width mask" for atmospheric pressure levels based on surface pressure and kernel data.

    The function determines which pressure levels are above or below the surface pressure (`ps`) 
    and generates a mask that includes NaN values for levels below the surface pressure and 
    interpolated values near the surface. It supports kernels from HUANG and ERA5 datasets.

    Parameters:
    -----------
    surf_pressure : xr.Dataset
        - An xarray dataset containing surface pressure (`ps`) values.
          The function computes a climatology based on mean monthly values.
        - If a string (path) is provided, the corresponding NetCDF file(s) are loaded.

    cart_out : str
        Path to the directory where processed kernel files (e.g., 'kHUANG.p', 'kERA5.p', 'vlevsHUANG.p', 'vlevsERA5.p') 
        are stored or will be saved.

    allkers  : dict
        Dictionary containing radiative kernels for different conditions (e.g., 'clr', 'cld').    

    Returns:
    --------
    wid_mask : xarray.DataArray
        A mask indicating the vertical pressure distribution for each grid point. 
        Dimensions depend on the kernel data and regridded surface pressure:
        - For HUANG: [`player`, `lat`, `lon`]
        - For ERA5: [`player`, `month`, `lat`, `lon`]

    Notes:
    ------
    - Surface pressure (`ps`) climatology is computed as the mean monthly values over all available time steps.
    - `wid_mask` includes NaN values for pressure levels below the surface and interpolated values for the 
      level nearest the surface.
    - For HUANG kernels, the `dp` (pressure thickness) values are directly used. For ERA5, the monthly mean `dp` is used.

    Dependencies:
    -------------
    - Xarray for dataset handling and computations.
    - Numpy for array manipulations.
    - Custom library `ctl` for regridding datasets.
    """

    # MODIFIED TO WORK BOTH WITH ARRAY AND FILE
    k = allkers[('cld', 't')]
    vlevs = pickle.load(open(os.path.join(cart_out, 'vlevs_HUANG.p'), 'rb'))

    # If surf_pressure is an array:
    if isinstance(surf_pressure, xr.Dataset):
        psclim = surf_pressure.groupby('time.month').mean(dim='time')
        psye = psclim['ps'].mean('month')

    # If surf_pressure is a path, open config_file
    elif isinstance(surf_pressure, str):
        if pressure_path is None:
            if config_file is None:
                raise ValueError("config_file must be provided when surf_pressure is a directory.")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            pressure_path = config["file_paths"].get("pressure_data", None)
    
        if not pressure_path:
            raise ValueError("No pressure_data path specified in the configuration file, but surf_pressure was given as a path.")

        ps_files = sorted(glob.glob(pressure_path))  
        if not ps_files:
            raise FileNotFoundError(f"No matching files found for pattern: {pressure_path}")

        ps = xr.open_mfdataset(ps_files, combine='by_coords')

        # Check that 'ps' exists
        if 'ps' not in ps:
            raise KeyError("The dataset does not contain the expected 'ps' variable.")

        # Convert time variable to datetime if necessary
        if not np.issubdtype(ps['time'].dtype, np.datetime64):
            ps = ps.assign_coords(time=pd.to_datetime(ps['time'].values))
    
        # Resample to monthly and calculate climatology
        ps_monthly = ps.resample(time='M').mean()
        psclim = ps_monthly.groupby('time.month').mean(dim='time')
        psye = psclim['ps'].mean('month')
   
    else:
        raise TypeError("surf_pressure must be an xarray.Dataset or a path to NetCDF files.")

    psye_rg = ctl.regrid_dataset(psye, k.lat, k.lon).compute()
    wid_mask = np.empty([len(vlevs.player)] + list(psye_rg.shape))

    for ila in range(len(psye_rg.lat)):
        for ilo in range(len(psye_rg.lon)):
            ind = np.where((psye_rg[ila, ilo].values/100. - vlevs.player.values) > 0)[0][0]
            wid_mask[:ind, ila, ilo] = np.nan
            wid_mask[ind, ila, ilo] = psye_rg[ila, ilo].values/100. - vlevs.player.values[ind]
            wid_mask[ind+1:, ila, ilo] = vlevs.dp.values[ind+1:]
        

    wid_mask = xr.DataArray(wid_mask, dims = k.dims[1:], coords = k.drop('month').coords)
    return wid_mask

def pliq(T):
    pliq = 0.01 * np.exp(54.842763 - 6763.22 / T - 4.21 * np.log(T) + 0.000367 * T + np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))
    return pliq

def pice(T):
    pice = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T) / 100.0
    return pice

def dlnws(T):
    """
    Calculates 1/(dlnq/dT_1K).
    """
    pliq0 = pliq(T)
    pice0 = pice(T)

    T1 = T + 1.0
    pliq1 = pliq(T1)
    pice1 = pice(T1)
    
    # Use np.where to choose between pliq and pice based on the condition T >= 273
    if isinstance(T, xr.DataArray):# and isinstance(T.data, da.core.Array):
        ws = xr.where(T >= 273, pliq0, pice0)    # Dask equivalent of np.where is da.where
        ws1 = xr.where(T1 >= 273, pliq1, pice1)
    else:
        ws = np.where(T >= 273, pliq0, pice0)
        ws1 = np.where(T1 >= 273, pliq1, pice1)
    
    # Calculate the inverse of the derivative dws
    dws = ws / (ws1 - ws)

    if isinstance(dws, np.ndarray):
        dws = ctl.transform_to_dataarray(T, dws, 'dlnws')
   
    return dws

############# SPATIAL PATTERN FUNCTION #############
def regress_pattern_vectorized(feedback_data, gtas):
    """
    Perform a linear regression between feedback_data (lat, lon, year) and gtas (year)
    using xarray.apply_ufunc for efficient, vectorized computation.

    Parameters:
    - feedback_data (xr.DataArray): feedback values (time, lat, lon)
    - gtas (xr.DataArray): global temperature anomaly over time (time,)

    Returns:
    - slope_map (xr.DataArray): slope (feedback pattern) for each lat/lon
    - stderr_map (xr.DataArray): standard error of the regression slope for each lat/lon
    """
    def linregress_1d(y, x):
        # Remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask] #gtas data
        y = y[mask] #fb data

        #skip if not enough data
        if len(x) < 2:
            return np.nan, np.nan
        
        #normalization 
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std == 0:
            return np.nan, np.nan
        x_norm = (x - x_mean) / x_std

        # Perform linear regression with LS
        A = np.vstack([x_norm, np.ones_like(x_norm)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0] #slope m and intercept c
        # Calculate residuals and standard error
        y_pred = m * x_norm + c
        residuals = y - y_pred
        dof = len(x) - 2
        if dof <= 0:
            return m / x_std, np.nan
        
        # Calculate standard error of the slope
        stderr = (np.sqrt(np.sum(residuals**2) / dof)/(np.sqrt(np.sum((x_norm)**2)) * x_std))
        
        #m/x_std to match ther original scale of x (normalized before)
        return m / x_std, stderr

    # Use apply_ufunc for broadcasting regression across all lat/lon points
    slope, stderr = xr.apply_ufunc(
        linregress_1d,
        feedback_data,
        gtas,
        input_core_dims=[['year'], ['year']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float]
    )

    return slope, stderr

############ ANOMALY PREPROCESSING FUNCTION #############
def compute_anomalies(var, clim, method="climatology", nonlinear=False, check=False):
    """
    Compute anomalies between a variable and its climatology/reference.

    Parameters
    ----------
    var : xr.DataArray
        Variable to anomaly-ize (e.g., ts, hus, etc.)
    clim : xr.DataArray
        Reference climatology or mean (monthly or time mean).
    method : str, default "monthly"
        Method for anomaly computation:
        - "climatology"            : monthly averaged climatology to calculate the anomaly
        - "running_m"              : 21-years running mean climatology to calculate the anomaly 
        - "climatology_mean"       : anomaly is computed as a single averaged value over the time dimension
        - "running_m_mean"         : anomaly is computed as a single averaged value over the time dimension
    nonlinear : bool, default False
        If True, apply log-difference instead of linear subtraction.
    check : bool, default False
        If True, perform check_data(var, clim) before computing anomalies.

    Returns
    -------
    xr.DataArray
        Anomalies
    """
    # choose linear or nonlinear subtraction
    if nonlinear:
        func = lambda x, c: np.log(x) - np.log(c)
    else:
        func = lambda x, c: x - c

    if method == "climatology":
        # group anomalies by month (climatology already grouped by month)
        return xr.apply_ufunc(func, var.groupby("time.month"), clim, dask="allowed")

    elif method == "running_m":
        if check:
            check_data(var, clim)
        # broadcast climatology to full time axis
        clim_aligned = clim.drop("time")
        clim_aligned["time"] = var["time"]
        clim_aligned = clim_aligned.chunk(var.chunks)
        return xr.apply_ufunc(func, var, clim_aligned, dask="allowed")

    elif method == "climatology_mean":
        # anomalies of mean seasonal cycle vs climatology
        return xr.apply_ufunc(func, var.groupby("time.month").mean(), clim, dask="allowed")

    elif method == "running_m_mean":
        if check:
            check_data(var, clim)
        # broadcast climatology to full time axis
        clim_aligned = clim.drop("time")
        clim_aligned["time"] = var["time"]
        clim_aligned = clim_aligned.chunk(var.chunks)
        # anomalies of mean seasonal cycle, both monthly-averaged
        return xr.apply_ufunc(
            func,
            var.groupby("time.month").mean(),
            clim_aligned.groupby("time.month").mean(),
            dask="allowed"
        )

    else:
        raise ValueError(f"Unknown anomaly method {method}")
       
# FUNCTION FOR WV ANOMALIES
# From Mass mixing Ratio (kg/kg to ppmv)
def q_to_ppmv(q_inp):
    Ma = 28.97  # Molecular weight of dry air
    Mw = 18.02  # Molecular weight of water vapor
    vw_ppmv = q_inp / (1 - q_inp) * (Ma / Mw) * 10**6
    return vw_ppmv

############ RADIATIVE ANOMALY FUNCTIONS #############
#PLANCK SURFACE
def Rad_anomaly_planck_surf_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    Wrapper for Rad_anomaly_planck_surf function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Planck-Surface anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    variable_mapping_file : str
        Path to the YAML file with variable standardization rules.

    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, variable_mapping_file)
    print("Variables to consider upload...")
    allvars = 'ts'
    print("Read parameters from configuration file...")
    if allvars not in ds.variables:
        raise ValueError("The ts variable is not in the dataset")
    
    cart_out = config['file_paths'].get("output")
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    save_pattern = config.get("save_pattern", False)
    save_pattern = bool(save_pattern)

    # Read time ranges from config
    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # Determine usage scenario
    if time_range_exp and not time_range_clim:
        print("Only experiment time range is provided. Using it for analysis.")
    elif time_range_exp and time_range_clim:
        print(f"Using separate time ranges for climatology: {time_range_clim} and experiment: {time_range_exp}")
    elif time_range_clim and not time_range_exp:
        print("Only climatology time range is provided. Using it for both climatology and experiment.")
        time_range_exp = time_range_clim  # fallback
    else:
        print("No valid time ranges provided. Proceeding with full time range in the data.")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config_file, allvars, ker, variable_mapping_file, allkers=allkers) 

    print("Planck-Surface radiative anomaly computing...")
    radiation = Rad_anomaly_planck_surf(ds, ref_clim_data, ker, allkers, cart_out, time_range_exp, method, save_pattern)

    return (radiation)

def Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, time_range=None, method=None, save_pattern=False):
    """
    Compute the Planck surface radiation anomaly using climate model data and radiative kernels.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing surface temperature (`ts`) and near-surface air temperature (`tas`).
    piok : xarray.Dataset
        Reference dataset containing climatological or multi-year mean surface temperature.
    ker : str
        Name of the kernel set used for radiative calculations (e.g., 'ERA5', 'HUANG').
    allkers : dict
        Dictionary of radiative kernels for different sky conditions and components,
        typically containing ('clr', 'planck_surf') and ('cld', 'planck_surf').
    cart_out : str
        Output directory where results will be saved.
    time_range : tuple of str, optional
        Time range for selecting data, e.g. ('2000-01-01', '2010-12-31').
    method : {"climatology", "running_m", "climatology_mean", "running_m_mean"}, default "climatology"
        Method for anomaly computation:
        - "climatology"            : monthly averaged climatology to calculate the anomaly
        - "running_m"              : 21-years running mean climatology to calculate the anomaly 
        - "climatology_mean"       : anomaly is computed as a single averaged value over the time dimension
        - "running_m_mean"         : anomaly is computed as a single averaged value over the time dimension
    save_pattern : bool, default False
        If True, save the full spatial anomaly patterns in addition to global means.

    Returns
    -------
    dict
        Dictionary containing computed Planck surface radiation anomalies:
        - ('clr', 'planck_surf') : clear-sky Planck surface anomaly
        - ('cld', 'planck_surf') : all-sky Planck surface anomaly

    Saved Outputs
    -------------
    - dRt_planck-surf_global_{tip}_{method}-{ker}kernels.nc
    (global mean anomaly for each sky condition, `tip` = clr or cld)

    If `save_pattern=True`, also saves:
    - dRt_planck-surf_pattern_{tip}_{method}-{ker}kernels.nc
    (full spatial anomaly field for each sky condition)
    """
    radiation = dict()
    k = allkers[('cld', 't')]

    # define suffix for saved files based on method only
    suffix = f"_{method}"

    # select and regrid variable
    if time_range is not None:
        var = ds['ts'].sel(time=slice(time_range['start'], time_range['end']))
    else:
        var = ds['ts']
    var = ctl.regrid_dataset(var, k.lat, k.lon)
  
    # Anomaly computation (always linear here)
    anoms = compute_anomalies(var, piok['ts'], method=method, nonlinear=False, check=True)
 
    for tip in ['clr', 'cld']:
        print(f"Processing {tip}")  
        try:
            kernel = allkers[(tip, 'ts')]
            print("Kernel loaded successfully")  
        except Exception as e:
            print(f"Error loading kernel for {tip}: {e}")  
            continue  

        # Apply kernel first
        if method in ["climatology_mean", "running_m_mean"]:
            #anomalies already averaged over months → just mean over month dimension
            dRt = (anoms * kernel).mean("month")
        elif method in ["climatology", "running_m"]:
            # monthly or direct → compute yearly mean after grouping anomalies by month
            dRt = (anoms.groupby("time.month") * kernel).groupby("time.year").mean("time")

        #Save full dRt pattern before global averaging
        if save_pattern: 
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} surface Planck dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_planck-surf_pattern_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt)
        planck = dRt_glob.compute()
        radiation[(tip, 'planck-surf')] = planck
        planck.to_netcdf(cart_out + "dRt_planck-surf_global_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")
        planck.close()
        
    return(radiation)

#PLANK-ATMO AND LAPSE RATE WITH VARYING TROPOPAUSE
def Rad_anomaly_planck_atm_lr_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    Wrapper for Rad_anomaly_planck_atm_lr function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Planck-Atmosphere-LpseRate anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    variable_mapping_file : str
        Path to the YAML file with variable standardization rules.


    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, variable_mapping_file)
    print("Variables to consider upload...")
    allvars = 'ts ta'.split()
    print("Read parameters from configuration file...")

    for var in allvars:
        if var not in ds.variables:
            raise ValueError(f"The variable '{var}' is not in the dataset")
    
    dataset_type = config.get('dataset_type', None)
    cart_out = config['file_paths'].get("output")
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    use_atm_mask = config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_atm_mask = bool(use_atm_mask)
    save_pattern = bool(save_pattern)

    # Read time ranges from config
    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # Determine usage scenario
    if time_range_exp and not time_range_clim:
        print("Only experiment time range is provided. Using it for analysis.")
    elif time_range_exp and time_range_clim:
        print(f"Using separate time ranges for climatology: {time_range_clim} and experiment: {time_range_exp}")
    elif time_range_clim and not time_range_exp:
        print("Only climatology time range is provided. Using it for both climatology and experiment.")
        time_range_exp = time_range_clim  # fallback
    else:
        print("No valid time ranges provided. Proceeding with full time range in the data.")

    # Surface pressure management
    surf_pressure = None
    if ker == 'HUANG':  # HUANG requires surface pressure
        pressure_path = config['file_paths'].get('pressure_data', None)
        
        if pressure_path:  # If pressure data is specified, load it
            print("Loading surface pressure data...")
            ps_files = sorted(glob.glob(pressure_path))  # Support for patterns like "*.nc"
            if not ps_files:
                raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
            
            surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')
            surf_pressure = standardize_names(surf_pressure, dataset_type, variable_mapping_file)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config, allvars, ker, variable_mapping_file, allkers=allkers) 

    print("Planck-Atmosphere-LapseRate radiative anomaly computing...")
    radiation = Rad_anomaly_planck_atm_lr(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, time_range_exp, config, method, use_atm_mask, save_pattern)
    
    return (radiation)

def Rad_anomaly_planck_atm_lr(ds, piok, ker, allkers, cart_out, surf_pressure=None, time_range=None, config_file=None, method=None, use_atm_mask=True, save_pattern=False):

    """
    Computes atmospheric Planck and lapse-rate radiation anomalies using climate model data and radiative kernels.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing atmospheric temperature (`ta`) and surface temperature (`ts`).
    piok : xr.Dataset
        Reference dataset containing atmospheric (`ta`) and surface (`ts`) temperatures
        for computing anomalies (climatology or multi-year mean).
    ker : str
        Name of the kernel set used for radiative calculations (e.g., 'ERA5', 'HUANG').
    allkers : dict
        Dictionary containing radiative kernels for clear-sky ('clr') and all-sky ('cld') conditions.
    cart_out : str
        Output directory where results will be saved.
    surf_pressure : xr.Dataset, optional
        Surface pressure dataset (`ps`), required for HUANG kernels.
    time_range : tuple of str, optional
        Time range for selecting data (format: ('YYYY-MM-DD', 'YYYY-MM-DD')).
    config_file : str, optional
        Path to configuration file, used when generating masks for kernels.
    method : {"climatology", "running_m", "climatology_mean", "running_m_mean"}, default "climatology"
        Method for anomaly computation:
        - "climatology"            : monthly averaged climatology to calculate the anomaly
        - "running_m"              : 21-years running mean climatology to calculate the anomaly 
        - "climatology_mean"       : anomaly is computed as a single averaged value over the time dimension
        - "running_m_mean"         : anomaly is computed as a single averaged value over the time dimension
    save_pattern : bool, optional
    use_atm_mask : bool, default True
        If True, apply an atmospheric mask to the anomalies before kernel multiplication.
    save_pattern : bool, default False
        If True, save full spatial anomaly patterns (not just global means).

    Returns
    -------
    dict
        Dictionary containing computed anomalies:
        - ('clr', 'planck-atmo')
        - ('clr', 'lapse-rate')
        - ('cld', 'planck-atmo')
        - ('cld', 'lapse-rate')

    Saved Outputs
    -------------
    - dRt_planck-atmo_global_{tip}_{method}-{ker}kernels.nc
    - dRt_lapse-rate_global_{tip}_{method}-{ker}kernels.nc
    If `save_pattern=True`, also saves:
    - dRt_planck-atmo_pattern_{tip}_{method}-{ker}kernels.nc
    - dRt_lapse-rate_pattern_{tip}_{method}-{ker}kernels.nc
    """
    if ker == 'HUANG' and surf_pressure is None:
        raise ValueError("Error: The 'surf_pressure' parameter cannot be None when 'ker' is 'HUANG'.")

    radiation=dict()
    k= allkers[('cld', 't')]

    cose=pickle.load(open(cart_out + 'cose_'+ker+'.p', 'rb'))
    if ker=='HUANG':
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
        wid_mask=mask_pres(surf_pressure, cart_out, allkers, config_file) 
    if ker=='ERA5':
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
   
    # define suffix for saved files based on method only
    suffix = f"_{method}"

    if time_range is not None:
        var = ds['ta'].sel(time=slice(time_range['start'], time_range['end'])) 
        var_ts = ds['ts'].sel(time=slice(time_range['start'], time_range['end'])) 
        var=ctl.regrid_dataset(var, k.lat, k.lon)
        var_ts=ctl.regrid_dataset(var_ts, k.lat, k.lon)
    else:
        var=ds['ta']
        var_ts=ds['ts']
        var = ctl.regrid_dataset(var, k.lat, k.lon)
        var_ts = ctl.regrid_dataset(var_ts, k.lat, k.lon)

    # Anomaly computation (always linear here)
    ta_anom = compute_anomalies(var, piok["ta"], method=method, nonlinear=False, check=True)
    ts_anom = compute_anomalies(var_ts, piok["ts"], method=method, nonlinear=False, check=True)

    if use_atm_mask==True:
        mask = mask_atm(var)
        if method not in ["climatology", "running_m"]:
            mask = mask.groupby("time.month").mean()
        ta_anom = (ta_anom * mask).interp(plev=cose)
    else:
        ta_anom = ta_anom.interp(plev=cose)
    
    if method in ["climatology", "running_m"]:
        anoms_lr = ta_anom - ts_anom
    else:
        anoms_lr = ta_anom - ts_anom.mean("month")
    anoms_unif = ta_anom - anoms_lr

    for tip in ['clr', 'cld']:
        print(f"Processing {tip}")  
        try:
            kernel = allkers[(tip, 't')]
            print("Kernel loaded successfully")  
        except Exception as e:
            print(f"Error loading kernel for {tip}: {e}")  
            continue  

        # Apply kernel first
        if method in ["climatology_mean", "running_m_mean"]:
            #anomalies already averaged over months → just mean over month dimension
            if ker == "HUANG":
                dRt_unif = (anoms_unif * kernel * wid_mask / 100).sum("player")
                dRt_lr = (anoms_lr * kernel * wid_mask / 100).sum("player")
            if ker == "ERA5":
                dRt_unif = (anoms_unif * (kernel * vlevs.dp / 100)).sum("player")
                dRt_lr = (anoms_lr * (kernel * vlevs.dp / 100)).sum("player")
            if ker == 'SPECTRAL':
                dRt_unif = (anoms_unif*kernel).sum(dim="player")
                dRt_lr = (anoms_lr*kernel).sum(dim="player")
        elif method in ["climatology", "running_m"]:
            if ker == "HUANG":
                dRt_unif = (anoms_unif.groupby('time.month') * kernel * wid_mask / 100).sum("player")
                dRt_lr = (anoms_lr.groupby('time.month') * kernel * wid_mask / 100).sum("player")
            if ker == "ERA5":
                dRt_unif = (anoms_unif.groupby('time.month') * (kernel * vlevs.dp / 100)).sum("player")
                dRt_lr = (anoms_lr.groupby('time.month') * (kernel * vlevs.dp / 100)).sum("player")
            if ker=='SPECTRAL':
                dRt_unif = (anoms_unif.groupby('time.month')*kernel).sum(dim="player")
                dRt_lr = (anoms_lr.groupby('time.month')*kernel).sum(dim="player")

        # Average according to method
        if method in ["climatology", "running_m"]:
            dRt_unif = dRt_unif.groupby("time.year").mean("time")
            dRt_lr = dRt_lr.groupby("time.year").mean("time")
        else:
            dRt_unif = dRt_unif.mean("month")
            dRt_lr = dRt_lr.mean("month")
        
        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt_unif.name = "dRt_atmo"
            dRt_unif.attrs["description"] = f"{tip} atmosperic Planck dRt pattern"
            dRt_unif.to_netcdf(cart_out + "dRt_planck-atmo_pattern_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")

            dRt_lr.name = "dRt_lr"
            dRt_lr.attrs["description"] = f"{tip} lapse-rate dRt pattern"
            dRt_lr.to_netcdf(cart_out + "dRt_lapse-rate_pattern_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        radiation[(tip,'planck-atmo')]=feedbacks_atmo
        radiation[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip + suffix +"-"+ker+"kernels.nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  + suffix +"-"+ker+"kernels.nc", format="NETCDF4")
        feedbacks_atmo.close()
        feedbacks_lr.close()

    return(radiation)

#ALBEDO
def Rad_anomaly_albedo_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    Wrapper for Rad_anomaly_albedo function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Albedo anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    variable_mapping_file : str
        Path to the YAML file with variable standardization rules.

    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, variable_mapping_file)
    print("Variables to consider upload...")
    allvars = 'alb'
    print("Read parameters from configuration file...")
    
    cart_out = config['file_paths'].get("output")
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    save_pattern = config.get("save_pattern", False)
    save_pattern = bool(save_pattern)

    # Read time ranges from config
    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # Determine usage scenario
    if time_range_exp and not time_range_clim:
        print("Only experiment time range is provided. Using it for analysis.")
    elif time_range_exp and time_range_clim:
        print(f"Using separate time ranges for climatology: {time_range_clim} and experiment: {time_range_exp}")
    elif time_range_clim and not time_range_exp:
        print("Only climatology time range is provided. Using it for both climatology and experiment.")
        time_range_exp = time_range_clim  # fallback
    else:
        print("No valid time ranges provided. Proceeding with full time range in the data.")

    print(f"Time range used for the simulation analysis: {time_range_exp}")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config, allvars, ker, variable_mapping_file, allkers=allkers) 

    print("Albedo radiative anomaly computing...")
    radiation = Rad_anomaly_albedo(ds, ref_clim_data, ker, allkers, cart_out, time_range_exp, method, save_pattern)

    return (radiation)

def Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, time_range=None, method=None, save_pattern=False):
    """
    Compute the albedo radiation anomaly using climate model output and radiative kernels.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing surface upward (`rsus`) and downward (`rsds`) shortwave radiation.
    piok : xarray.Dataset
        Reference dataset containing climatological or multi-year mean albedo values.
    ker : str
        Name of the kernel set used for radiative calculations (e.g., 'ERA5', 'HUANG').
    allkers : dict
        Dictionary of radiative kernels for different sky conditions and components,
        typically containing ('clr', 'alb') and ('cld', 'alb').
    cart_out : str
        Output directory where results will be saved.
    method : {"climatology", "running_m", "climatology_mean", "running_m_mean"}, default "climatology"
        Method for anomaly computation:
        - "climatology"            : monthly averaged climatology to calculate the anomaly
        - "running_m"              : 21-years running mean climatology to calculate the anomaly 
        - "climatology_mean"       : anomaly is computed as a single averaged value over the time dimension
        - "running_m_mean"         : anomaly is computed as a single averaged value over the time dimension
    save_pattern : bool, default False
        If True, save the full spatial anomaly patterns in addition to global means.
    time_range : tuple of str, optional
        Time range for selecting data, e.g. ('2000-01-01', '2010-12-31').

    Returns
    -------
    dict
        Dictionary containing computed albedo radiation anomalies:
        - ('clr', 'albedo') : clear-sky albedo anomaly
        - ('cld', 'albedo') : all-sky albedo anomaly

    Saved Outputs
    -------------
    - dRt_albedo_global_{tip}_{method}-{ker}kernels.nc
    (global mean anomaly for each sky condition, `tip` = clr or cld)

    If `save_pattern=True`, also saves:
    - dRt_albedo_pattern_{tip}_{method}-{ker}kernels.nc
    (full spatial anomaly field for each sky condition)
"""
    
    radiation=dict()

    if ker == "SPECTRAL":
        print("Skipping albedo feedback for SPECTRAL kernels (not defined).")
        return radiation

    k=allkers[('cld', 't')]

    # define suffix for saved files based on method only
    suffix = f"_{method}"

    var_rsus= ds['rsus']
    var_rsds=ds['rsds'] 

    # n_zeros = (var_rsds == 0).sum().compute()
    # print(f"Warning: {n_zeros} zeros rds values!")
    # var_rsds_safe = xr.where(var_rsds == 0, np.nan, var_rsds)

    var = var_rsus/var_rsds #or var_rsds_safe 
    var = var.fillna(0)

    if time_range is not None:
        var = var.sel(time=slice(time_range['start'], time_range['end']))
    var = ctl.regrid_dataset(var, k.lat, k.lon)

    # Removing inf and nan from alb
    piok = piok['alb'].where(piok['alb'] > 0., 0.)
    var = var.where(var > 0., 0.)

    # Anomaly computation (always linear here)
    anoms = compute_anomalies(var, piok, method=method, nonlinear=False, check=True)

    for tip in [ 'clr','cld']:
        kernel = allkers[(tip, 'alb')]
        # Apply kernel first
        if method in ["climatology_mean", "running_m_mean"]:
            #anomalies already averaged over months → just mean over month dimension
            dRt = (anoms * kernel).mean("month")
        elif method in ["climatology", "running_m"]:
            # monthly or direct → compute yearly mean after grouping anomalies by month
            dRt = (anoms.groupby("time.month") * kernel).groupby("time.year").mean("time")
            
        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} albedo dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_albedo_pattern_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        radiation[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip + suffix +"-"+ker+"kernels.nc", format="NETCDF4")
        alb.close()

    return(radiation)

#W-V COMPUTATION
def Rad_anomaly_wv_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    Wrapper for Rad_anomaly_wv function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Water-Vapour anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    variable_mapping_file : str
        Path to the YAML file with standardization rules for variables.

    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, variable_mapping_file)
    print("Variables to consider upload...")
    allvars = 'hus ta'.split()
    print("Read parameters from configuration file...")

    for var in allvars:
        if var not in ds.variables:
            raise ValueError(f"The variable '{var}' is not in the dataset")

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    dataset_type = config.get('dataset_type', None)
    cart_out = config['file_paths'].get("output")
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    use_atm_mask = config.get("use_atm_mask",True)
    save_pattern = config.get("save_pattern", False)    
    use_atm_mask = bool(use_atm_mask)
    save_pattern = bool(save_pattern)

    # Read time ranges from config
    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # Determine usage scenario
    if time_range_exp and not time_range_clim:
        print("Only experiment time range is provided. Using it for analysis.")
    elif time_range_exp and time_range_clim:
        print(f"Using separate time ranges for climatology: {time_range_clim} and experiment: {time_range_exp}")
    elif time_range_clim and not time_range_exp:
        print("Only climatology time range is provided. Using it for both climatology and experiment.")
        time_range_exp = time_range_clim  # fallback
    else:
        print("No valid time ranges provided. Proceeding with full time range in the data.")

    # Surface pressure management
    surf_pressure = None
    if ker == 'HUANG':  # HUANG requires surface pressure
        pressure_path = config['file_paths'].get('pressure_data', None)
        
        if pressure_path:  # If pressure data is specified, load it
            print("Loading surface pressure data...")
            ps_files = sorted(glob.glob(pressure_path))  # Support for patterns like "*.nc"
            if not ps_files:
                raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
            
            surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')
            surf_pressure = standardize_names(surf_pressure, dataset_type, variable_mapping_file)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config, allvars, ker, variable_mapping_file, allkers=allkers) 

    print("Water-Vapour radiative anomaly computing...")
    radiation = Rad_anomaly_wv(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, time_range_exp, config, method, use_atm_mask, save_pattern)

    return (radiation)

def Rad_anomaly_wv(ds, piok, ker, allkers, cart_out, surf_pressure, time_range=None, config_file=None, method=None, use_atm_mask=True, save_pattern=False):
    
    """
    Compute water vapor radiation anomalies using climate model output and radiative kernels.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing specific humidity (`hus`) and atmospheric temperature (`ta`).
    piok : xr.Dataset
        Reference dataset containing climatological or multi-year mean values of
        specific humidity (`hus`) and atmospheric temperature (`ta`).
    ker : str
        Name of the kernel set used for radiative calculations (e.g., 'ERA5', 'HUANG').
    allkers : dict
        Dictionary of radiative kernels for different sky conditions and components,
        e.g. ('clr', 'wv_lw'), ('clr', 'wv_sw'), ('cld', 'wv_lw'), ('cld', 'wv_sw').
    cart_out : str
        Output directory where results will be saved.
    surf_pressure : xr.Dataset, optional
        Surface pressure dataset (`ps`), required for HUANG kernels.
    time_range : tuple of str, optional
        Time range for selecting data, e.g. ('2000-01-01', '2010-12-31').
    config_file : str, optional
        Path to configuration file, used when generating pressure masks for HUANG kernels.
    method : {"climatology", "running_m", "climatology_mean", "running_m_mean"}, default "climatology"
        Method for anomaly computation:
        - "climatology"            : monthly averaged climatology to calculate the anomaly
        - "running_m"              : 21-years running mean climatology to calculate the anomaly 
        - "climatology_mean"       : anomaly is computed as a single averaged value over the time dimension
        - "running_m_mean"         : anomaly is computed as a single averaged value over the time dimension
    use_atm_mask : bool, default True
        If True, apply an atmospheric mask before kernel multiplication.
    save_pattern : bool, default False
        If True, save the full spatial anomaly patterns in addition to global means.

    Returns
    -------
    dict
        Dictionary containing computed water vapor radiation anomalies:
        - ('clr', 'water-vapor') : clear-sky water vapor anomaly
        - ('cld', 'water-vapor') : all-sky water vapor anomaly

    Saved Outputs
    -------------
    - dRt_water-vapor_global_{tip}_{method}-{ker}kernels.nc
      (global mean anomaly for each sky condition, `tip` = clr or cld)

    If `save_pattern=True`, also saves:
    - dRt_water-vapor_pattern_{tip}_{method}-{ker}kernels.nc
      (full spatial anomaly field for each sky condition)
    """
    if ker == 'HUANG' and surf_pressure is None:
        raise ValueError("Error: The 'surf_pressure' parameter cannot be None when 'ker' is 'HUANG'.")

    k=allkers[('cld', 't')]
    cose=pickle.load(open(cart_out + 'cose_'+ker+'.p', 'rb'))
    radiation=dict()
    if ker=='ERA5':
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
    
    # define suffix for saved files based on method only
    suffix = f"_{method}"

    var=ds['hus']
    var_ta=ds['ta']
    if time_range is not None:
        var = ds['hus'].sel(time=slice(time_range['start'], time_range['end'])) 
        var_ta = ds['ta'].sel(time=slice(time_range['start'], time_range['end']))
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    var_ta = ctl.regrid_dataset(var_ta, k.lat, k.lon)
    if use_atm_mask==True:
        mask=mask_atm(var_ta)

    Rv = 461.5 # gas constant of water vapor
    Lv = 2.25e+06 # latent heat of water vapor

    if method in ["running_m", "running_m_mean"]:
        # Time alignment needed
        check_data(var_ta, piok["ta"])
        piok_ta = piok["ta"].drop("time")
        piok_ta["time"] = var["time"]

        piok_hus = piok["hus"].drop("time")
        piok_hus["time"] = var["time"]
    else:
        piok_ta = piok["ta"]
        piok_hus = piok["hus"]

    #ta_abs_pi = piok_ta.interp(plev = cose)
    if use_atm_mask==True:
        ta_abs_pi = (piok_ta*mask.groupby("time.month")).interp(plev = cose)
        piok_int = (piok_hus*mask.groupby("time.month")).interp(plev = cose)
    else:
        ta_abs_pi = piok_ta.interp(plev = cose)
        piok_int = piok_hus.interp(plev = cose)
    
    var_int = var.interp(plev = cose)
    
    if ker != 'SPECTRAL':
        vlevs = pickle.load(open(cart_out + f"vlevs_{ker}.p", "rb"))

    if ker == "HUANG":
        wid_mask = mask_pres(surf_pressure, cart_out, allkers, config_file)

        # nonlinear anomalies (log)
        anoms_ok3 = compute_anomalies( var, piok_hus, method=method, nonlinear=True, check=True)
        anoms_ok3 = (anoms_ok3*mask).interp(plev = cose)            
        coso3 = anoms_ok3 * dlnws(ta_abs_pi)

    if ker == "ERA5":
        # linear anomalies (x - clim)
        anoms = compute_anomalies(var, piok_hus, method=method, nonlinear=False, check=True)
        anoms = (anoms*mask).interp(plev = cose)  
        coso = (anoms / piok_int) * (ta_abs_pi**2) * Rv / Lv

    if ker == "SPECTRAL":
        var_wv = q_to_ppmv(var_int)
        piok_wv =q_to_ppmv(piok_int)
        anoms = compute_anomalies(var_wv, piok_wv, method=method, nonlinear=False, check=True)


    for tip in ['clr','cld']: 
        if ker != 'SPECTRAL':
            kernel_lw = allkers[(tip, 'wv_lw')]
            kernel_sw = allkers[(tip, 'wv_sw')]
            kernel = kernel_lw + kernel_sw
        else:
            kernel_lw = allkers[(tip, 'wv_lw')]
            kernel= kernel_lw
        
        if ker=='HUANG':
            if method in ["climatology", "running_m"]:
                dRt = (coso3.groupby('time.month')* kernel* wid_mask/100).sum('player').groupby('time.year').mean('time')
                dRt_lw = (coso3.groupby('time.month')* kernel_lw* wid_mask/100).sum('player').groupby('time.year').mean('time')
                dRt_sw = (coso3.groupby('time.month')* kernel_sw* wid_mask/100).sum('player').groupby('time.year').mean('time')
            else:
                dRt = (coso3* kernel* wid_mask/100).sum('player').mean('month')
                dRt_lw = (coso3* kernel_lw* wid_mask/100).sum('player').mean('month')
                dRt_sw = (coso3* kernel_sw* wid_mask/100).sum('player').mean('month')

        if ker=='ERA5':
            if method in ["climatology", "running_m"]:
                dRt = (coso.groupby('time.month')*( kernel* vlevs.dp/100) ).sum('player').groupby('time.year').mean('time')
                dRt_lw = (coso.groupby('time.month')*( kernel_lw* vlevs.dp/100) ).sum('player').groupby('time.year').mean('time')
                dRt_sw = (coso.groupby('time.month')*( kernel_sw* vlevs.dp/100) ).sum('player').groupby('time.year').mean('time')
            
            else:
                dRt = (coso*( kernel* vlevs.dp / 100)).sum('player').mean('month')
                dRt_lw = (coso*( kernel_lw* vlevs.dp / 100)).sum('player').mean('month')
                dRt_sw = (coso*( kernel_sw* vlevs.dp / 100)).sum('player').mean('month')

        if ker=='SPECTRAL':
            if method in ["climatology", "running_m"]:
                dRt= (anoms.groupby('time.month')*kernel).sum(dim="player").groupby('time.year').mean('time')
            else:
                dRt = (anoms*kernel).sum(dim="player").mean('month')
                
        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} water vapor dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_water-vapor_pattern_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")
            if ker != 'SPECTRAL':
                dRt_lw.name = "dRt_lw"
                dRt_lw.attrs["description"] = f"{tip} water vapor dRt_lw pattern"
                dRt_lw.to_netcdf(cart_out + "dRt_lw_water-vapor_pattern_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")
                dRt_sw.name = "dRt_sw"
                dRt_sw.attrs["description"] = f"{tip} water vapor dRt_sw pattern"
                dRt_sw.to_netcdf(cart_out + "dRt_sw_water-vapor_pattern_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")

        if ker != 'SPECTRAL':
            dRt_glob_lw = ctl.global_mean(dRt_lw)
            wv_lw= dRt_glob_lw.compute()
            wv_lw.to_netcdf(cart_out+ "dRt_lw_water-vapor_global_" +tip+ suffix  +"-"+ker+"kernels.nc", format="NETCDF4")
            dRt_glob_sw = ctl.global_mean(dRt_sw)
            wv_sw= dRt_glob_sw.compute()
            wv_sw.to_netcdf(cart_out+ "dRt_sw_water-vapor_global_" +tip+ suffix +"-"+ker+"kernels.nc", format="NETCDF4")
            
        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt)
        wv= dRt_glob.compute()
        radiation[(tip, 'water-vapor')] = wv
        wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" + tip + suffix +"-"+ker+"kernels.nc", format="NETCDF4")
        wv.close()
        
    return radiation

#ALL RAD_ANOM COMPUTATION
def calc_anoms_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    High-level wrapper for computing radiative anomaly components using
    multiple kernel types and flexible configuration settings.

    This function orchestrates the full anomaly-computation workflow:
    it loads the configuration file, imports the kernel data, reads and
    standardizes the input dataset, determines which variables and time
    ranges to use, loads surface pressure if required, computes the
    reference climatology, and finally calls `calc_anoms` to obtain all
    anomaly components.

    Parameters
    ----------
    config_file : str or dict
        Path to a YAML configuration file or an already-loaded dict.
        The configuration must include:
        - file_paths (input/output directories)
        - anomaly_method
        - time_range / time_range_exp (optional)
        - pressure_data (for HUANG kernels)
        - use_atm_mask, save_pattern (optional)

    ker : str
        Kernel type to use (`'ERA5'`, `'HUANG'`, `'SPECTRAL'`).
        Determines which kernel loader is invoked.

    variable_mapping_file : str
        YAML file defining variable name mappings and computed variables,
        used to standardize datasets through `standardize_names`.

    Returns
    -------
    (anom_ps, anom_pal, anom_a, anom_wv) : tuple of xr.Dataset
        The four major radiative anomaly components:
        - Surface Planck anomaly
        - Atmospheric Planck anomaly
        - Albedo anomaly
        - Water vapor anomaly

    Workflow
    --------
    1. Load configuration settings.
    2. Load kernels using `load_kernel_wrapper`.
    3. Read and standardize the dataset via `read_data`.
    4. Determine which variables and time ranges to use.
    5. Load surface pressure if required by the kernel type.
    6. Load reference climatology fields with `ref_clim`.
    7. Compute anomalies via `calc_anoms`.

    Notes
    -----
    - The function checks whether climatology and experiment time ranges
      are provided and handles missing values gracefully.
    - Surface pressure is only required for HUANG kernels; other kernel
      families ignore this argument.
    - This wrapper is the main user-facing entry point for computing
      anomaly components in the workflow.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file
    
    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, variable_mapping_file)
    print("Variables to consider upload...")
    allvars = 'ts tas hus alb ta'.split()
    allvars_c = 'rlutcs rsutcs rlut rsut'.split()
    if all(var in ds.variables for var in allvars_c):
        allvars = allvars + allvars_c  # extend the list
    print("Read parameters from configuration file...")
    
    dataset_type = config.get('dataset_type', None)
    cart_out = config['file_paths'].get("output")
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    use_atm_mask = config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_atm_mask = bool(use_atm_mask)
    save_pattern = bool(save_pattern)

    # Read time ranges from config
    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # Determine usage scenario
    if time_range_exp and not time_range_clim:
        print("Only experiment time range is provided. Using it for analysis.")
    elif time_range_exp and time_range_clim:
        print(f"Using separate time ranges for climatology: {time_range_clim} and experiment: {time_range_exp}")
    elif time_range_clim and not time_range_exp:
        print("Only climatology time range is provided. Using it for both climatology and experiment.")
        time_range_exp = time_range_clim  # fallback
    else:
        print("No valid time ranges provided. Proceeding with full time range in the data.")

    # Surface pressure management
    surf_pressure = None
    if ker == 'HUANG':  # HUANG requires surface pressure
        pressure_path = config['file_paths'].get('pressure_data', None)
        
        if pressure_path:  # If pressure data is specified, load it
            print("Loading surface pressure data...")
            ps_files = sorted(glob.glob(pressure_path))  # Support for patterns like "*.nc"
            if not ps_files:
                raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
            
            surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')
            surf_pressure = standardize_names(surf_pressure, dataset_type, variable_mapping_file)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology for Rad anomaly...")
    ref_clim_data = ref_clim(config_file, allvars, ker, variable_mapping_file, allkers) 
    
    anom_ps, anom_pal, anom_a, anom_wv = calc_anoms(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, time_range_exp, method, config_file, use_atm_mask, save_pattern)

    return (anom_ps, anom_pal, anom_a, anom_wv)

def calc_anoms(ds, piok_rad, ker, allkers, cart_out, surf_pressure, time_range=None, method=None, config_file =None, use_atm_mask=True, save_pattern=False):
    """
    Computes radiative anomalies for multiple feedback components using
    precomputed kernels.

    This function evaluates the radiative anomaly associated with several
    radiative feedback components—surface Planck response, atmospheric
    Planck response, albedo, and water vapor—by applying the selected
    kernels to the input dataset. Each component is computed only if a
    preprocessed NetCDF file is not already present; otherwise the function
    loads the saved results to avoid redundant computation.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the model or observational fields needed
        to compute anomalies.

    piok_rad : xr.Dataset
        Reference climatology against which anomalies are computed.

    ker : str
        Identifier of the kernel type being used 
        (e.g., `'ERA5'`, `'HUANG'`, `'SPECTRAL'`).

    allkers : dict
        Dictionary containing the kernels previously processed by a
        kernel-loading function (e.g., `load_kernel_ERA5`).

    cart_out : str
        Output directory where anomaly components will be saved as
        NetCDF files when first computed.

    surf_pressure : xr.Dataset or None
        Surface pressure dataset, required for certain kernel types (e.g. HUANG).
        If not needed for the kernel type, can be left as `None`.

    time_range : dict or None, optional
        Time range to apply when computing anomalies (e.g. 
        `{'start': '2000', 'end': '2010'}`).  
        If `None`, the full dataset time range is used.

    method : str, optional
        Method used to compute anomalies (e.g., `'climatology'`, `'running_m'`,
        `'climatology_mean'`, etc.).  
        This determines the output filename suffix.

    config_file : str or dict, optional
        Path to a configuration YAML file or the already-loaded config dict.
        Passed to subfunctions when additional settings are required.

    use_atm_mask : bool, optional
        If `True`, masks regions not used for atmospheric anomaly computation.

    save_pattern : bool, optional
        If `True`, also saves spatial anomaly patterns (not just global means).

    Returns
    -------
    anom_ps : xr.Dataset
        Radiative anomaly from the surface Planck response.

    anom_pal : xr.Dataset
        Radiative anomaly from the atmospheric Planck response.

    anom_a : xr.Dataset
        Radiative anomaly from surface albedo changes.

    anom_wv : xr.Dataset
        Radiative anomaly from water vapor changes.

    Notes
    -----
    - The function avoids recomputation by checking whether output files
      already exist in `cart_out`.
    - The actual computation is delegated to the respective functions:
        - `Rad_anomaly_planck_surf`
      	- `Rad_anomaly_planck_atm_lr`
      	- `Rad_anomaly_albedo`
      	- `Rad_anomaly_wv`
    - Output files follow the naming pattern:
          dRt_<component>_global_clr_<method>-<ker>kernels.nc
    """
    # define suffix for saved files based on method only
    suffix = f"_{method}"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+ suffix +"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_ps = Rad_anomaly_planck_surf(ds, piok_rad, ker, allkers, cart_out, time_range, method, save_pattern)
    else:
        anom_ps = xr.open_dataset(path)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+ suffix +"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_pal = Rad_anomaly_planck_atm_lr(ds, piok_rad, ker, allkers, cart_out, surf_pressure, time_range, config_file, method, use_atm_mask, save_pattern)
    else:
        anom_pal = xr.open_dataset(path)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+ suffix +"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_a = Rad_anomaly_albedo(ds, piok_rad, ker, allkers, cart_out, time_range, method, save_pattern)
    else:
        anom_a = xr.open_dataset(path)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+ suffix +"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_wv = Rad_anomaly_wv(ds, piok_rad, ker, allkers, cart_out, surf_pressure, time_range, config_file, method, use_atm_mask, save_pattern)
    else:
        anom_wv = xr.open_dataset(path)  

    return anom_ps, anom_pal, anom_a, anom_wv 

##FEEDBACK COMPUTATION
def calc_fb_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    Wrapper function to compute radiative and cloud feedbacks based on the provided configuration file and kernel type.
    
    This function orchestrates the full workflow for calculating climate
    feedbacks based on a configuration file. It loads kernels, climate datasets,
    reference climatology, and surface pressure (if required), then calls
    `calc_fb` to compute radiative and cloud feedbacks. Optional spatial
    feedback patterns can also be saved.

    Parameters
    ----------
    config_file : str or dict
        Path to a YAML configuration file or an already-loaded configuration
        dictionary. Must include:
        - file_paths (input/output directories)
        - anomaly_method
        - time_range / time_range_exp (optional)
        - pressure_data (required for HUANG kernels)
        - use_atm_mask, save_pattern (optional)

    ker : str
        Kernel type to use (`'ERA5'`, `'HUANG'`, `'SPECTRAL'`). Determines
        which kernel loader is invoked and whether surface pressure is required.

    variable_mapping_file : str
        Path to a YAML file defining variable name mappings and computed
        variables, used to standardize datasets via `standardize_names`.

    Returns
    -------
    fb_coef : dict
        Dictionary of radiative feedback coefficients for each component
        and atmospheric condition.

    fb_cloud : xarray.DataArray
        Cloud feedback data array computed from global mean regressions.

    fb_cloud_err : xarray.DataArray
        Error associated with the cloud feedback calculation.

    fb_pattern : dict, optional
        Dictionary of spatial feedback patterns and standard errors for each
        component and atmospheric condition. Returned only if `save_pattern=True`.

    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, variable_mapping_file)
    print("Variables to consider upload...")
    allvars = 'ts tas hus alb ta'.split()
    allvars_c = 'rlutcs rsutcs rlut rsut'.split()
    if all(var in ds.variables for var in allvars_c):
        allvars = allvars + allvars_c  
    print("Read parameters from configuration file...")
    
    dataset_type = config.get('dataset_type', None)
    cart_out = config['file_paths'].get("output")
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    use_atm_mask=config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_atm_mask = bool(use_atm_mask)
    save_pattern = bool(save_pattern)
    num=config.get("num_year_regr", 10)

    # Read time ranges from config
    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # Determine usage scenario
    if time_range_exp and not time_range_clim:
        print("Only experiment time range is provided. Using it for analysis.")
    elif time_range_exp and time_range_clim:
        print(f"Using separate time ranges for climatology: {time_range_clim} and experiment: {time_range_exp}")
    elif time_range_clim and not time_range_exp:
        print("Only climatology time range is provided. Using it for both climatology and experiment.")
        time_range_exp = time_range_clim  # fallback
    else:
        print("No valid time ranges provided. Proceeding with full time range in the data.")

    # Surface pressure management
    surf_pressure = None
    if ker == 'HUANG':  # HUANG requires surface pressure
        pressure_path = config['file_paths'].get('pressure_data', None)
        
        if pressure_path:  # If pressure data is specified, load it
            print("Loading surface pressure data...")
            ps_files = sorted(glob.glob(pressure_path))  # Support for patterns like "*.nc"
            if not ps_files:
                raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
            
            surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')
            surf_pressure = standardize_names(surf_pressure, dataset_type, variable_mapping_file)
        else:
            raise ValueError("HUANG kernels require surface pressure data, but none was provided.")
        
    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config_file, allvars, ker, variable_mapping_file, allkers=allkers) 

    outputs = calc_fb(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure,
                    time_range_exp, method, config_file, use_atm_mask, save_pattern, num)

    return outputs
        
    
def calc_fb(ds, piok, ker, allkers, cart_out, surf_pressure, time_range=None, method=None, config_file =None, use_atm_mask=True, save_pattern=False, num=10):
    """
    Compute radiative and cloud feedbacks using preprocessed kernels.

    This function calculates feedbacks from multiple radiative components:
    - Planck surface
    - Planck atmospheric
    - Albedo
    - Water vapor

    It also computes cloud feedbacks by performing linear regression between
    global mean temperature anomalies and radiative anomalies. Optionally,
    spatial feedback patterns can be computed and saved.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing climate model output variables.

    piok : xarray.Dataset
        Reference climatology dataset used for computing radiative anomalies.

    ker : str
        Kernel type (e.g., `'HUANG'`) that determines specific calculations.

    allkers : dict
        Dictionary of kernels for the different radiative feedback components.

    cart_out : str
        Output directory where computed feedbacks and patterns are saved.

    surf_pressure : xarray.Dataset or None
        Surface pressure dataset, required for certain kernel types.

    time_range : dict or None, optional
        Time range used for computing anomalies, e.g.,
        `{'start': '2000', 'end': '2010'}`. If `None`, full dataset range
        is used.

    method : str, optional
        Method used for anomaly computation. Options include:
        - `"climatology"`: monthly averaged climatology
        - `"running_m"`: 21-year running mean
        - `"climatology_mean"`: time-averaged anomaly
        - `"running_m_mean"`: running mean anomaly

    config_file : str or dict, optional
        Configuration settings passed to subfunctions if needed.

    use_atm_mask : bool, optional
        If `True`, masks regions not used in atmospheric anomaly calculations.

    save_pattern : bool, optional
        If `True`, computes and saves spatial feedback patterns and
        standard errors.

    num : int, optional
        Number of years per regression block when grouping anomalies for
        decadal averaging (default 10).

    Returns
    -------
    fb_coef : dict
        Radiative feedback coefficients for each component and condition
        (`'clr'` or `'cld'`).

    fb_cloud : xarray.DataArray
        Cloud feedback data array derived from global mean regressions.

    fb_cloud_err : xarray.DataArray
        Error associated with the cloud feedback calculation.

    fb_pattern : dict or None
        Dictionary of spatial feedback slopes and standard errors for each
        component and condition. Only returned if `save_pattern=True`.
    """

    if 'tas' not in piok:
        raise ValueError("Reference climatology for 'tas' is missing in piok. Ensure 'tas' is included in 'allvars' when calling ref_clim.")
    
    # define suffix for saved files based on method only
    suffix = f"_{method}"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, time_range, method, save_pattern)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, time_range, method, save_pattern)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_cld"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_atm_lr(ds, piok, ker, allkers, cart_out, surf_pressure, time_range, config_file, method, use_atm_mask, save_pattern)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_wv(ds, piok, ker, allkers, cart_out, surf_pressure, time_range, config_file, method, use_atm_mask, save_pattern)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()
    fb_pattern = dict()

    #compute gtas
    k=allkers[('cld', 't')]
    if time_range is not None:
        var_tas = ds['tas'].sel(time=slice(time_range['start'], time_range['end'])) 
        var_tas= ctl.regrid_dataset(var_tas, k.lat, k.lon)  
    else:
        var_tas= ctl.regrid_dataset(ds['tas'], k.lat, k.lon) 

    anoms_tas = compute_anomalies(var_tas, piok['tas'], method=method, nonlinear=False, check=True)

    gtas = ctl.global_mean(anoms_tas).groupby('time.year').mean('time')
    start_year = int(gtas.year.min()) 
    gtas = gtas.groupby((gtas.year-start_year) // num * num).mean()

    if save_pattern:
        gtas = gtas.chunk({'year': -1})

    if save_pattern:
        fb_pattern = {}
    else:
        fb_pattern = None

    print('feedback calculation...')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ suffix +"-"+ker+"kernels.nc",  use_cftime=True)
            start_year = int(feedbacks.year.min())
            feedback=feedbacks.groupby((feedbacks.year-start_year) // num * num).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res

            if save_pattern:
                print(f"Computing spatial feedback pattern for {tip}-{fbn}...")
                # Open the dRt pattern
                feedbacks_pattern = xr.open_dataarray(cart_out+"dRt_"+fbn+"_pattern_"+tip + suffix +"-"+ker+"kernels.nc", use_cftime=True) 
                start_year = int(feedbacks_pattern.year.min())
                feedbacks_pattern_dec = feedbacks_pattern.groupby((feedbacks_pattern.year - start_year) // num * num).mean('year')
                feedbacks_pattern_dec = feedbacks_pattern_dec.chunk({'year': -1})
                # Perform regression at each grid point
                slope, stderr = regress_pattern_vectorized(feedbacks_pattern_dec, gtas)
                fb_pattern[(tip, fbn)] = (slope, stderr)
                slope.to_netcdf(cart_out + "feedback_pattern_"+ fbn +"_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")
                stderr.to_netcdf(cart_out + "feedback_pattern_error_"+ fbn +"_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")
    
    #cloud
    required_cloud_vars = {"rlut", "rsut", "rlutcs", "rsutcs"}
    can_compute_cloud = required_cloud_vars.issubset(set(ds.variables))
    if can_compute_cloud:
        print('cloud feedback calculation...')
        fb_cloud, fb_cloud_err = feedback_cloud(ds, piok, fb_coef, gtas, time_range, num)
    else:
        print("Cloud variables not found → skipping cloud feedback.")
        fb_cloud = None
        fb_cloud_err = None
    
    return {
        "fb_coeffs": fb_coef,
        "fb_cloud": fb_cloud,
        "fb_cloud_err": fb_cloud_err,
        "fb_pattern": fb_pattern if save_pattern else None,
    }

def calc_fb_interannual_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    Wrapper function for computing interannual radiative and cloud feedbacks.

    This function manages the full workflow for interannual feedback calculation,
    using a running-mean approach to remove decadal trends from global mean
    temperature and radiative anomalies. It loads kernels, climate datasets,
    reference climatology, and surface pressure (if required), then calls
    `calc_fb_interannual` to compute the feedbacks. Optionally, spatial feedback
    patterns can also be saved.

    Parameters
    ----------
    config_file : str or dict
        Path to a YAML configuration file or a preloaded configuration dictionary.
        Must include:
        - file_paths (input/output directories)
        - anomaly_method
        - time_range / time_range_exp (optional)
        - pressure_data (required for HUANG kernels)
        - use_atm_mask, save_pattern (optional)
        - num_running_years_trend (number of years for running mean)

    ker : str
        Kernel type to use (`'ERA5'`, `'HUANG'`, `'SPECTRAL'`). Determines
        which kernel loader is invoked and whether surface pressure is required.

    variable_mapping_file : str
        Path to a YAML file defining variable name mappings and computed
        variables, used to standardize datasets via `standardize_names`.

    Returns
    -------
    fb_coef : dict
        Dictionary of interannual radiative feedback coefficients for each
        component and atmospheric condition.

    fb_cloud : xarray.DataArray
        Cloud feedback data array computed using interannual variations.

    fb_cloud_err : xarray.DataArray
        Error associated with the interannual cloud feedback calculation.

    fb_pattern : dict, optional
        Dictionary of spatial interannual feedback slopes and standard errors
        for each component and atmospheric condition. Returned only if
        `save_pattern=True`.
   
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, variable_mapping_file)
    print("Variables to consider upload...")
    allvars = 'ts tas hus alb ta'.split()
    allvars_c = 'rlutcs rsutcs rlut rsut'.split()
    if all(var in ds.variables for var in allvars_c):
        allvars = allvars + allvars_c  
    print("Read parameters from configuration file...")


    dataset_type = config.get('dataset_type', None)
    cart_out = config['file_paths'].get("output")
    method = config.get("anomaly_method")
    if method is None:
        raise ValueError("The config file must specify 'anomaly_method' (e.g., 'climatology', 'running_m', 'climatology_mean', 'running_m_mean')")
    use_atm_mask=config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_atm_mask = bool(use_atm_mask)
    save_pattern = bool(save_pattern)
    running_years=config.get("num_running_years_trend", 10)
    
    # Read time ranges from config
    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # Determine usage scenario
    if time_range_exp and not time_range_clim:
        print("Only experiment time range is provided. Using it for analysis.")
    elif time_range_exp and time_range_clim:
        print(f"Using separate time ranges for climatology: {time_range_clim} and experiment: {time_range_exp}")
    elif time_range_clim and not time_range_exp:
        print("Only climatology time range is provided. Using it for both climatology and experiment.")
        time_range_exp = time_range_clim  # fallback
    else:
        print("No valid time ranges provided. Proceeding with full time range in the data.")

    # Surface pressure management
    surf_pressure = None
    if ker == 'HUANG':  # HUANG requires surface pressure
        pressure_path = config['file_paths'].get('pressure_data', None)
        
        if pressure_path:  # If pressure data is specified, load it
            print("Loading surface pressure data...")
            ps_files = sorted(glob.glob(pressure_path))  # Support for patterns like "*.nc"
            if not ps_files:
                raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
            
            surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')
            surf_pressure = standardize_names(surf_pressure, dataset_type, variable_mapping_file)
        else:
            raise ValueError("HUANG kernels require surface pressure data, but none was provided.")
        
    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config_file, allvars, ker, variable_mapping_file, allkers=allkers) 

    outputs = calc_fb_interannual(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, time_range_exp, method, config_file, use_atm_mask, save_pattern, running_years)

    return outputs


def calc_fb_interannual(ds, piok, ker, allkers, cart_out, surf_pressure, time_range=None, method=None, config_file =None, use_atm_mask=True, save_pattern=False, running_years=25):   
    """
    Compute interannual radiative and cloud feedbacks using a running-mean approach.

    This function calculates feedbacks from multiple radiative components:
    - Planck surface
    - Planck atmospheric
    - Albedo
    - Water vapor

    Unlike the standard feedback calculation, this function removes long-term
    trends using a running mean over `running_years` to isolate interannual
    variability. Cloud feedbacks are computed via linear regression between
    detrended global mean temperature anomalies and radiative anomalies. Spatial
    feedback patterns can also be computed and saved if requested.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing climate model output variables.

    piok : xarray.Dataset
        Reference climatology dataset used for computing radiative anomalies.

    ker : str
        Kernel type (e.g., `'HUANG'`) that determines specific calculations.

    allkers : dict
        Dictionary of kernels for the different radiative feedback components.

    cart_out : str
        Output directory where computed feedbacks and patterns are saved.

    surf_pressure : xarray.Dataset or None
        Surface pressure dataset, required for certain kernel types.

    time_range : dict or None, optional
        Time range used for computing anomalies, e.g.,
        `{'start': '2000', 'end': '2010'}`. If `None`, full dataset range
        is used.

    method : str, optional
        Method used for anomaly computation. Options include:
        - `"climatology"`: monthly averaged climatology
        - `"running_m"`: 21-year running mean
        - `"climatology_mean"`: time-averaged anomaly
        - `"running_m_mean"`: running mean anomaly

    config_file : str or dict, optional
        Configuration settings passed to subfunctions if needed.

    use_atm_mask : bool, optional
        If `True`, masks regions not used in atmospheric anomaly calculations.

    save_pattern : bool, optional
        If `True`, computes and saves spatial feedback patterns and
        standard errors.

    running_years : int, optional
        Number of years for the running mean used to remove trends
        (default is 25).

    Returns
    -------
    fb_coef : dict
        Interannual radiative feedback coefficients for each component and
        condition (`'clr'` or `'cld'`).

    fb_cloud : xarray.DataArray
        Cloud feedback data array derived from interannual regressions.

    fb_cloud_err : xarray.DataArray
        Error associated with the interannual cloud feedback calculation.

    fb_pattern : dict or None
        Dictionary of spatial feedback slopes and standard errors for each
        component and condition. Only returned if `save_pattern=True`.
    Notes
    -----
    - Removes long-term trends via a running mean before computing feedbacks.
    - Performs linear regression between detrended global mean temperature
      anomalies and radiative anomalies for each component.
    - Handles optional spatial feedback calculation.
    - Useful for analyzing interannual variability in climate feedbacks.
    """

    if 'tas' not in piok:
        raise ValueError("Reference climatology for 'tas' is missing in piok. Ensure 'tas' is included in 'allvars' when calling ref_clim.")
    
    # define suffix for saved files based on method only
    suffix = f"_{method}"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, time_range, method, save_pattern)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, time_range, method, save_pattern)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_cld"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_atm_lr(ds, piok, ker, allkers, cart_out, surf_pressure, time_range, config_file, method, use_atm_mask, save_pattern)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+suffix+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_wv(ds, piok, ker, allkers, cart_out, surf_pressure, time_range, config_file, method, use_atm_mask, save_pattern)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()
    fb_pattern = dict()

    #compute gtas
    k=allkers[('cld', 't')]
    if time_range is not None:
        var_tas = ds['tas'].sel(time=slice(time_range['start'], time_range['end'])) 
        var_tas= ctl.regrid_dataset(var_tas, k.lat, k.lon)  
    else:
        var_tas= ctl.regrid_dataset(ds['tas'], k.lat, k.lon)

    anoms_tas = compute_anomalies(var_tas, piok['tas'], method=method, nonlinear=False, check=True) 
    
    gtas = ctl.global_mean(anoms_tas).groupby('time.year').mean('time')
    trend = ctl.running_mean(gtas, running_years)
    temp=gtas-trend

    if save_pattern:
        fb_pattern = {}
    else:
        fb_pattern = None

    print('feedback calculation...')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ suffix +"-"+ker+"kernels.nc",  use_cftime=True)
            trend_fed=ctl.running_mean(feedbacks, running_years)
            inter=feedbacks-trend_fed

            res = stats.linregress(temp,inter)
            fb_coef[(tip, fbn)] = res
            if save_pattern:
                print(f"Computing spatial feedback pattern for {tip}-{fbn}...")
                # Open the dRt pattern
                feedbacks_pattern = xr.open_dataarray(cart_out+"dRt_"+fbn+"_pattern_"+tip+suffix+"-"+ker+"kernels.nc", use_cftime=True)
                trend_patt=ctl.running_mean(feedbacks_pattern, running_years)
                feedbacks_pattern_dec=feedbacks_pattern-trend_patt                

                feedbacks_pattern_dec = feedbacks_pattern_dec.chunk({'year': -1})
                gtas1 = temp.chunk({'year': -1})
                # Perform regression at each grid point
                slope, stderr = regress_pattern_vectorized(feedbacks_pattern_dec, gtas1)
                fb_pattern[(tip, fbn)] = (slope, stderr)
                slope.to_netcdf(cart_out + "feedback_pattern_"+ fbn +"_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")
                stderr.to_netcdf(cart_out + "feedback_pattern_error_"+ fbn +"_" + tip + suffix + "-" + ker + "kernels.nc", format="NETCDF4")

    #cloud
    required_cloud_vars = {"rlut", "rsut", "rlutcs", "rsutcs"}
    can_compute_cloud = required_cloud_vars.issubset(set(ds.variables))
    if can_compute_cloud:
        print('cloud interannual feedback calculation...')
        fb_cloud, fb_cloud_err = feedback_cloud_interannual(ds, piok, fb_coef, temp, time_range, running_years)
    else:
        print("Cloud variables not found → skipping cloud feedback.")
        fb_cloud = None
        fb_cloud_err = None
    
    return {
        "fb_coeffs": fb_coef,
        "fb_cloud": fb_cloud,
        "fb_cloud_err": fb_cloud_err,
        "fb_pattern": fb_pattern if save_pattern else None,
    }
    

#CLOUD FEEDBACK shell 2008
def feedback_cloud(ds, piok, fb_coef, surf_anomaly, time_range=None, num=10):
   #questo va testato perchè non sono sicura che funzionino le cose con pimean (calcolato con climatology ha il groupby.month di cui qui non si tiene conto)
    """
    Compute cloud radiative feedback strength and associated error.

    This function calculates cloud feedback as the difference between net
    radiative flux anomalies under all-sky and clear-sky conditions. It
    performs linear regression between global mean surface temperature anomalies
    and radiative fluxes to obtain the cloud feedback slope and its uncertainty.
    Radiative anomalies are regridded to a common spatial grid and optionally
    restricted to a specific time range.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing radiative flux variables:
        - 'rlut' : Outgoing longwave radiation at top of atmosphere (all-sky)
        - 'rsut' : Reflected shortwave radiation (all-sky)
        - 'rlutcs' : Outgoing longwave radiation (clear-sky)
        - 'rsutcs' : Reflected shortwave radiation (clear-sky)

    piok : xarray.Dataset
        Reference climatology used to compute radiative anomalies.

    fb_coef : dict
        Dictionary of radiative feedback coefficients for non-cloud components
        (Planck, albedo, water vapor, lapse rate).

    surf_anomaly : xarray.DataArray
        Global mean surface temperature anomalies used for regression.
        Expected to be pre-processed as:
        ```
        gtas = ctl.global_mean(tas_anomaly).groupby('time.year').mean('time')
        gtas = gtas.groupby((gtas.year-1)//10*10).mean()
        ```

    time_range : dict, optional
        Time range for selecting data from `ds`, e.g.,
        `{'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}`. Default is None
        (full time range is used).

    num : int, optional
        Number of years to aggregate in regressions (default is 10).

    Returns
    -------
    fb_cloud : float
        Cloud radiative feedback strength (W/m²/K).

    fb_cloud_err : float
        Estimated standard error of the cloud feedback calculation (W/m²/K).
    """
    if not (ds['rlut'].dims == ds['rsutcs'].dims):
        raise ValueError("Error: The spatial grids ('lon' and 'lat') datasets must match.")
    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    
    if time_range is not None:
        rlut=ds['rlut'].sel(time=slice(time_range['start'], time_range['end']))
        rsut=ds['rsut'].sel(time=slice(time_range['start'], time_range['end']))
        rsutcs = ds['rsutcs'].sel(time=slice(time_range['start'], time_range['end']))
        rlutcs = ds['rlutcs'].sel(time=slice(time_range['start'], time_range['end']))
    else:
        rlut=ds['rlut']
        rsut=ds['rsut']
        rsutcs = ds['rsutcs']
        rlutcs = ds['rlutcs']

    N = - rlut - rsut
    N0 = - rsutcs - rlutcs
    crf = (N0 - N) 

    lat_target = np.linspace(-90, 90, 73)
    lon_target = np.linspace(0, 357.5, 144)
    crf = ctl.regrid_dataset(crf, lat_target, lon_target)

    crf_glob = ctl.global_mean(crf).groupby('time.year').mean('time')
    N_glob = ctl.global_mean(N).groupby('time.year').mean('time')
    N0_glob = ctl.global_mean(N0).groupby('time.year').mean('time')
    start_year = int(crf_glob.year.min()) 
    crf_glob = crf_glob.groupby((crf_glob.year-start_year) // num * num).mean()
    start_year = int(N_glob.year.min()) 
    N_glob = N_glob.groupby((N_glob.year-start_year) // num * num).mean()
    start_year = int(N0_glob.year.min()) 
    N0_glob = N0_glob.groupby((N0_glob.year-start_year) // num * num).mean()

    res_N = stats.linregress(surf_anomaly, N_glob)
    res_N0 = stats.linregress(surf_anomaly, N0_glob)
    res_crf = stats.linregress(surf_anomaly, crf_glob)

    F0 = res_N0.intercept + piok[('rlutcs')] + piok[('rsutcs')] 
    F = res_N.intercept + piok[('rlut')] + piok[('rsut')]
    F0.compute()
    F.compute()

    F_glob = ctl.global_mean(F)
    F0_glob = ctl.global_mean(F0)
    F_glob = F_glob.compute()
    F0_glob = F0_glob.compute()

    fb_cloud = -res_crf.slope + np.nansum([fb_coef[( 'clr', fbn)].slope - fb_coef[('cld', fbn)].slope for fbn in fbnams]) #letto in Caldwell2016

    fb_cloud_err = np.sqrt(res_crf.stderr**2 + np.nansum([fb_coef[('cld', fbn)].stderr**2 for fbn in fbnams]))

    
    return fb_cloud, fb_cloud_err

def feedback_cloud_interannual(ds, piok, fb_coef, surf_anomaly, time_range=None, running_years=25):
    """
    Compute interannual cloud radiative feedback and associated error.

    This function calculates cloud feedback on interannual timescales by first
    removing multi-year running mean trends from radiative fluxes and temperature
    anomalies. It performs linear regression between detrended global mean
    surface temperature anomalies and detrended cloud radiative fluxes
    (difference between all-sky and clear-sky net radiation) to estimate the
    cloud feedback slope and its uncertainty.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing radiative flux variables:
        - 'rlut' : Outgoing longwave radiation at top of atmosphere (all-sky)
        - 'rsut' : Reflected shortwave radiation (all-sky)
        - 'rlutcs' : Outgoing longwave radiation (clear-sky)
        - 'rsutcs' : Reflected shortwave radiation (clear-sky)

    piok : xarray.Dataset
        Reference climatology dataset used for baseline radiative fluxes.

    fb_coef : dict
        Dictionary of radiative feedback coefficients for non-cloud components
        (Planck, albedo, water vapor, lapse rate) used in the cloud feedback
        calculation.

    surf_anomaly : xarray.DataArray
        Global mean surface temperature anomalies (detrended) used for regression.

    time_range : dict, optional
        Time range for selecting data from `ds`, e.g.,
        `{'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}`. Default is None
        (full time range is used).

    running_years : int, optional
        Number of years to apply running mean for detrending interannual
        variations (default is 25 years).

    Returns
    -------
    fb_cloud : float
        Interannual cloud radiative feedback strength (W/m²/K).

    fb_cloud_err : float
        Estimated standard error of the interannual cloud feedback calculation
        (W/m²/K).
    """

    if not (ds['rlut'].dims == ds['rsutcs'].dims):
        raise ValueError("Error: The spatial grids ('lon' and 'lat') datasets must match.")
    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    
    if time_range is not None:
        rlut=ds['rlut'].sel(time=slice(time_range['start'], time_range['end']))
        rsut=ds['rsut'].sel(time=slice(time_range['start'], time_range['end']))
        rsutcs=ds['rsutcs'].sel(time=slice(time_range['start'], time_range['end']))
        rlutcs = ds['rlutcs'].sel(time=slice(time_range['start'], time_range['end']))
    else:
        rlut=ds['rlut']
        rsut=ds['rsut']
        rsutcs = ds['rsutcs']
        rlutcs = ds['rlutcs']

    N = - rlut - rsut
    N0 = - rsutcs - rlutcs

    crf = (N0 - N) 
    crf = crf.groupby('time.year').mean('time')

    N = N.groupby('time.year').mean('time')
    N0 = N0.groupby('time.year').mean('time')

    crf_glob = ctl.global_mean(crf).compute()
    trend_crf_glob = ctl.running_mean(crf_glob, running_years)
    N_glob = ctl.global_mean(N).compute()
    trend_N_glob = ctl.running_mean(N_glob, running_years)
    N0_glob = ctl.global_mean(N0).compute()
    trend_N0_glob = ctl.running_mean(N0_glob, running_years)
    

    res_N = stats.linregress(surf_anomaly, (N_glob-trend_N_glob))
    res_N0 = stats.linregress(surf_anomaly, (N0_glob-trend_N0_glob))
    res_crf = stats.linregress(surf_anomaly, (crf_glob-trend_crf_glob))

    F0 = res_N0.intercept + piok[('rlutcs')] + piok[('rsutcs')] 
    F = res_N.intercept + piok[('rlut')] + piok[('rsut')]
    F0.compute()
    F.compute()

    F_glob = ctl.global_mean(F)
    F0_glob = ctl.global_mean(F0)
    F_glob = F_glob.compute()
    F0_glob = F0_glob.compute()

    fb_cloud = -res_crf.slope + np.nansum([fb_coef[( 'clr', fbn)].slope - fb_coef[('cld', fbn)].slope for fbn in fbnams]) #letto in Caldwell2016

    fb_cloud_err = np.sqrt(res_crf.stderr**2 + np.nansum([fb_coef[('cld', fbn)].stderr**2 for fbn in fbnams]))

    
    return fb_cloud, fb_cloud_err
