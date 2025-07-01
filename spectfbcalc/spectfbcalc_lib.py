#!/usr/bin/python
# -*- coding: utf-8 -*-

### Our library!

##### Package imports

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
def load_spectral_kernel(cart_k: str, cart_out: str):
    """Loads and preprocesses STE kernels for further analysis."""  
    
    tips = ['cs','cld']
    vnams = ['temp_jac', 'ts_jac', 'wv_jac', 'ozo_jac','ch4_jac', 'n2o_jac', 'co2_jac']
    finam = 'spectral_kernel_ste_{}.nc'
    
    allkers = dict()
    
    for tip in tips:
        
        # Load kernels NetCDF files (3 years mean 2008-2010)
        kernels = xr.open_dataset(cart_k + finam.format(tip),chunks={'time':12})
        
        for vna_local in vnams:
            print(vna_local)
            if vna_local =='temp_jac':
                kernels[vna_local] = kernels[vna_local].rename({'lev': 'player'})
                vna = 't'
            if vna_local =='ts_jac':
                vna = 'ts'
            if vna_local =='wv_jac':
                kernels[vna_local] = kernels[vna_local].rename({'lev': 'player'})
                vna = 'wv_lw'
            if vna_local =='ozo_jac':
                kernels[vna_local] = kernels[vna_local].rename({'lev': 'player'})
                vna = 'o3_lw'
            if vna_local =='ch4_jac':
                kernels[vna_local] = kernels[vna_local].rename({'lev': 'player'})
                vna = 'ch4_lw'
            if vna_local =='n2o_jac':
                kernels[vna_local] = kernels[vna_local].rename({'lev': 'player'})
                vna = 'n2o'
            if vna_local =='co2_jac':
                kernels[vna_local] = kernels[vna_local].rename({'lev': 'player'})
                vna = 'co2_lw'
            if tip == 'cs' or 'forum_cs':
                tip = 'clr'
            
            allkers[(tip, vna)] = kernels[vna_local].sel(freq=slice(650,2750)) #frequency selection 650-2750 cm-1 
            
            #Save all kernels, t kernel and pressure levels to an external file
            k = allkers[('cld', 't')]
            vlevs = xr.load_dataset( cart_k + finam.format(tip),chunks={'time':12})['lev']
            vlevs = vlevs.rename({'lev': 'player'})
            pickle.dump(k, open(cart_out + 'k_STE.p', 'wb'))
            pickle.dump(allkers, open(cart_out + 'allkers_STE.p', 'wb'))
            pickle.dump(vlevs, open(cart_out + 'vlevs_ERA5.p', 'wb')) #save vlevs
    
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

    if ker=='ERA5':
        cart_k = config['kernels']['era5']['path_input']
        cart_out = config['kernels']['era5']['path_output']
        finam = config['kernels']['era5']['filename_template']

    if ker=='HUANG':
       cart_k = config['kernels']['huang']['path_input']
       cart_out = config['kernels']['huang']['path_output']
       finam = config['kernels']['huang']['filename_template']
    
     
    allkers = load_kernel(ker, cart_k, cart_out, finam)

    return allkers

def load_kernel(ker, cart_k, cart_out, finam):
    if ker=='ERA5':
         allkers=load_kernel_ERA5(cart_k, cart_out, finam)
    if ker=='HUANG':
         allkers=load_kernel_HUANG(cart_k, cart_out, finam)

    return allkers

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
    use_climatology = config.get("use_climatology", True)   
    use_climatology = bool(use_climatology)
    
    if not filin_pi:
        raise ValueError("Error: the 'reference_dataset' path is not specified in the configuration file.")

    ds_list = [xr.open_mfdataset(filin_pi, combine='by_coords', compat='no_conflicts', use_cftime=True, chunks={'time': time_chunk})]
    
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
        if vnams in ['rsut', 'rlut', 'rsutcs', 'rlutcs']:
            piok[vnams] = climatology(ds_ref, allkers, vnams, time_range_clim, True, time_chunk)
        else:
            piok[vnams] = climatology(ds_ref, allkers, vnams, time_range_clim, use_climatology, time_chunk)


    return piok

def climatology(filin_pi:str,  allkers, allvars:str, time_range=None, use_climatology=True, time_chunk=12):
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
    use_climatology : bool, optional (default=True)
        If True, computes the mean climatology over the entire time period.
        If False, computes a running mean (e.g., 252-month moving average) over the selected time period.
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

            if use_climatology:
                var_mean = var.groupby('time.month').mean()
                var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)
                pimean[vnam] = var_mean
            else:
                pimean[vnam] = ctl.regrid_dataset(var, k.lat, k.lon)

        piok = pimean['rsus'] / pimean['rsds']
        if not use_climatology:
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

        if use_climatology:
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


############# NEW FUNCTIONS FOR SPECTRAL KERNELS ########################
# LOAD KERNEL PLANK SURFACE


def Rad_anomaly_spectral_planck_surf_core(ds, allkers, ker:str, use_climatology=True, ref_clim=None, lat_range=None, lon_range=None, time_chunk=12):

    """  Computes the radiative anomalies due to surface temperatures using surface temperature anomalies and precomputed kernels.

    Parameters:
    -----------
    ds: dataset variables

    allkers: dataset kernels

    ref_clim : climatology ---- if you provide climatology dataset use_climatology=true

    ker:str to distinguish between kernel datasets (STE or HUANG)

    time_chunk : int, optional (default=12)
        Chunk size for loading data with xarray to optimize memory usage.

    Returns:
    --------
    feedbacks : dict
        A dictionary containing the radiative anomalies due to surface temperatures for clear-sky 
        (`clr`) and all-sky (`cld`) conditions. Keys of the dictionary:
        - `('clr', 'planck-surf')`: Clear-sky surface Planck feedback.
        - `('cld', 'planck-surf')`: All-sky surface Planck feedback. """
   
  
    feedbacks=dict()

    var = ds.ts

    if use_climatology:
        if ref_clim is None:
            raise ValueError("ref_clim must be provide if use_climatology is True")
        var_clim = ref_clim.ts
    else:
        var_clim = var.groupby('time.month').mean()


    for tip in ['clr', 'cld']:
        kernel = allkers[(tip, 'ts')]

    # Condizione Lat, Lon
        if lat_range is not None and lon_range is not None:

            # Seleziona latitudine e longitudine per var
            var = ctl.sel_area_xr(var,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))
            var_clim = ctl.sel_area_xr(var_clim,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))
            kernel = ctl.sel_area_xr(kernel,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))

        anoms =  var.groupby('time.month') - var_clim
        anoms_monthly = anoms.groupby('time.month')

        # Prodotto 

        if ker=='STE':

            dRt = anoms_monthly*kernel 
            dRt_glob = ctl.global_mean(dRt)

        if ker=='HUANG':

            dRt = anoms_monthly*kernel 
            dRt_glob = ctl.global_mean(dRt)

        planck= dRt_glob.compute()
        feedbacks[(tip, 'planck-surf')] = planck

    return(feedbacks)



def Rad_anomaly_spectral_planck_atmo_core(ds, allkers, ker:str, use_climatology=True, ref_clim=None, lat_range=None, lon_range=None, time_chunk=12):

    """  Computes the radiative anomalies due to atmospheric temperatures using atmospheric temperature anomalies and precomputed kernels.

      Parameters:
    -----------
    ds: dataset variables

    allkers: dataset kernels

    ref_clim : climatology ---- if you provide climatology dataset use_climatology=true

    ker:str to distinguish between kernel datasets (STE or HUANG)

    time_chunk : int, optional (default=12)
        Chunk size for loading data with xarray to optimize memory usage.



    Returns:
    --------
    feedbacks : dict
        A dictionary containing radiative anomalies due to atmospheric temperatures for clear-sky 
        (`clr`) and all-sky (`cld`) conditions. Keys of the dictionary:
        - `('clr', 'planck-surf')`: Clear-sky surface Planck feedback.
        - `('cld', 'planck-surf')`: All-sky surface Planck feedback. """
   

    var = ds.t
  
    feedbacks=dict()

    if use_climatology:
        if ref_clim is None:
            raise ValueError("ref_clim must be provide if use_climatology is True")
        var_clim = ref_clim.t
    else:
        var_clim = var.groupby('time.month').mean()

    for tip in ['clr', 'cld']:
        kernel = allkers[(tip, 't')]

    # Condizione Lat, Lon
        if lat_range is not None and lon_range is not None:

            # Seleziona latitudine e longitudine per var
                var = ctl.sel_area_xr(var,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))
                var_clim = ctl.sel_area_xr(var_clim,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))
                kernel = ctl.sel_area_xr(kernel,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))


        anoms =  var.groupby('time.month') - var_clim
        anoms_monthly = anoms.groupby('time.month')

        # Prodotto 

        if ker=='STE':

            dRt = anoms_monthly*kernel 
            dRt = dRt.sum(dim="level")
            dRt_glob = ctl.global_mean(dRt)

        if ker=='HUANG':

            dRt = anoms_monthly*kernel 
            dRt_glob = ctl.global_mean(dRt)

        planck = dRt_glob.compute()
        feedbacks[(tip, 'planck-atmo')] = planck

    return(feedbacks)


def Rad_anomaly_spectral_wv_core(ds, allkers, ker:str, use_climatology=True, ref_clim=None, lat_range=None, lon_range=None, time_chunk=12):

    """  Computes the radiative anomalies due to water vapor concentration using  water vapor concentration anomalies and precomputed kernels.

       Parameters:
    -----------
    ds: dataset variables

    allkers: dataset kernels

    ref_clim : climatology ---- if you provide climatology dataset use_climatology=true

    ker:str to distinguish between kernel datasets (STE or HUANG)

    time_chunk : int, optional (default=12)
        Chunk size for loading data with xarray to optimize memory usage.


    Returns:
    --------
    feedbacks : dict
        A dictionary containingthe radiative anomalies due to water vapor concentration for clear-sky 
        (`clr`) and all-sky (`cld`) conditions. Keys of the dictionary:
        - `('clr', 'planck-surf')`: Clear-sky surface Planck feedback.
        - `('cld', 'planck-surf')`: All-sky surface Planck feedback. """
   
    var = ds.wv
 
    feedbacks=dict()

    if use_climatology:
        if ref_clim is None:
            raise ValueError("ref_clim must be provide if use_climatology is True")
        var_clim = ref_clim.wv
        var_clim = q_to_ppmv(var_clim)
    else:
        var_clim = var.groupby('time.month').mean()
        var_clim = q_to_ppmv(var_clim)


    for tip in ['clr', 'cld']:
        kernel = allkers[(tip, 'wv')]

    # Condizione Lat, Lon
        if lat_range is not None and lon_range is not None:

            # Seleziona latitudine e longitudine per var
                var = ctl.sel_area_xr(var,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))
                var_clim = ctl.sel_area_xr(var_clim,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))
                kernel = ctl.sel_area_xr(kernel,(lon_range[0],lon_range[1],lat_range[0],lat_range[1]))

        anoms =  var.groupby('time.month') - var_clim
        anoms_monthly = anoms.groupby('time.month')

        # Prodotto 

        if ker=='STE':
            # WV in ppmv
            dRt = anoms_monthly*kernel 
            dRt = dRt.sum(dim="level")
            dRt_glob = ctl.global_mean(dRt)

        if ker=='HUANG':
            # CAMBIA TUTTO
            dRt = anoms_monthly*kernel 
            dRt_glob = ctl.global_mean(dRt)

        planck= dRt_glob.compute()
        feedbacks[(tip, 'wv')] = planck

    return(feedbacks)


# FUNCTION FOR WV ANOMALIES
# From Mass mixing Ratio (kg/kg to ppmv)
def q_to_ppmv(q_inp):
    Ma = 28.97  # Molecular weight of dry air
    Mw = 18.02  # Molecular weight of water vapor
    vw_ppmv = q_inp / (1 - q_inp) * (Ma / Mw) * 10**6
    return vw_ppmv

############# FINE NEW FUNCTIONS FOR SPECTRAL KERNELS ########################


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
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", False)
    save_pattern = config.get("save_pattern", False)
    use_ds_climatology = bool(use_ds_climatology)
    use_climatology = bool(use_climatology)
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
    radiation = Rad_anomaly_planck_surf(ds, ref_clim_data, ker, allkers, cart_out, use_climatology, time_range_exp, use_ds_climatology, save_pattern)

    return (radiation)

def Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, use_climatology=True, time_range=None, use_ds_climatology=True, save_pattern=False):
    """
    Computes the Planck surface radiation anomaly using climate model data and radiative kernels.

    Parameters:
    - ds (xarray.Dataset): Input dataset containing surface temperature (ts) and near-surface air temperature (tas).
    - piok (xarray.Dataset): Climatological or multi-year mean surface temperature reference.
    - piok_tas (xarray.DataArray): Climatological or multi-year mean near-surface air temperature reference.
    - ker (str): Name of the kernel set used for radiative calculations.
    - allkers (dict): Dictionary containing radiative kernels for different conditions (e.g., 'clr', 'cld').
    - cart_out (str): Output directory where computed results will be saved.
    - use_climatology (bool, optional): Whether to use climatological anomalies (default is True).
    - time_range (tuple of str, optional): Time range for selecting data (format: ('YYYY-MM-DD', 'YYYY-MM-DD')).
    - ds_clim (bool, optional): Whether to use climatology anomaly calculations (default is False).

    Returns:
    - dict: A dictionary containing computed Planck surface radiation anomalies for clear-sky ('clr') and cloud ('cld') conditions.
    
    Outputs Saved to `cart_out`:
    - `gtas{suffix}.nc`: Global temperature anomaly series, grouped by year.
    - `dRt_planck-surf_global_{tip}{suffix}.nc`: Global surface Planck feedback for clear (`clr`) and 
      all (`cld`) sky conditions.

      Here, `{suffix}` = `"_climatology-{ker}kernels"` or `"_21yearmean-{ker}kernels"`, based on the 
      `use_climatology` flag and kernel type (`ker`).   
    """
    radiation = dict()
    k = allkers[('cld', 't')]
    cos = "_climatology" if use_climatology else "_21yearmean"

    if time_range is not None:
        var = ds['ts'].sel(time=slice(time_range['start'], time_range['end']))
        var=ctl.regrid_dataset(var, k.lat, k.lon)
    else:
        var=ds['ts']
        var = ctl.regrid_dataset(ds['ts'], k.lat, k.lon)  

    if use_ds_climatology == False:
        if use_climatology == False:
            check_data(var, piok['ts'])
            piok = piok['ts'].drop('time')
            piok['time'] = var['time']
            piok = piok.chunk(var.chunks)
            anoms = var - piok
        else:
            anoms = var.groupby('time.month') - piok['ts']
    else:
        if use_climatology == False:
            check_data(var, piok['ts'])
            piok = piok['ts'].drop('time')
            piok['time'] = var['time']
            piok = piok.chunk(var.chunks)
            anoms = var.groupby('time.month').mean() - piok.groupby('time.month').mean()
        else:
            anoms = var.groupby('time.month').mean() - piok['ts']
    
 
    for tip in ['clr', 'cld']:
        print(f"Processing {tip}")  
        try:
            kernel = allkers[(tip, 'ts')]
            print("Kernel loaded successfully")  
        except Exception as e:
            print(f"Error loading kernel for {tip}: {e}")  
            continue  
        if use_ds_climatology == False:
           dRt = (anoms.groupby('time.month') * kernel).groupby('time.year').mean('time')
        else:
           dRt = (anoms* kernel).mean('month')

        #Save full dRt pattern before global averaging
        if save_pattern: 
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} surface Planck dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_planck-surf_pattern_" + tip + cos + "-" + ker + "kernels.nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt)
        planck = dRt_glob.compute()
        radiation[(tip, 'planck-surf')] = planck
        planck.to_netcdf(cart_out + "dRt_planck-surf_global_" + tip + cos + "-" + ker + "kernels.nc", format="NETCDF4")
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
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_atm_mask = config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_ds_climatology = bool(use_ds_climatology)
    use_climatology = bool(use_climatology)
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
    radiation = Rad_anomaly_planck_atm_lr(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_exp, config, use_ds_climatology, use_atm_mask, save_pattern)
    
    return (radiation)

def Rad_anomaly_planck_atm_lr(ds, piok, ker, allkers, cart_out, surf_pressure=None, use_climatology=True, time_range=None, config_file=None, use_ds_climatology=True, use_atm_mask=True, save_pattern=False):

    """
    Computes the Planck atmospheric and lapse-rate radiation anomalies using climate model data and radiative kernels.

    Parameters:
    - ds (xarray.Dataset): Input dataset containing atmospheric temperature (ta) and surface temperature (ts).
    - piok (xarray.Dataset): Input dataset containing Climatological or multi-year mean atmospheric temperature reference (ta) and surface temperature reference (ts).
    - cart_out (str): Output directory where computed results will be saved.
    - ker (str): Name of the kernel set used for radiative calculations (e.g., 'ERA5', 'HUANG').
    - allkers (dict): Dictionary containing radiative kernels for different conditions (e.g., 'clr', 'cld').
    - surf_pressure (xr.Dataset):An xarray dataset containing surface pressure (`ps`) values.
    - time_range (tuple of str, optional): Time range for selecting data (format: ('YYYY-MM-DD', 'YYYY-MM-DD')).
    - use_climatology (bool, optional): Whether to use climatological anomalies (default is True).

    Returns:
    - dict: A dictionary containing computed Planck atmospheric and lapse-rate radiation anomalies for clear-sky ('clr') and cloud ('cld') conditions.
    
    Outputs Saved to `cart_out`:
    ----------------------------
    - `dRt_planck-atmo_global_{tip}{suffix}.nc`: Global atmospheric Planck feedback for clear (`clr`) and all (`cld`) sky conditions.
    - `dRt_lapse-rate_global_{tip}{suffix}.nc`: Global lapse-rate feedback for clear (`clr`) and all (`cld`) sky conditions.

      Here, `{suffix}` = `"_climatology-{ker}kernels"` or `"_21yearmean-{ker}kernels"`, based on the `use_climatology` flag and kernel type (`ker`).
  
    
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
   
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

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

    if use_ds_climatology == False:
        if use_climatology==False:
            check_data(var, piok['ta'])
            piok_ta=piok['ta'].drop('time')
            piok_ta['time'] = var['time']
            piok_ts=piok['ts'].drop('time')
            piok_ts['time'] = var['time']
            anoms_ok = var - piok_ta
            ts_anom = var_ts - piok_ts
        else:
            anoms_ok=var.groupby('time.month') - piok['ta']
            ts_anom=var_ts.groupby('time.month') - piok['ts']
        if use_atm_mask==True:
            mask=mask_atm(var)
    else: 
        if use_climatology==False:
            check_data(ds['ta'], piok['ta'])
            piok_ta=piok['ta'].drop('time')
            piok_ta['time'] = var['time']
            piok_ts=piok['ts'].drop('time')
            piok_ts['time'] = var['time']
            anoms_ok = var.groupby('time.month').mean() - piok_ta.groupby('time.month').mean()
            ts_anom = var_ts.mean('time') - piok_ts.mean('time')
        else:
            anoms_ok=var.groupby('time.month').mean() - piok['ta']
            ts_anom = var_ts.groupby('time.month') - piok['ts']
            ts_anom = ts_anom.compute()
        mask=mask_atm(var)
        mask = mask.groupby('time.month').mean()

    if use_atm_mask==True:
        anoms_ok = (anoms_ok*mask).interp(plev = cose) 
    else:
        anoms_ok = anoms_ok.interp(plev = cose)

    if ker=='HUANG':
        if use_ds_climatology == False:
                anoms_lr = (anoms_ok - ts_anom)  
                anoms_unif = (anoms_ok - anoms_lr)
        else: 
                anoms_lr = (anoms_ok - ts_anom.mean('time'))
                anoms_unif = (anoms_ok - anoms_lr)
    if ker=='ERA5':
        if use_ds_climatology == False:
                anoms_lr = (anoms_ok - ts_anom)  
                anoms_unif = (anoms_ok - anoms_lr)
        else: 
                anoms_lr = (anoms_ok - ts_anom.groupby('time.month').mean())
                anoms_unif = (anoms_ok - anoms_lr)

    for tip in ['clr', 'cld']:
        print(f"Processing {tip}")  
        try:
            kernel = allkers[(tip, 't')]
            print("Kernel loaded successfully")  
        except Exception as e:
            print(f"Error loading kernel for {tip}: {e}")  
            continue  

        if ker=='HUANG':
            if use_ds_climatology == False:
                dRt_unif = (anoms_unif.groupby('time.month')*kernel*wid_mask/100).sum('player').groupby('time.year').mean('time')  
                dRt_lr = (anoms_lr.groupby('time.month')*kernel*wid_mask/100).sum('player').groupby('time.year').mean('time')   
            else:
                dRt_unif = (anoms_unif*kernel*wid_mask/100.).sum('player').mean('month')  
                dRt_lr = (anoms_lr*kernel*wid_mask/100.).sum('player').mean('month')  

        if ker=='ERA5':
            if use_ds_climatology == False:
                dRt_unif =(anoms_unif.groupby('time.month')*(kernel*vlevs.dp/100)).sum('player').groupby('time.year').mean('time')  
                dRt_lr = (anoms_lr.groupby('time.month')*(kernel*vlevs.dp/100)).sum('player').groupby('time.year').mean('time') 
            else:
                dRt_unif = ((anoms_unif)*(kernel*vlevs.dp/100)).sum('player').mean('month')  
                dRt_lr = ((anoms_lr)*(kernel*vlevs.dp/100)).sum('player').mean('month') 

        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt_unif.name = "dRt_atmo"
            dRt_unif.attrs["description"] = f"{tip} atmosperic Planck dRt pattern"
            dRt_unif.to_netcdf(cart_out + "dRt_planck-atmo_pattern_" + tip + cos + "-" + ker + "kernels.nc", format="NETCDF4")

            dRt_lr.name = "dRt_lr"
            dRt_lr.attrs["description"] = f"{tip} lapse-rate dRt pattern"
            dRt_lr.to_netcdf(cart_out + "dRt_lapse-rate_pattern_" + tip + cos + "-" + ker + "kernels.nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        radiation[(tip,'planck-atmo')]=feedbacks_atmo
        radiation[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  +cos+"-"+ker+"kernels.nc", format="NETCDF4")
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
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    save_pattern = config.get("save_pattern", False)
    use_ds_climatology = bool(use_ds_climatology)
    use_climatology = bool(use_climatology)
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
    radiation = Rad_anomaly_albedo(ds, ref_clim_data, ker, allkers, cart_out, use_climatology, time_range_exp, use_ds_climatology, save_pattern)

    return (radiation)

def Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, use_climatology=True, time_range=None, use_ds_climatology=True, save_pattern=False):
    """
    Computes the albedo radiation anomaly using climate model data and radiative kernels.

    Parameters:
    - ds (xarray.Dataset): Input dataset containing surface upward (rsus) and downward (rsds) shortwave radiation.
    - piok (xarray.Dataset): Climatological or multi-year mean albedo reference.
    - ker (str): Name of the kernel set used for radiative calculations.
    - allkers (dict): Dictionary containing radiative kernels for different conditions (e.g., 'clr', 'cld').
    - cart_out (str): Output directory where computed results will be saved.
    - use_climatology (bool, optional): Whether to use climatological anomalies (default is True).
    - time_range (tuple of str, optional): Time range for selecting data (format: ('YYYY-MM-DD', 'YYYY-MM-DD')).

    Returns:
    - dict: A dictionary containing computed albedo radiation anomalies for clear-sky ('clr') and cloud ('cld') conditions.
    
    Outputs Saved to `cart_out`:
    ----------------------------
    - `dRt_albedo_global_{tip}{suffix}.nc`: Global annual mean albedo feedback for clear (`clr`) and all (`cld`) sky conditions.
    
      Here, `{suffix}` = `"_climatology-{ker}kernels"` or `"_21yearmean-{ker}kernels"`, based on the `use_climatology` flag and kernel type (`ker`).    
    """
    
    radiation=dict()
    k=allkers[('cld', 't')]

    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

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
        
    if use_ds_climatology==False:
        if use_climatology==False:
            check_data(var, piok)
            piok=piok.drop('time')
            piok['time'] = var['time']
            anoms =  var - piok
        else:
            anoms =  var.groupby('time.month') - piok
    else:
        if use_climatology==False:
            check_data(var, piok)
            piok=piok.drop('time')
            piok['time'] = var['time']
            anoms =  var.groupby('time.month').mean() - piok.groupby('time.month').mean()
        else:
            anoms =  var.groupby('time.month').mean() - piok

    for tip in [ 'clr','cld']:
        kernel = allkers[(tip, 'alb')]
        if use_ds_climatology==False:
            dRt = (anoms.groupby('time.month')* kernel).groupby('time.year').mean('time') 
        else: 
            dRt = (anoms* kernel).mean('month') 

        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} albedo dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_albedo_pattern_" + tip + cos + "-" + ker + "kernels.nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        radiation[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
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
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_atm_mask = config.get("use_atm_mask",True)
    save_pattern = config.get("save_pattern", False)    
    use_ds_climatology = bool(use_ds_climatology)
    use_climatology = bool(use_climatology) 
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
    radiation = Rad_anomaly_wv(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_exp, config, use_ds_climatology, use_atm_mask, save_pattern)
    
    return (radiation)

def Rad_anomaly_wv(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology=True, time_range=None, config_file=None, use_ds_climatology=True, use_atm_mask=True, save_pattern=False):
    
    """
    Computes the water vapor radiation anomaly using climate model data and radiative kernels.

    Parameters:
    - ds (xarray.Dataset): Input dataset containing specific humidity (hus) and atmospheric temperature (ta).
    - piok (xarray.Dataset): Input dataset containing Climatological or multi-year mean reference for specific humidity (hus) and atmospheric temperature (ta).
    - cart_out (str): Output directory where computed results will be saved.
    - ker (str): Name of the kernel set used for radiative calculations (e.g., 'ERA5', 'HUANG').
    - allkers (dict): Dictionary containing radiative kernels for different conditions (e.g., 'clr', 'cld').
    - surf_pressure (xr.Dataset):An xarray dataset containing surface pressure (`ps`) values.
    - time_range (tuple of str, optional): Time range for selecting data (format: ('YYYY-MM-DD', 'YYYY-MM-DD')).
    - use_climatology (bool, optional): Whether to use climatological anomalies (default is True).

    Returns:
    - dict: A dictionary containing computed water vapor radiation anomalies for clear-sky ('clr') and cloud ('cld') conditions.

    Additional Outputs:
    -------------------
    The function saves the following files to the `cart_out` directory:
    - **`dRt_water-vapor_global_clr.nc`**: Clear-sky water vapor feedback as a NetCDF file.
    - **`dRt_water-vapor_global_cld.nc`**: All-sky water vapor feedback as a NetCDF file.

    Depending on the value of `use_climatology`, the function saves different NetCDF files to the `cart_out` directory:
    If `use_climatology=True` it adds "_climatology", elsewhere it adds "_21yearmean"
    """
    if ker == 'HUANG' and surf_pressure is None:
        raise ValueError("Error: The 'surf_pressure' parameter cannot be None when 'ker' is 'HUANG'.")

    k=allkers[('cld', 't')]
    cose=pickle.load(open(cart_out + 'cose_'+ker+'.p', 'rb'))
    radiation=dict()
    if ker=='ERA5':
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    var=ds['hus']
    var_ta=ds['ta']
    if time_range is not None:
        var = ds['hus'].sel(time=slice(time_range['start'], time_range['end'])) 
        var_ta = ds['ta'].sel(time=slice(time_range['start'], time_range['end']))
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    var_ta = ctl.regrid_dataset(var_ta, k.lat, k.lon)
    if use_atm_mask==True:
        mask=mask_atm(var_ta)

    Rv = 487.5 # gas constant of water vapor
    Lv = 2.5e+06 # latent heat of water vapor

    if use_climatology==False:
        check_data(var_ta, piok['ta'])
        piok_hus=piok['hus'].drop('time')
        piok_hus['time'] = var['time']
        piok_ta=piok['ta'].drop('time')
        piok_ta['time'] = var['time']
    else:
        piok_ta=piok['ta']
        piok_hus=piok['hus']

    ta_abs_pi = piok_ta.interp(plev = cose)
    if use_atm_mask==True:
        var_int = (var*mask).interp(plev = cose)
    else:
        var_int = var.interp(plev = cose)
    piok_int = piok_hus.interp(plev = cose)

    if ker=='HUANG':
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
        wid_mask=mask_pres(surf_pressure, cart_out, allkers, config_file)
        
        if use_ds_climatology==False:
            if use_climatology==True:
                anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int.groupby('time.month'), piok_int , dask = 'allowed')
                coso3= anoms_ok3.groupby('time.month') *dlnws(ta_abs_pi)
            else:
                anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int, piok_int , dask = 'allowed')
                coso3= anoms_ok3*dlnws(ta_abs_pi)
        else:
            if use_climatology==True:
                anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int.groupby('time.month').mean(), piok_int, dask = 'allowed')
                coso3= anoms_ok3 *dlnws(ta_abs_pi)
            else:
                anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int.groupby('time.month').mean(), piok_int.groupby('time.month').mean() , dask = 'allowed')
                coso3= anoms_ok3*dlnws(ta_abs_pi)
    
    if ker=='ERA5': 
        if use_ds_climatology==False:
            if use_climatology==False:
                anoms= var_int-piok_int
                coso = (anoms/piok_int) * (ta_abs_pi**2) * Rv/Lv
            else:
                anoms= var_int.groupby('time.month')-piok_int
                coso = (anoms.groupby('time.month')/piok_int).groupby('time.month') * (ta_abs_pi**2) * Rv/Lv #dlnws(ta_abs_pi) you can also use the functio
        else: 
            if use_climatology==False:
                anoms= var_int.groupby('time.month').mean()-piok_int.groupby('time.month').mean()
                coso = (anoms/piok_int) * (ta_abs_pi**2) * Rv/Lv
            else:
                anoms= var_int.groupby('time.month').mean()-piok_int
                coso = (anoms/piok_int) * (ta_abs_pi**2) * Rv/Lv
    
    for tip in ['clr','cld']:
        kernel_lw = allkers[(tip, 'wv_lw')]
        kernel_sw = allkers[(tip, 'wv_sw')]
        kernel = kernel_lw + kernel_sw
        
        if ker=='HUANG':
            if use_ds_climatology==False:
                dRt = (coso3.groupby('time.month')* kernel* wid_mask/100).sum('player').groupby('time.year').mean('time')
            else:
                dRt = (coso3* kernel* wid_mask/100).sum('player').mean('month')

        if ker=='ERA5':
            if use_ds_climatology==False:
                dRt = (coso.groupby('time.month')*( kernel* vlevs.dp / 100) ).sum('player').groupby('time.year').mean('time')
            else:
                dRt = (coso*( kernel* vlevs.dp / 100)).sum('player').mean('month')


        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} water vapor dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_water-vapor_pattern_" + tip + cos + "-" + ker + "kernels.nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt)
        wv= dRt_glob.compute()
        radiation[(tip, 'water-vapor')]=wv
        wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" +tip+cos +"-"+ker+"kernels.nc", format="NETCDF4")
        wv.close()
        
    return radiation

#ALL RAD_ANOM COMPUTATION
def calc_anoms_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
   
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
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_atm_mask = config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_climatology = bool(use_climatology)
    use_ds_climatology = bool(use_ds_climatology)
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
    
    anom_ps, anom_pal, anom_a, anom_wv = calc_anoms(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_exp, use_ds_climatology, config_file, use_atm_mask, save_pattern)

    return (anom_ps, anom_pal, anom_a, anom_wv)

def calc_anoms(ds, piok_rad, ker, allkers, cart_out, surf_pressure, use_climatology=True, time_range=None, use_ds_climatology=False, config_file =None, use_atm_mask=True, save_pattern=False):
    """
    
    """
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_ps = Rad_anomaly_planck_surf(ds, piok_rad, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology, save_pattern)
    else:
        anom_ps = xr.open_dataset(path)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_pal = Rad_anomaly_planck_atm_lr(ds, piok_rad, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology, use_atm_mask, save_pattern)
    else:
        anom_pal = xr.open_dataset(path)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_a = Rad_anomaly_albedo(ds, piok_rad, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology, save_pattern)
    else:
        anom_a = xr.open_dataset(path)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_wv = Rad_anomaly_wv(ds, piok_rad, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology, use_atm_mask, save_pattern)
    else:
        anom_wv = xr.open_dataset(path)  

    return anom_ps, anom_pal, anom_a, anom_wv 

##FEEDBACK COMPUTATION
def calc_fb_wrapper(config_file: str, ker, variable_mapping_file: str):
    """
    Wrapper function to compute radiative and cloud feedbacks based on the provided configuration file and kernel type.
    
    This function loads the necessary data and kernels from the configuration file, processes the datasets, and 
    computes the feedback coefficients for both radiative anomalies and cloud feedbacks. It takes into account time 
    ranges for climatology and experimental data, as well as surface pressure if required by the kernel type.
    
    Parameters:
    - config_file (str): Path to the configuration YAML file containing the model settings and file paths.
    - ker (str): Kernel type (e.g., 'HUANG') that determines if surface pressure is required for the analysis.
    -  variable_mapping_file : str
        Path to the YAML file with standardization rules for variables.
    
    Returns:
    - fb_coef (dict): Dictionary containing feedback coefficients for radiative anomalies.
    - fb_cloud (xarray.DataArray): Cloud feedback data array.
    - fb_cloud_err (xarray.DataArray): Cloud feedback error data array.
   
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
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_atm_mask=config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_climatology = bool(use_climatology)
    use_ds_climatology = bool(use_ds_climatology)
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
            raise ValueError("HUANG kernels require surface pressure data, but none was provided.")
        
    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config_file, allvars, ker, variable_mapping_file, allkers=allkers) 
    
    fb_coef, fb_cloud, fb_cloud_err, fb_pattern = calc_fb(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_exp, use_ds_climatology, config_file, use_atm_mask, save_pattern)
    
    return fb_coef, fb_cloud, fb_cloud_err, fb_pattern

   

def calc_fb(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology=True, time_range=None, use_ds_climatology=True, config_file =None, use_atm_mask=True, save_pattern=False):
    """
    Compute the radiative feedback and cloud feedback based on the provided datasets and kernels.
    
    This function calculates the radiative feedbacks by processing the Planck surface, Planck atmospheric, 
    albedo, and water vapor anomalies. It also computes cloud feedbacks by applying linear regression to global 
    mean temperature anomalies and feedbacks. The function handles both climatology-based and 21-year mean data, 
    depending on the `use_climatology` flag. It also considers the specified time range for the analysis.
    
    Parameters:
    - ds (xarray.Dataset): The dataset containing climate model outputs.
    - piok_rad (xarray.Dataset): The reference climatology dataset for radiative anomalies.
    - piok_cloud (xarray.Dataset): The reference climatology dataset for cloud feedbacks.
    - ker (str): Kernel type (e.g., 'HUANG') that determines specific feedback calculations.
    - allkers (dict): Dictionary of kernel data arrays for different feedback types.
    - cart_out (str): Path to the output directory where the feedback results are stored.
    - surf_pressure (xarray.Dataset or None): The surface pressure dataset (if required by the kernel type).
    - use_climatology (bool, optional): Whether to use climatology data (default is True).
    - time_range (tuple, optional): The time range to be used for the analysis (default is None).
    
    Returns:
    - fb_coef (dict): Dictionary containing feedback coefficients for different feedback types.
    - fb_cloud (xarray.DataArray): Cloud feedback data array.
    - fb_cloud_err (xarray.DataArray): Cloud feedback error data array.
    """

    if 'tas' not in piok:
        raise ValueError("Reference climatology for 'tas' is missing in piok. Ensure 'tas' is included in 'allvars' when calling ref_clim.")
    
    if use_climatology==True: 
        cos="_climatology"
        print(cos)
    else:
        cos="_21yearmean"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology, save_pattern)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology, save_pattern)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_cld"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_atm_lr(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology, use_atm_mask, save_pattern)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_wv(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology,use_atm_mask, save_pattern)    
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

    if use_climatology == False:
        piok_tas=piok['tas'].drop('time')
        piok_tas['time'] = var_tas['time']
        piok_tas = piok_tas.chunk(var_tas.chunks)
        anoms_tas = var_tas - piok_tas
    else:
        anoms_tas = var_tas.groupby('time.month') - piok['tas']


    gtas = ctl.global_mean(anoms_tas).groupby('time.year').mean('time') 
    gtas = gtas.groupby((gtas.year-1) // 10 * 10).mean()


    print('feedback calculation...')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ cos+"-"+ker+"kernels.nc",  use_cftime=True)
            feedback=feedbacks.groupby((feedbacks.year-1) // 10 * 10).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res

            if save_pattern:
                print(f"Computing spatial feedback pattern for {tip}-{fbn}...")
                # Open the dRt pattern
                feedbacks_pattern = xr.open_dataarray(cart_out+"dRt_"+fbn+"_pattern_"+tip+cos+"-"+ker+"kernels.nc", use_cftime=True)
                feedbacks_pattern_dec = feedbacks_pattern.groupby((feedbacks_pattern.year - 1) // 10 * 10).mean('year')
                feedbacks_pattern_dec = feedbacks_pattern_dec.chunk({'year': -1})
                gtas = gtas.chunk({'year': -1})
                # Perform regression at each grid point
                slope, stderr = regress_pattern_vectorized(feedbacks_pattern_dec, gtas)
                fb_pattern[(tip, fbn)] = (slope, stderr)
    
    #cloud
    print('cloud feedback calculation...')
    fb_cloud, fb_cloud_err = feedback_cloud(ds, piok, fb_coef, gtas, time_range)
    
    return fb_coef, fb_cloud, fb_cloud_err, (fb_pattern if save_pattern else None)
    

#CLOUD FEEDBACK shell 2008
def feedback_cloud_wrapper(config_file: str, ker, variable_mapping_file: str):

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
    allvars = 'tas'
    allvars1= 'rlutcs rsutcs rlut rsut'.split()
    print("Read parameters from configuration file...")
    
    dataset_type = config.get('dataset_type', None)
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_atm_mask = config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    use_climatology = bool(use_climatology)
    use_ds_climatology = bool(use_ds_climatology)
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
    allvars_combined = allvars + allvars1
    ref_clim_data = ref_clim(config_file, allvars_combined, ker, variable_mapping_file, allkers=allkers) 

    fb_coef = calc_fb(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_exp, use_ds_climatology, config_file, use_atm_mask, save_pattern)

    k=allkers[('cld', 't')]
    for nom in allvars1:
        ds[nom] = ctl.regrid_dataset(ds[nom], k.lat, k.lon) 

    var_tas= ctl.regrid_dataset(ds['tas'], k.lat, k.lon) 

    if use_climatology == False:
        piok_tas=ref_clim_data.drop('time')
        piok_tas['time'] = var_tas['time']
        piok_tas = piok_tas.chunk(var_tas.chunks)
        anoms_tas = var_tas - piok_tas
    else:
        anoms_tas = var_tas.groupby('time.month') - ref_clim_data['tas']
        
    gtas = ctl.global_mean(anoms_tas).groupby('time.year').mean('time') 
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()

    fb_cloud, fb_cloud_err = feedback_cloud(ds, ref_clim_data, fb_coef, gtas, time_range_exp)

    return fb_cloud, fb_cloud_err

def feedback_cloud(ds, piok, fb_coef, surf_anomaly, time_range=None):
   #questo va testato perchè non sono sicura che funzionino le cose con pimean (calcolato con climatology ha il groupby.month di cui qui non si tiene conto)
    """
    Computes cloud radiative feedback anomalies using climate model data.

    Parameters:
    - ds (xarray.Dataset): Input dataset containing outgoing longwave radiation (rlut), reflected shortwave radiation (rsut), clear-sky outgoing longwave (rlutcs) and shortwave (rsutcs) radiation.
    - pimean (dict): Dictionary of pre-industrial mean values for radiative fluxes.
    - fb_coef (dict): Dictionary containing radiative feedback coefficients for different feedback mechanisms.
    - surf_anomaly (xarray.DataArray): Surface temperature anomaly data used for regression, which should be pre-processed as follows:  
      `gtas = ctl.global_mean(tas_anomaly).groupby('time.year').mean('time')`  
      `gtas = gtas.groupby((gtas.year-1) // 10 * 10).mean()`
    - time_range (tuple of str, optional): Time range for selecting data (format: ('YYYY-MM-DD', 'YYYY-MM-DD')).

    Returns:
    - tuple: 
        - fb_cloud (float): Cloud radiative feedback strength.
        - fb_cloud_err (float): Estimated error in the cloud radiative feedback calculation.
    """
    if not (ds['rlut'].dims == ds['rsutcs'].dims):
        raise ValueError("Error: The spatial grids ('lon' and 'lat') datasets must match.")
    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    
    if time_range is not None:
        nomi='rlut rsut rlutcs rsutcs'.split()
        for nom in nomi:
            ds[nom] = ds[nom].sel(time=slice(time_range['start'], time_range['end']))


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
    N_glob = ctl.global_mean(N).compute()
    N0_glob = ctl.global_mean(N0).compute()

    crf_glob= crf_glob.groupby((crf_glob.year-1) // 10 * 10).mean(dim='year')
    N_glob=N_glob.groupby((N_glob.year-1) // 10 * 10).mean(dim='year')
    N0_glob=N0_glob.groupby((N0_glob.year-1) // 10 * 10).mean(dim='year')

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


###### Plotting ######
