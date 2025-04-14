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

###### LOAD KERNELS
def load_spectral_kernel(config_file):

    ## per ste

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
standard_names = {
    "rsus": "surface upwelling shortwave radiation",
    "rsds": "surface downwelling shortwave radiation",
    "time": "time dimension",
    "lat": "latitude",
    "lon": "longitude",
    "plev": "pressure levels",
    "ps": "surface pressure",
    "ts": "surface temperature",
    "tas": "near-surface air temperature",
    "ta": "atmospheric temperature",
    "hus": "specific humidity",
    "rlut":"outgoing longwave radiation",
    "rsut":"reflected shortwave radiation",
    "rlutcs":"clear-sky outgoing longwave radiation",
    "rsutcs":"clear-sky outgoing shortwave radiation"
}

def read_data(config_file: str, standard_names):
    """
    Reads the configuration from the YAML file, opens the NetCDF file specified in the config,
    and standardizes the variable names in the dataset.
    
    Parameters:
    -----------
    config_file : str
        The path to the YAML configuration file.
    
    standard_names : dict
        A dictionary containing the standard names for variables.
    
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

    if not file_path1:
        raise ValueError("Error: The 'experiment_dataset' path is not specified in the configuration file")

    
    ds_list = [xr.open_mfdataset(file_path1,  use_cftime=True, chunks={'time': time_chunk})]
 
    if file_path2 and file_path2.strip():
        ds_list.append(xr.open_mfdataset(file_path2, combine='by_coords', use_cftime=True, chunks={'time': time_chunk}))

    if file_pathpl and file_pathpl.strip():
        ds_list.append(xr.open_mfdataset(file_pathpl, combine='by_coords', use_cftime=True,  chunks={'time': time_chunk}))

    # Merge dataset
    ds = xr.merge(ds_list, compat="override")

    required_vars = list(standard_names.keys())
    missing_vars = [var for var in required_vars if var not in ds.variables]

    if missing_vars:
        print(f"Warning: The following required variables are not present in the dataset: {missing_vars}")
        print("Check that the files in config have the necessary variables.")

    ds = standardize_names(ds, standard_names)

    return ds

def standardize_names(ds, standard_names):
    """
    Standardizes variable names in a dataset by checking against a dictionary of standard names.
    
    If a variable is not recognized, the function suggests a possible match based on similarity,
    and asks the user if they want to rename it.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the variables to be checked.
    
    standard_names : dict
        A dictionary with standard variable names.
    
    Returns:
    --------
    ds : xarray.Dataset
        The dataset with standardized variable names.
    """
    print("Standard variable names:", list(standard_names.keys()))

    renamed_vars = {}  
    unrecognized_vars = []  

    for var_name in list(ds.variables):  
        # Skip already standard variables
        if var_name in standard_names:
            continue 
        
        # Suggest a renaming based on close matches
        close_match = get_close_matches(var_name, standard_names.keys(), n=1, cutoff=0.9)
        
        if close_match:
            suggested_name = close_match[0]
            response = input(f"Variable '{var_name}' not recognized. Do you want to rename it to '{suggested_name}'? (y/n): ")
            if response.lower() == 'y':
                renamed_vars[var_name] = suggested_name
            else:
                unrecognized_vars.append(var_name)
        else:
            # If no close match is found, let the user decide
            response = input(f"Variable '{var_name}' not recognized and no close match found. Do you want to rename it manually? (y/n): ")
            if response.lower() == 'y':
                new_name = input(f"Please enter the new name for the variable '{var_name}': ")
                renamed_vars[var_name] = new_name
            else:
                unrecognized_vars.append(var_name)
    
    # Rename variables in the dataset
    if renamed_vars:
        ds = ds.rename(renamed_vars)
        print(f"Variables automatically renamed: {renamed_vars}")

    # Print out any unrecognized variables
    if unrecognized_vars:
        print(f"Not recognized variables that were not renamed: {unrecognized_vars}")

    print("Variables in dataset after renaming:", list(ds.variables))

    # Check for missing variables and suggest calculating them
    all_vars = set(ds.variables) | set(unrecognized_vars)
    # New conditions for calculating missing variables
    if 'rsutcs' not in ds.variables and 'rsdt' in all_vars and 'rsntcs' in all_vars:
        response = input("Missing 'rsutcs'. You can calculate it as 'rsutcs = rsdt - rsntcs'. Do you want to do that? (y/n): ")
        if response.lower() == 'y':
            ds['rsutcs'] = ds['rsdt'] - ds['rsntcs']
    
    if 'rlutcs' not in ds.variables and 'rlntcs' in ds.variables:
        response = input("Missing 'rlutcs'. You can calculate it as 'rlutcs = -rlntcs'. Do you want to do that? (y/n): ")
        if response.lower() == 'y':
            ds['rlutcs'] = -ds['rlntcs']

    if 'rsus' not in ds.variables and 'ssr' in all_vars and 'rsds' in all_vars:
        response = input("Missing 'rsus'. You can calculate it as 'rsus = rsds - ssr'. Do you want to do that? (y/n): ")
        if response.lower() == 'y':
            ds['rsus'] = ds['rsds'] - ds['ssr']
    
    if 'rsutcs' not in ds.variables and 'rsut' in all_vars and 'tsrc' in all_vars and 'tsr' in all_vars:
        response = input("Missing 'rsutcs'. You can calculate it as 'rsutcs = rsut x (tsrc/tsr)'. Do you want to do that? (y/n): ")
        if response.lower() == 'y':
            ds['rsutcs'] = ds['rsut'] * (ds['tsrc']/ds['tsr'])
    
    # Handle tas/ts equivalence
    if 'ts' not in ds.variables and 'tas' in all_vars:
        response = input("Missing 'ts'. You can use 'tas' as 'ts'. Do you want to do that? (y/n): ")
        if response.lower() == 'y':
            ds['ts'] = ds['tas']
    
    return ds

def check_data(ds, piok):
    if len(ds["time"]) != len(piok["time"]):
        raise ValueError("Error: The 'time' columns in 'ds' and 'piok' must have the same length. To fix use variable 'time_range' of the function")
    return

######################################################################################
#### Aux functions
def ref_clim(config_file: str, allvars, ker, standard_names, allkers=None):
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
    standard_names : dict
        A dictionary containing the standard names for variables.

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

    filin_pi = config['file_paths'].get('reference_dataset', None)
    filin_pi_pl = config['file_paths'].get('reference_dataset_pl', None)
    time_chunk = config.get('time_chunk', None)
    use_climatology = config.get("use_climatology", True)   
    use_climatology = bool(use_climatology)
    
    if not filin_pi:
        raise ValueError("Error: the 'reference_dataset' path is not specified in the configuration file.")

    ds_list = [xr.open_mfdataset(filin_pi, use_cftime=True, chunks={'time': time_chunk})]
    
    if filin_pi_pl and filin_pi_pl.strip():
        ds_list.append(xr.open_mfdataset(filin_pi_pl, combine='by_coords', use_cftime=True, chunks={'time': time_chunk}))

    ds_ref = xr.merge(ds_list, compat="override")

    required_vars = list(standard_names.keys())
    missing_vars = [var for var in required_vars if var not in ds_ref.variables]
    if missing_vars:
        print(f"Warning: the following required variables are missing in the dataset: {missing_vars}")
        print("Check that specified files in config have the necessary variables.")
    
    ds_ref = standardize_names(ds_ref, standard_names)

    time_range_clim = config.get("time_range", {})
    time_range_clim = (time_range_clim.get("start"), time_range_clim.get("end"))
    time_range_clim = time_range_clim if all(time_range_clim) else None

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
                var = var.sel(time=slice(time_range[0], time_range[1]))

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
            var = var.sel(time=slice(time_range[0], time_range[1]))

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

#PLANCK SURFACE
def Rad_anomaly_planck_surf_wrapper(config_file: str, ker, standard_names):
    """
    Wrapper for Rad_anomaly_planck_surf function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Planck-Surface anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    standard_names : dict
        standardized variables dictionary.

    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, standard_names)
    print("Variables to consider upload...")
    allvars = 'ts'
    print("Read parameters from configuration file...")

    if allvars not in ds.variables:
        raise ValueError("The ts variable is not in the dataset")

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", False)
    use_ds_climatology = bool(use_ds_climatology)
    use_climatology = bool(use_climatology)

    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})

    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # If `time_range_exp` is defined use it, otherwise use `time_range`
    time_range_to_use = time_range_exp if time_range_exp else time_range_clim

    print(f"Time range used for the simulation analysis: {time_range_to_use}")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config_file, allvars, ker, standard_names, allkers=allkers) 

    print("Planck-Surface radiative anomaly computing...")
    radiation = Rad_anomaly_planck_surf(ds, ref_clim_data, ker, allkers, cart_out, use_climatology, time_range_to_use, use_ds_climatology)

    return (radiation)

def Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, use_climatology=True, time_range=None, use_ds_climatology=True):
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
        dRt_glob = ctl.global_mean(dRt)
        planck = dRt_glob.compute() 
        radiation[(tip, 'planck-surf')] = planck
        planck.to_netcdf(cart_out + "dRt_planck-surf_global_" + tip + cos + "-" + ker + "kernels.nc", format="NETCDF4")
        planck.close()
    return(radiation)

#PLANK-ATMO AND LAPSE RATE WITH VARYING TROPOPAUSE
def Rad_anomaly_planck_atm_lr_wrapper(config_file: str, ker, standard_names):
    """
    Wrapper for Rad_anomaly_planck_atm_lr function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Planck-Atmosphere-LpseRate anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    standard_names : dict
        standardized variables dictionary.

    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, standard_names)
    print("Variables to consider upload...")
    allvars = 'ts ta'.split()
    print("Read parameters from configuration file...")

    for var in allvars:
        if var not in ds.variables:
            raise ValueError(f"The variable '{var}' is not in the dataset")

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_ds_climatology = bool(use_ds_climatology)
    use_climatology = bool(use_climatology)
    use_atm_mask=True

    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})

    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # If `time_range_exp` is defined use it, otherwise use `time_range`
    time_range_to_use = time_range_exp if time_range_exp else time_range_clim

    print(f"Time range used for the simulation analysis: {time_range_to_use}")

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
            surf_pressure = standardize_names(surf_pressure, standard_names)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config, allvars, ker, standard_names, allkers=allkers) 

    print("Planck-Atmosphere-LapseRate radiative anomaly computing...")
    radiation = Rad_anomaly_planck_atm_lr(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_to_use, config, use_ds_climatology, use_atm_mask)

    return (radiation)

def Rad_anomaly_planck_atm_lr(ds, piok, ker, allkers, cart_out, surf_pressure=None, use_climatology=True, time_range=None, config_file=None, use_ds_climatology=True, use_atm_mask=True):

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
                print("anoms_lr:", anoms_lr.sizes) #DA CONTROLLARE!
                print("dims:", anoms_lr.dims)
                print("anoms_lr:", anoms_ok.sizes)
                print("dims:", anoms_ok.dims)

    for tip in ['clr','cld']:
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
            
        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        radiation[(tip,'planck-atmo')]=feedbacks_atmo
        radiation[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        feedbacks_lr.close()
        feedbacks_atmo.close()

    return(radiation)

#ALBEDO
def Rad_anomaly_albedo_wrapper(config_file: str, ker, standard_names):
    """
    Wrapper for Rad_anomaly_albedo function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Albedo anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    standard_names : dict
        standardized variables dictionary.

    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, standard_names)
    print("Variables to consider upload...")
    allvars = 'alb'
    print("Read parameters from configuration file...")

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_ds_climatology = bool(use_ds_climatology)
    use_climatology = bool(use_climatology)

    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})

    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # If `time_range_exp` is defined use it, otherwise use `time_range`
    time_range_to_use = time_range_exp if time_range_exp else time_range_clim

    print(f"Time range used for the simulation analysis: {time_range_to_use}")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config, allvars, ker, standard_names, allkers=allkers) 

    print("Albedo radiative anomaly computing...")
    radiation = Rad_anomaly_albedo(ds, ref_clim_data, ker, allkers, cart_out, use_climatology, time_range_to_use, use_ds_climatology)

    return (radiation)

def Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, use_climatology=True, time_range=None, use_ds_climatology=True):
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
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        radiation[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        alb.close()

    return(radiation)

#W-V COMPUTATION
def Rad_anomaly_wv_wrapper(config_file: str, ker, standard_names):
    """
    Wrapper for Rad_anomaly_wv function, which upload automatically the dataset,
    kernels and climatology are necessary to calculate the radiative Water-Vapour anomaly.

    Parameters:
    -----------
    config_file : str
        configuration file YAML.
    ker : str
        kernels to upload ('ERA5' o 'HUANG').
    standard_names : dict
        standardized variables dictionary.

    Returns:
    --------
    rad_anomaly : dict
        radiative anomalies dictionary for clear sky ('clr') and all ('cld').
    """

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, standard_names)
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
    
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_ds_climatology = bool(use_ds_climatology)
    print('ds_climatology: ')
    print(use_ds_climatology)
    use_climatology = bool(use_climatology) 
    use_atm_mask=True

    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})

    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # If `time_range_exp` is defined use it, otherwise use `time_range`
    time_range_to_use = time_range_exp if time_range_exp else time_range_clim

    print(f"Time range used for the simulation analysis: {time_range_to_use}")

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
            surf_pressure = standardize_names(surf_pressure, standard_names)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology...")
    ref_clim_data = ref_clim(config, allvars, ker, standard_names, allkers=allkers) 

    print("Water-Vapour radiative anomaly computing...")
    radiation = Rad_anomaly_wv(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_to_use, config, use_ds_climatology, use_atm_mask)

    return (radiation)

def Rad_anomaly_wv(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology=True, time_range=None, config_file=None, use_ds_climatology=True, use_atm_mask=True):
    
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

    #mask = mask.interp(lat=var.lat, lon=var.lon, method="nearest")
    ta_abs_pi = piok_ta.interp(plev = cose)
    if use_atm_mask==True:
        var_int = (var*mask).interp(plev = cose)
    else:
        var_int = var.interp(plev = cose)
    piok_int = piok_hus.interp(plev = cose)

    if ker=='HUANG':
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
        wid_mask=mask_pres(surf_pressure, cart_out, allkers, config_file)
        print(wid_mask)
        
        if use_ds_climatology==False:
            if use_climatology==True:
                anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int.groupby('time.month'), piok_int , dask = 'allowed')
                coso3= anoms_ok3.groupby('time.month') *dlnws(ta_abs_pi)
                process = psutil.Process(os.getpid())
                print('total RAM memory used (anoms and coso):', process.memory_info().rss/1e9)
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
                process = psutil.Process(os.getpid())
                print('total RAM memory used by dRt:', process.memory_info().rss/1e9)
            else:
                dRt = (coso3* kernel* wid_mask/100).sum('player').mean('month')

        if ker=='ERA5':
            if use_ds_climatology==False:
                dRt = (coso.groupby('time.month')*( kernel* vlevs.dp / 100) ).sum('player').groupby('time.year').mean('time')
            else:
                dRt = (coso*( kernel* vlevs.dp / 100)).sum('player').mean('month')


        dRt_glob = ctl.global_mean(dRt)
        process = psutil.Process(os.getpid())
        print('total RAM memory used for global mean (no compute):', process.memory_info().rss/1e9)
        wv= dRt_glob.compute()
        radiation[(tip, 'water-vapor')]=wv
        wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" +tip+cos +"-"+ker+"kernels.nc", format="NETCDF4")
        wv.close()
        
    return radiation

#ALL RAD_ANOM COMPUTATION
def calc_anoms_wrapper(config_file: str, ker, standard_names):
    """
   
    """
    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, standard_names)
    print("Variables to consider upload...")
    allvars = 'ts tas hus alb ta rlutcs rsutcs rlut rsut'.split()
    print("Read parameters from configuration file...")

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)

    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})

    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # If `time_range_exp` is defined use it, otherwise use `time_range`
    time_range_to_use = time_range_exp if time_range_exp else time_range_clim
    print(f"Time range used for the simulation analysis: {time_range_to_use}")

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
            surf_pressure = standardize_names(surf_pressure, standard_names)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology for Rad anomaly...")
    ref_clim_data_1=dict()
    ref_clim_data_1 = ref_clim(config_file, allvars, ker, standard_names, allkers) 
    
    anom_ps, anom_pal, anom_a, anom_wv = calc_anoms(ds, ref_clim_data_1, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_to_use, use_ds_climatology, config_file)

    return (anom_ps, anom_pal, anom_a, anom_wv)

def calc_anoms(ds, piok_rad, ker, allkers, cart_out, surf_pressure, use_climatology=True, time_range=None, use_ds_climatology=False, config_file =None):
    """
    
    """
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_ps = Rad_anomaly_planck_surf(ds, piok_rad, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology)
    else:
        anom_ps = xr.open_dataset(path)
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_pal = Rad_anomaly_planck_atm_lr(ds, piok_rad, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology)
    else:
        anom_pal = xr.open_dataset(path)
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_a = Rad_anomaly_albedo(ds, piok_rad, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology)
    else:
        anom_a = xr.open_dataset(path)
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        anom_wv = Rad_anomaly_wv(ds, piok_rad, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology)
    else:
        anom_wv = xr.open_dataset(path)  

    return anom_ps, anom_pal, anom_a, anom_wv 

##FEEDBACK COMPUTATION
def calc_fb_wrapper(config_file: str, ker, standard_names):
    """
    Wrapper function to compute radiative and cloud feedbacks based on the provided configuration file and kernel type.
    
    This function loads the necessary data and kernels from the configuration file, processes the datasets, and 
    computes the feedback coefficients for both radiative anomalies and cloud feedbacks. It takes into account time 
    ranges for climatology and experimental data, as well as surface pressure if required by the kernel type.
    
    Parameters:
    - config_file (str): Path to the configuration YAML file containing the model settings and file paths.
    - ker (str): Kernel type (e.g., 'HUANG') that determines if surface pressure is required for the analysis.
    - standard_names (dict): Dictionary of standardized variable names to be used in the dataset.
    
    Returns:
    - fb_coef (dict): Dictionary containing feedback coefficients for radiative anomalies.
    - fb_cloud (xarray.DataArray): Cloud feedback data array.
    - fb_cloud_err (xarray.DataArray): Cloud feedback error data array.
   
    """
    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, standard_names)
    print("Variables to consider upload...")
    allvars = 'ts tas hus alb ta'.split()
    allvars1= 'rlutcs rsutcs rlut rsut'.split()
    print("Read parameters from configuration file...")

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_atm_mask=True

    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})

    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # If `time_range_exp` is defined use it, otherwise use `time_range`
    time_range_to_use = time_range_exp if time_range_exp else time_range_clim

    print(f"Time range used for the simulation analysis: {time_range_to_use}")

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
            surf_pressure = standardize_names(surf_pressure, standard_names)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology...")
    allvars_combined = allvars + allvars1
    ref_clim_data = ref_clim(config_file, allvars_combined, ker, standard_names, allkers=allkers) 
    
    fb_coef, fb_cloud, fb_cloud_err = calc_fb(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_to_use, use_ds_climatology, config_file, use_atm_mask)

    return fb_coef, fb_cloud, fb_cloud_err

def calc_fb(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology=True, time_range=None, use_ds_climatology=True, config_file =None, use_atm_mask=True):
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

    if use_climatology==True:
        cos="_climatology"
        print(cos)
    else:
        cos="_21yearmean"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_atm_lr(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology, use_atm_mask)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, use_climatology, time_range, use_ds_climatology)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_wv(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology, time_range, config_file, use_ds_climatology,use_atm_mask)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()

    #compute gtas
    k=allkers[('cld', 't')]
    var_tas= ctl.regrid_dataset(ds['tas'], k.lat, k.lon) 

    if use_climatology == False:
        piok_tas=piok['tas'].drop('time')
        piok_tas['time'] = var_tas['time']
        piok_tas = piok_tas.chunk(var_tas.chunks)
        anoms_tas = var_tas - piok_tas
    else:
        anoms_tas = var_tas.groupby('time.month') - piok['tas']
        
    gtas = ctl.global_mean(anoms_tas).groupby('time.year').mean('time') 
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()

    print('feedback calculation...')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ cos+"-"+ker+"kernels.nc",  use_cftime=True)
            feedback=feedbacks.groupby((feedbacks.year-1) // 10 * 10).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res
    
    # #cloud
    print('cloud feedback calculation...')
    fb_cloud, fb_cloud_err = feedback_cloud(ds, piok, fb_coef, gtas, time_range)
    
    return fb_coef, fb_cloud, fb_cloud_err

#CLOUD FEEDBACK shell 2008
def feedback_cloud_wrapper(config_file: str, ker, standard_names):

    print("Kernel upload...")
    allkers = load_kernel_wrapper(ker, config_file)
    print("Dataset to analyze upload...")
    ds = read_data(config_file, standard_names)
    print("Variables to consider upload...")
    allvars = 'tas'
    allvars1= 'rlutcs rsutcs rlut rsut'.split()
    print("Read parameters from configuration file...")

    if isinstance(config_file, str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = config_file 
    
    cart_out = config['file_paths'].get("output")
    use_climatology = config.get("use_climatology", True)  # Default True
    use_ds_climatology = config.get("use_ds_climatology", True)
    use_atm_mask=True

    time_range_clim = config.get("time_range", {})
    time_range_exp = config.get("time_range_exp", {})

    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    # If `time_range_exp` is defined use it, otherwise use `time_range`
    time_range_to_use = time_range_exp if time_range_exp else time_range_clim
    
    print(f"Time range used for the simulation analysis: {time_range_to_use}")

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
            surf_pressure = standardize_names(surf_pressure, standard_names)
        else:
            print("Using surface pressure passed as an array.")

    print("Upload reference climatology...")
    allvars_combined = allvars + allvars1
    ref_clim_data = ref_clim(config_file, allvars_combined, ker, standard_names, allkers=allkers) 

    fb_coef = calc_fb(ds, ref_clim_data, ker, allkers, cart_out, surf_pressure, use_climatology, time_range_to_use, use_ds_climatology, config_file, use_atm_mask)

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

    fb_cloud, fb_cloud_err = feedback_cloud(ds, ref_clim_data, fb_coef, gtas, time_range_to_use)

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
            ds[nom] = ds[nom].sel(time=slice(time_range[0], time_range[1]))


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
