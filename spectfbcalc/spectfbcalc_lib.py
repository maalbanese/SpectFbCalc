#!/usr/bin/python
# -*- coding: utf-8 -*-

##### Package imports

import sys
sys.path.append('/work/users/malbanese/radspesoft/ClimTools/climtools')

import os
import glob
import re
import numpy as np
import xarray as xr
import matplotlib.cbook as cbook
import pickle
import dask.array as da
import time 
import logging 
import pandas as pd
import yaml

from climtools import climtools_lib as ctl
from matplotlib import pyplot as plt
from scipy import stats
from smmregrid import Regridder, cdo_generate_weights
from cftime import DatetimeGregorian

######################################################################
### Functions

def mytestfunction():
    print('test!')
    return

# FROM UNSTRUCTURED TO LATLON GRID X EC-EARTH4 FILES
def regrid_files(input_pattern, target_grid, method, output_dir):
    """
    Regrids input files to a target grid using the smmregrid package.

    Parameters:
        input_pattern (str): Path pattern for input files (e.g., "/path/to/files/*.nc").
        target_grid (str): Target grid in CDO format (e.g., "r180x90").
        method (str): Interpolation method supported by CDO (e.g., "ycon").
        output_dir (str): Directory to save the regridded files.
    """
    # Use glob to match input files based on the pattern
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        print(f"No files matched the pattern: {input_pattern}")
        return

    for file in input_files:
        # Load input dataset
        ds = xr.open_dataset(file)
        
        # Extract a sample of the source grid from the data
        source_grid = ds.isel(time_counter=0)  # Assuming 'time' is a dimension
        
        # Generate weights
        weights = cdo_generate_weights(source_grid, target_grid=target_grid, method=method)
        
        # Initialize Regridder with generated weights
        regridder = Regridder(weights=weights)
        
        # Apply regridding
        regridded_ds = regridder.regrid(ds)
        
        # Save the regridded file
        output_file = os.path.join(output_dir, f"{os.path.basename(file).replace('.nc', '_regridded.nc')}")
        regridded_ds.to_netcdf(output_file)
        # print(f"Regridded file saved: {output_file}")

# AGGIUNGI FUNZIONE RENAME DATASET ???


###### Broad-band kernels ######

# LOAD KERNELS
def load_kernel(config_file: str):
    """
    Loads and processes climate kernel datasets, and saves specific datasets to pickle files.

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

     # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract settings from the config file
    file_paths = config['file_paths']
    cart_k = file_paths['kernel']
    cart_ck = file_paths['output_k']
    vlevs = file_paths['pressure_level']

    variables = config['variables']
    vnams = variables['vnams']
    tips = variables['tips']

    allkers = dict()

    for tip in tips:
        for vna in vnams:
            ker = xr.load_dataset(cart_k.format(vna, tip))

            allkers[(tip, vna)] = ker.assign_coords(month = np.arange(1, 13))

    vlevs = xr.load_dataset(vlevs) 
    k = allkers[('cld', 't')].lwkernel
    pickle.dump(vlevs, open(os.path.join(cart_ck, 'vlevs.p'), 'wb')) #save vlevs
    pickle.dump(k, open(os.path.join(cart_ck,'k.p'), 'wb')) #save k
    cose = 100*vlevs.player
    pickle.dump(cose, open(os.path.join(cart_ck, 'cose.p'), 'wb'))
    pickle.dump(allkers, open(os.path.join(cart_ck, 'allkers.p'), 'wb'))
    return allkers

# PIMEAN COMPUTATION
def climatology(config_file: str, allvars: str):
    """
    Computes climatological means or running means for specified variables, processes data using kernels, and 
    saves the results to netCDF files.

    Parameters:
    -----------
    config_file : str
        Path to the YAML configuration file that defines the variables, file paths, and other parameters.

    allvars : str
        Variable name(s) to process. For the `alb` case, it automatically processes the 'rsus' and 'rsds' components.

    Returns:
    --------
    piok : xarray.DataArray
        Processed and regridded dataset.
    """

    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract settings from the config file
    variables = config['variables']
    file_paths = config['file_paths']
    time_chunk = config['chunk_size']
    
    use_climatology = variables.get(allvars, {}).get('use_climatology', True)
    time_range = variables.get(allvars, {}).get('time_range', ["1990-01-01", "2021-12-31"])
    if allvars in ['hus', 'ta']:
        filin_pi = file_paths['reference_dataset_pl']
    else:
        filin_pi = file_paths['reference_dataset']
    cart_k = file_paths['kernel']
    cart_ck = file_paths['output_k']

    # Initialize kernel
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers = load_kernel(cart_k, cart_ck)
    k = pickle.load(open(k_file_path, 'rb'))

    if allvars == 'alb':
        allvars = ['rsus', 'rsds']
        piok = {}
        
        for vnam in allvars:
            filist = glob.glob(filin_pi.format(vnam))
            filist.sort()
            var = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)
            
            if use_climatology:
                var_mean = var.groupby('time_counter.month').mean()
                piok[vnam] = ctl.regrid_dataset(var_mean, k.lat, k.lon).compute()
            else:
                var = var.sel(time_counter = slice(time_range[0], time_range[1]))
                piok[vnam] = ctl.regrid_dataset(var[vnam], k.lat, k.lon)
        
        alb = piok['rsus'] / piok['rsds']
        if not use_climatology:
            alb = ctl.regrid_dataset(alb, k.lat, k.lon)
            alb = ctl.running_mean(alb, 252)
        
        piok = alb
    
    else:
        filist = glob.glob(filin_pi.format(allvars))
        filist.sort()
        var = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)
        
        if use_climatology:
            var_mean = var.groupby('time_counter.month').mean()
            var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)
            piok = var_mean[allvars].compute()
        else:
            var = var.sel({'time_counter': slice(time_range[0], time_range[1])})
            piok = ctl.regrid_dataset(var[allvars], k.lat, k.lon)
            piok = ctl.running_mean(piok, 252)

    return piok

# MASKING FUNCTIONS
# Tropopause calculation (Reichler 2003) MODIFICATA X CLIMATOLOGIE MENSILI
def mask_atm(config_file: str):
    """
    Generates a mask for atmospheric temperature data based on the lapse rate threshold.
    as in (Reichler 2003) 

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'ta').
        Placeholders should be formatted as `{}` to allow string formatting.

    time_chunk : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    mask : xarray.DataArray
        A mask array where:
        - Values are 1 where the lapse rate (`laps1`) is less than or equal to -2 K/km.
        - Values are NaN elsewhere.
    """

    # Load the configuration from the YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract file paths and variable names from the config
    file_paths = config['file_paths']
    cart_ck = file_paths['output_k']
    cart_k = file_paths['kernel']
    time_chunk = config['chunk_size']
    climatology = config.get('climatology', False)
    
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        allkers=dict()
        print("Kernel file not found. Running ker() to generate it.")
        allkers= load_kernel(cart_k,cart_ck)  # Ensure that ker() is properly defined elsewhere in the code
    k=pickle.load(open(cart_ck + 'k.p', 'rb')) # from cart_out
    
    # Load temperature data from files
    filin_4c = file_paths['reference_dataset_pl']  # Path to the dataset
    filist = glob.glob(filin_4c.format('ta'))
    filist.sort()
    temp = xr.open_mfdataset(filist, chunks={'time_counter': time_chunk}, use_cftime=True)

    # Monthly climatology or standard data
    if climatology: 
        # Monthly climatology 
        temp_monthly = temp.resample(time_counter='MS').mean()  # Resample to monthly data
        temp_clim = temp_monthly.groupby('time_counter.month').mean() 
        # Lapse-rate climatology based
        A=(temp_clim.pressure_levels/temp_clim['ta'])*(9.81/1005) # modified 
        laps1=(temp_clim['ta'].diff(dim='pressure_levels'))*A  # derivative on the vertical = laspe-rate, modified
    else:
        # Standard lapse-rate
        A = (temp.pressure_levels / temp['ta']) * (9.81 / 1005)
        laps1 = (temp['ta'].diff(dim='pressure_levels')) * A  # Derivata verticale = lapse rate

    laps1=laps1.where(laps1<=-2)
    mask = laps1/laps1

    mask = ctl.regrid_dataset(mask, k.lat, k.lon).compute()

    return mask

# Mask for surf pressure
def mask_pres(config_file:str):
    """
    Generates a pressure mask based on climatological surface pressure and vertical levels.

    Parameters:
    -----------
    pressure_directory : str
        Path to the input NetCDF file or directory containing surface pressure (`ps`) data.
        The data is expected to include a time dimension for monthly averaging.

    cart_out : str
        Path template to save the outputs. 
        
    cart_k : str
        Path template to the kernel dataset files, required for generating vertical level data.
        Used by the `ker()` function if the kernel data is missing.

    Returns:
    --------
    wid_mask : xarray.DataArray
        A 3D mask array (pressure level, latitude, longitude) where:
        - Values are NaN for pressure levels above the surface pressure.
        - Values represent vertical level pressure differences for levels below the surface pressure.
    """

    start_time = time.time()

    # Load the configuration from the YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract file paths and other configuration parameters
    file_paths = config['file_paths']
    pressure_directory = file_paths['pressure_data']
    cart_ck = file_paths['output_k']
    cart_k = file_paths['kernel']
    
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        allkers=dict()
        print("Kernel file not found. Running ker() to generate it.")
        allkers= load_kernel(cart_k,cart_ck)  # Ensure that ker() is properly defined elsewhere in the code
    k=pickle.load(open(cart_ck + 'k.p', 'rb'))#prendi da cart_out
    vlevs=pickle.load(open(cart_ck + 'vlevs.p', 'rb'))#prendi da cart_out

    #ADDED
    all_files = os.listdir(pressure_directory)
    ps_files = [os.path.join(pressure_directory, f) for f in all_files if re.search(r's000_atm_cmip6_1m_\d{4}-\d{4}_regridded\.nc$', f)]
    if not ps_files:
        raise FileNotFoundError("No matching files found in the pressure directory.")
    #
    ps = xr.open_mfdataset(ps_files, combine='by_coords')
    
    #ADDED
    # Convert time_variable to datetime if necessary
    if not np.issubdtype(ps['time_counter'].dtype, np.datetime64):
        ps['time_counter'] = xr.cftime_range(start='1990-01-01', periods=ps.dims['time_counter'], freq='D') # To modify (try to adjust it)

    # Resample to monthly and calculate climatology
    ps_monthly = ps.resample(time_counter='ME').mean()
    psclim = ps_monthly.groupby('time_counter.month').mean()
    #
    psye = psclim['ps'].mean('month')
    psye_rg = ctl.regrid_dataset(psye, k.lat, k.lon).compute()
    
    wid_mask = np.empty([len(vlevs.player)] + list(psye_rg.shape))
    for ila in range(len(psye_rg.lat)):
    #print(ila)
        for ilo in range(len(psye_rg.lon)):
            ind = np.where((psye_rg[ila, ilo].values/100. - vlevs.player.values) > 0)[0][0]
            wid_mask[:ind, ila, ilo] = np.nan
            wid_mask[ind, ila, ilo] = psye_rg[ila, ilo].values/100. - vlevs.player.values[ind]
            wid_mask[ind+1:, ila, ilo] = vlevs.dp.values[ind+1:]

    wid_mask = xr.DataArray(wid_mask, dims = k.dims[1:], coords = k.drop('month').coords)

    logging.info("Function completed in %.2f seconds.", time.time() - start_time)

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
    if isinstance(T, xr.DataArray) and isinstance(T.data, da.core.Array):
        ws = da.where(T >= 273, pliq0, pice0)    # Dask equivalent of np.where is da.where
        ws1 = da.where(T1 >= 273, pliq1, pice1)
    else:
        ws = np.where(T >= 273, pliq0, pice0)
        ws1 = np.where(T1 >= 273, pliq1, pice1)
    
    # Calculate the inverse of the derivative dws
    dws = ws / (ws1 - ws)

    if isinstance(dws, np.ndarray):
        dws = ctl.transform_to_dataarray(T, dws, 'dlnws')
    
    return dws


# TS_ANOM AND GTAS AND PLANK SURF COMPUTATION
def fb_planck_surf(config_file: str):
    """
    Computes the surface Planck feedback using temperature anomalies and precomputed kernels.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'ts').
        Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) temperature files (required to compute anomalies) or for the standard simulation.

    cart_out : str
        Path template to save the outputs. 

    cart_ck : str
        Path to save kernel objects.

    cart_k : str
        Path template to the kernel dataset files, used by the `ker()` function if kernel data is missing.

    use_climatology : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_ta.nc`).
        If False, uses running mean data from precomputed PI files (`piok_ta_21y.nc`).

        TO ASK


    time_chunk : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    feedbacks : dict
        A dictionary containing the computed global annual mean surface Planck feedback
        for clear-sky (`clr`) and all-sky (`cld`) conditions. The keys of the dictionary are:
        - `('clr', 'planck-surf')`: Clear-sky surface Planck feedback.
        - `('cld', 'planck-surf')`: All-sky surface Planck feedback.

    Additional Outputs:
    -------------------
    The function also saves the following files in the `cart_out` directory:
    - **`ts_anom.nc`**: NetCDF file containing temperature anomalies (`ts_anom`) relative to the PI climatology.
    - **`gtas.nc`**: NetCDF file with global temperature anomaly series (`gtas`), grouped by year.
    - **`dRt_planck-surf_global_clr.nc`**: Clear-sky global surface Planck feedback as a NetCDF file.
    - **`dRt_planck-surf_global_cld.nc`**: All-sky global surface Planck feedback as a NetCDF file.

    Depending on the value of `use_climatology`, the function saves different NetCDF files to the `cart_out` directory:
    If `use_climatology=True` it adds "_climatology", elsewhere it adds "_21yearmean"
    """   

    logging.info("Starting fb_planck_surf function.")
    start_time = time.time()

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    file_paths = config['file_paths']
    variables = config['variables']
    time_chunk = config['chunk_size']

    filin_4c = file_paths['experiment_dataset']
    filin_pi = file_paths['reference_dataset']
    cart_out = file_paths['output']
    cart_ck = file_paths['output_k']
    cart_k = file_paths['kernel']
    
    allvars='ts'
    use_climatology = variables.get(allvars, {}).get('use_climatology', True)
    time_range = variables.get(allvars, {}).get('time_range', ["1990-01-01", "2021-12-31"])
    
    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        logging.info("Kernel file not found. Running ker() to generate it.")
        allkers= load_kernel(cart_k, cart_ck) # Ensure that ker() is properly defined elsewhere in the code
    else:
       logging.info("Loading kernel data from file.")
       allkers=pickle.load(open(cart_ck + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_ck + 'k.p', 'rb')) # from cart_out

    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    logging.info("Loading input files.") 
    filist = glob.glob(filin_4c.format('ts'))
    filist.sort()
    var = xr.open_mfdataset(filist, chunks= {'time_counter': time_chunk}, use_cftime=True) 
    var = var.sel(time_counter = slice(time_range[0], time_range[1]))
    var = ctl.regrid_dataset(var['ts'], k.lat, k.lon)

    logging.info("Computing climatology.")
    piok=climatology(config_file, allvars)

    if use_climatology == False:
        piok=piok.drop('time_counter')
        piok['time_counter'] = var['time_counter']
        piok = piok.chunk(var.chunks)
    
    logging.info("Computing anomalies.")
    var_clim = var.groupby('time_counter.month').mean() # ADDED
    anoms_clim = var_clim - piok #ADDED
    anoms = var.groupby('time_counter.month') - piok
    ts_anom = anoms.compute()
    ts_anom.to_netcdf(cart_out+ "ts_anom"+cos+".nc", format="NETCDF4")
    logging.info("Computing global temperature anomalies.")
    gtas = ctl.global_mean(anoms).groupby('time_counter.year').mean('time_counter')
    gtas.to_netcdf(cart_out+ "gtas"+cos+".nc", format="NETCDF4")
 
    for tip in ['clr', 'cld']:
        logging.info("Processing feedbacks for %s.", tip)
        kernel = allkers[(tip, 'ts')].lwkernel

        dRt = xr.apply_ufunc(lambda x, ker: x*ker, anoms_clim, kernel, dask = 'allowed').mean('month') # MODIFIED
        dRt_glob = ctl.global_mean(dRt)
        planck= dRt_glob.compute()
        feedbacks[(tip, 'planck-surf')]=planck
        planck.to_netcdf(cart_out+ "dRt_planck-surf_global_" +tip +cos+".nc", format="NETCDF4")
        logging.info("Saved %s feedback to file.", tip)

    logging.info("Function completed in %.2f seconds.", time.time() - start_time)
        
    return(feedbacks)


# PLANK-ATMO AND LAPSE RATE WITH VARIABLE TROPOPAUSE
def fb_plank_atm_lr(config_file: str):
    """
    Computes atmospheric Planck and lapse-rate feedbacks using temperature anomalies, kernels, and masking.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'ta').
        Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
         Path template for the preindustrial (PI) temperature files (required to compute anomalies) or for the standard simulation.

    cart_out : str
        Output directory where precomputed files (`k.p`, `cose.p`, `vlevs.p`) and results are stored.

    cart_ck : str
        Path to save kernel objects.

    cart_k : str
        Path template to the kernel dataset files, used by the `ker()` function if kernel data is missing.

    pressure_directory : str
        Directory containing surface pressure (`ps`) data required for pressure masking.

    use_climatology : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_ta.nc`).
        If False, uses running mean data from precomputed PI files (`piok_ta_21y.nc`).


    time_chunk : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    feedbacks : dict
        A dictionary containing the computed global annual mean Planck and lapse-rate feedbacks 
        for clear-sky (`clr`) and all-sky (`cld`) conditions. The keys of the dictionary are:
        - `('clr', 'planck-atmo')`: Clear-sky atmospheric Planck feedback.
        - `('cld', 'planck-atmo')`: All-sky atmospheric Planck feedback.
        - `('clr', 'lapse-rate')`: Clear-sky lapse-rate feedback.
        - `('cld', 'lapse-rate')`: All-sky lapse-rate feedback.

    Additional Outputs:
    -------------------
    The function saves the following files to the `cart_out` directory:
    - **`ta_abs_pi.nc`**: Interpolated preindustrial absolute temperature profile at kernel levels.
    - **`dRt_planck-atmo_global_clr.nc`**: Clear-sky atmospheric Planck feedback as a NetCDF file.
    - **`dRt_planck-atmo_global_cld.nc`**: All-sky atmospheric Planck feedback as a NetCDF file.
    - **`dRt_lapse-rate_global_clr.nc`**: Clear-sky lapse-rate feedback as a NetCDF file.
    - **`dRt_lapse-rate_global_cld.nc`**: All-sky lapse-rate feedback as a NetCDF file.

    Depending on the value of `use_climatology`, the function saves different NetCDF files to the `cart_out` directory:
    If `use_climatology=True` it adds "_climatology", elsewhere it adds "_21yearmean"
    """

    start_time = time.time()
    logging.info("Starting fb_plank_atm_lr function.")

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    file_paths = config['file_paths']
    variables = config['variables']
    time_chunk = config['chunk_size']

    filin_4c = file_paths['experiment_dataset_pl']
    filin_pi = file_paths['reference_dataset_pl']
    cart_out = file_paths['output']
    cart_ck = file_paths['output_k']
    cart_k = file_paths['kernel']
    pressure_directory = file_paths['pressure_data']

    allvars='ta'
    use_climatology = variables.get(allvars, {}).get('use_climatology', True)
    time_range = variables.get(allvars, {}).get('time_range', ["1990-01-01", "2021-12-31"])

    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        logging.info("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_ck)  # Ensure that ker() is properly defined elsewhere in the code
    else: 
        logging.info("Loading kernel data from file.")
        allkers=pickle.load(open(cart_ck + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_ck + 'k.p', 'rb')) # from cart_out
    cose=pickle.load(open(cart_ck + 'cose.p', 'rb'))
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    logging.info("Loading input files.") 
    filist = glob.glob(filin_4c.format('ta'))
    filist.sort()
    var = xr.open_mfdataset(filist, chunks= {'time_counter': time_chunk}, use_cftime=True) 
    var = var['ta']
    var = var.sel(time_counter = slice(time_range[0], time_range[1]))
    var = ctl.regrid_dataset(var, k.lat, k.lon)

    logging.info("Computing climatology.")
    piok=climatology(config_file, allvars)

    if use_climatology==False:
        piok=piok.drop('time_counter')
        piok['time_counter'] = var['time_counter']
    
    logging.info("Computing anomalies.")
    var_clim = var.groupby('time_counter.month').mean() # ADDED
    anoms_ok_clim = var_clim - piok #ADDED
    anoms_ok = var.groupby('time_counter.month') - piok 

    ta_abs_pi = piok.interp(pressure_levels = cose) # I changed plev with pressure_levels according to the variable name in .nc files
    ta_abs_pi.to_netcdf(cart_out+ "ta_abs_pi"+cos+".nc", format="NETCDF4")
    mask=mask_atm(config_file)
    wid_mask=mask_pres(config_file)
    anoms_ok_clim = (anoms_ok_clim*mask).interp(pressure_levels = cose) # MODIFIED
    ts_anom=xr.open_dataarray(cart_out+"ts_anom"+cos+".nc", chunks = {'time_counter': time_chunk}, use_cftime=True) 

    for tip in ['clr','cld']:
        logging.info("Processing feedbacks for %s.", tip)
        kernel = allkers[(tip, 't')].lwkernel
        anoms_lr = (anoms_ok_clim - ts_anom.mean('time_counter')) # MODIFIED
        anoms_unif = (anoms_ok_clim - anoms_lr) # MODIFIED

        try: 
            dRt_unif = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_unif, kernel, wid_mask/100., dask = 'allowed')).sum('player').mean('month') # MODIFIED
        except Exception as e:
            logging.error(f"Error in apply_ufunc for dRt_unif: {e}")
            raise
        
        dRt_lr = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_lr, kernel, wid_mask/100., dask = 'allowed')).sum('player').mean('month') # MODIFIED
        
        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        feedbacks[(tip,'planck-atmo')]=feedbacks_atmo
        feedbacks[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip +cos+".nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  +cos+".nc", format="NETCDF4")
        logging.info("Saved %s feedback to file.", tip)

    logging.info("fb_plank_atm_lr function completed in %.2f seconds.", time.time() - start_time)  
   
    return(feedbacks)


# ALBEDO FEEDBACKS COMPUTATION !!! TO MODIFY !!!
def fb_albedo(config_file: str):
    """
    Computes the albedo feedback using surface albedo anomalies and precomputed kernels.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'rsus', 'rsds').
        Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) temperature files (required to compute anomalies) or for the standard simulation.

    cart_out : str
        Output directory where precomputed files (`k.p`) and results are stored.

    cart_ck : str
        Path to save kernel objects.

    cart_k : str
        Path template to the kernel dataset files, used by the `ker()` function if kernel data is missing.

    use_climatology : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_alb.nc`).
        If False, uses running mean data from precomputed PI files (`piok_alb_21y.nc`).

    time_chunk : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    feedbacks : dict
        A dictionary containing the computed global annual mean albedo feedbacks for clear-sky and all-sky conditions. 
        The keys of the dictionary are:
        - `('clr', 'albedo')`: Clear-sky albedo feedback.
        - `('cld', 'albedo')`: All-sky albedo feedback.

    Additional Outputs:
    -------------------
    The function saves the following files to the `cart_out` directory:
    - **`dRt_albedo_global_clr.nc`**: Clear-sky albedo feedback as a NetCDF file.
    - **`dRt_albedo_global_cld.nc`**: All-sky albedo feedback as a NetCDF file.

    Depending on the value of `use_climatology`, the function saves different NetCDF files to the `cart_out` directory:
    If `use_climatology=True` it adds "_climatology", elsewhere it adds "_21yearmean"
    """

    logging.info("Starting fb_planck_surf function.")
    start_time = time.time()

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    file_paths = config['file_paths']
    variables = config['variables']
    time_chunk = config['chunk_size']

    filin_4c = file_paths['experiment_dataset']
    filin_pi = file_paths['reference_dataset']
    cart_out = file_paths['output']
    cart_ck = file_paths['output_k']
    cart_k = file_paths['kernel']

    allvars='alb'
    use_climatology = variables.get(allvars, {}).get('use_climatology', True)
    time_range = variables.get(allvars, {}).get('time_range', ["1990-01-01", "2021-12-31"])

    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        logging.info("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_ck)  # Ensure that ker() is properly defined elsewhere in the code
    else:
        logging.info("Loading kernel data from file.")
        allkers=pickle.load(open(cart_ck + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_ck + 'k.p', 'rb')) # from cart_out
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"
  
    logging.info("Loading input files.") 
    filist_1 = glob.glob(filin_4c.format('rsus')) # I changed the code because in EC-EARTH4 there is directly albsn
    filist_1.sort()
    var_rsus = xr.open_mfdataset(filist_1, chunks = {'time_counter': time_chunk}, use_cftime=True)['rsus']
    filist_2 = glob.glob(filin_4c.format('rsds')) # I changed the code because in EC-EARTH4 there is directly albsn
    filist_2.sort()
    var_rsds = xr.open_mfdataset(filist_2, chunks = {'time_counter': time_chunk}, use_cftime=True)['rsds']
    logging.info("Dataset size: %s, %s", var_rsus.shape, var_rsds.shape ) 
    var = var_rsus/var_rsds
    var = var.sel(time_counter = slice(time_range[0], time_range[1]))
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    logging.info("Computing climatology.")
    piok=climatology(config_file, allvars)

    if use_climatology==False:
        piok=piok.drop('time_counter')
        piok['time_counter'] = var['time_counter']

    # Removing inf and nan from alb
    piok = piok.where(piok > 0., 0.)
    var = var.where(var > 0., 0.)

    var_clim = var.groupby('time_counter.month').mean() # MODIFIED
    anoms_clim = var_clim - piok 

    for tip in [ 'clr','cld']:
        logging.info("Processing feedbacks for %s.", tip)
        kernel = allkers[(tip, 'alb')].swkernel

        dRt = xr.apply_ufunc(lambda x, ker: x*ker, anoms_clim, kernel, dask = 'allowed').mean('month')
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        feedbacks[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip +cos+".nc", format="NETCDF4")
        logging.info("Saved %s feedback to file.", tip)

    logging.info("Function completed in %.2f seconds.", time.time() - start_time)
        
    return(feedbacks)


# W-V FEEDBACKS COMPUTATION
def fb_wv(config_file: str):
    
    """
    Computes the water vapor feedback using specific humidity (hus) anomalies, precomputed kernels,
    and vertical integration.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files containing climate data, with a placeholder for the variable
        name (e.g., 'hus'). Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) temperature files (required to compute anomalies) or for the standard simulation.

    cart_out : str
        Output directory where precomputed files (e.g., `k.p`, `cose.p`, `vlevs.p`) and results are stored.

    cart_ck : str
        Path to save kernel objects.

    cart_k : str
        Path template for kernel dataset files, used by the `ker()` function if kernel data is missing.

    pressure_directory : str
        Directory containing surface pressure (`ps`) data required for pressure masking.

    use_climatology : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_hus.nc`).
        If False, uses running mean data from precomputed PI files (`piok_hus_21y.nc`).

    time_chunk : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

     Returns:
    --------
    feedbacks : dict
        A dictionary containing the computed global annual mean water vapor feedbacks for clear-sky and all-sky conditions.
        The keys of the dictionary are:
        - `('clr', 'water-vapor')`: Clear-sky water vapor feedback.
        - `('cld', 'water-vapor')`: All-sky water vapor feedback.

    Additional Outputs:
    -------------------
    The function saves the following files to the `cart_out` directory:
    - **`dRt_water-vapor_global_clr.nc`**: Clear-sky water vapor feedback as a NetCDF file.
    - **`dRt_water-vapor_global_cld.nc`**: All-sky water vapor feedback as a NetCDF file.

    Depending on the value of `use_climatology`, the function saves different NetCDF files to the `cart_out` directory:
    If `use_climatology=True` it adds "_climatology", elsewhere it adds "_21yearmean"
    """

    logging.info("Starting fb_planck_surf function.")
    start_time = time.time()

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    file_paths = config['file_paths']
    variables = config['variables']
    time_chunk = config['chunk_size']

    filin_4c = file_paths['experiment_dataset_pl']
    filin_pi = file_paths['reference_dataset_pl']
    cart_out = file_paths['output']
    cart_ck = file_paths['output_k']
    cart_k = file_paths['kernel']
    pressure_directory = file_paths['pressure_data']

    allvars='hus'
    use_climatology = variables.get(allvars, {}).get('use_climatology', True)
    time_range = variables.get(allvars, {}).get('time_range', ["1990-01-01", "2021-12-31"])

    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        logging.info("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_ck)  # Ensure that ker() is properly defined elsewhere in the code
    else:
        logging.info("Loading kernel data from file.")
        allkers=pickle.load(open(cart_ck + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_ck + 'k.p', 'rb')) # from cart_out
    cose=pickle.load(open(cart_ck + 'cose.p', 'rb'))
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    ta_abs_pi=xr.open_dataarray(cart_out+"ta_abs_pi"+cos+".nc",  chunks = {'time_counter': time_chunk},  use_cftime=True)
    mask=mask_atm(config_file)
    wid_mask=mask_pres(config_file)

    logging.info("Loading input files.") 
    filist = glob.glob(filin_4c.format('hus')) 
    filist.sort()
    var = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)
    var = var['hus']
    var = var.sel(time_counter = slice(time_range[0], time_range[1]))
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    allvars='hus'
    logging.info("Computing climatology.")
    piok=climatology(config_file, allvars)

    if use_climatology==False:
        piok=piok.drop('time_counter')
        piok['time_counter'] = var['time_counter']

    logging.info("Computing anomalies.")
    var_int = (var*mask).interp(pressure_levels = cose)
    piok_int = piok.interp(pressure_levels = cose)
    var_clim = var_int.groupby('time_counter.month').mean()
        
    anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_clim, piok_int , dask = 'allowed') # MODIFIED
    coso3= xr.apply_ufunc(lambda x, ta: x*ta, anoms_ok3, dlnws(ta_abs_pi), dask = 'allowed') #(using dlnws) # MODIFIED
    
    for tip in ['clr','cld']:
        logging.info("Processing feedbacks for %s.", tip)
        kernel_lw = allkers[(tip, 'wv_lw')].lwkernel
        kernel_sw = allkers[(tip, 'wv_sw')].swkernel
        kernel = kernel_lw + kernel_sw
        dRt = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, coso3, kernel, wid_mask/100., dask = 'allowed')).sum('player').mean('month')
        dRt_glob = ctl.global_mean(dRt)
        wv= dRt_glob.compute()
        feedbacks[(tip, 'water-vapor')]=wv
        wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" +tip+cos +".nc", format="NETCDF4")
        logging.info("Saved %s feedback to file.", tip)

    logging.info("Function completed in %.2f seconds.", time.time() - start_time)


    return(feedbacks)


# TOTAL FEEDBACKS COMPUTATION
def calc_fb(filin_4c:str, filin_4c1:str, filin_pi:str, filin_pi1:str, cart_out:str, cart_ck:str, cart_k:str, pressure_directory:str, use_climatology=True, time_chunk=12):
    """
    Computes climate feedback coefficients by combining feedback components (Planck, albedo, water vapor, etc.)
    with global temperature anomalies and performs a linear regression analysis.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files containing climate data, with a placeholder for the variable
        name (e.g., 'ta'). Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) temperature files (required to compute anomalies) or for the standard simulation.

    cart_out : str
        Output directory where intermediate feedback data and results are stored.

    cart_k : str
        Path template for kernel dataset files, used by feedback functions like `fb_planck_surf`.

    pressure_directory : str
        Path to the pressure-level data directory, used for additional computations such as water vapor feedback.

    use_climatology : bool, optional (default=True)
        If True, computes anomalies relative to the mean climatology from precomputed PI files.
        If False, computes anomalies relative to running mean PI data.

    time_chunk : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    dict
        A dictionary (`fb_coef`) containing feedback coefficients for each combination of:
        - Feedback component: `planck-surf`, `planck-atmo`, `lapse-rate`, `water-vapor`, `albedo`.
        - Sky condition: `clr` (clear-sky), `cld` (cloudy-sky).
        Feedback coefficients are computed as the slope of the linear regression between global temperature
        anomalies (`gtas`) and feedback response (`dRt`).

    Notes:
    ------
     **Feedback Computation**:
       - This function relies on precomputed feedback outputs (e.g., `dRt_planck-surf_global_clr.nc`).
       - If the required feedback files are missing, corresponding feedback functions (`fb_planck_surf`, `fb_plank_atm_lr`, etc.) are automatically called.

    Outputs:
    --------
    The function returns a dictionary with regression results for each feedback component and sky condition.
    Intermediate feedback outputs (e.g., `dRt_planck-surf_global_clr.nc`) are generated as needed.
    """    

    logging.info("Starting fb_planck_surf function.")
    start_time = time.time()

    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    logging.info('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_planck_surf(filin_4c, filin_pi, cart_out, cart_ck, cart_k, use_climatology, time_chunk)
    logging.info('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_plank_atm_lr(filin_4c1, filin_pi1, cart_out, cart_ck, cart_k, pressure_directory, use_climatology, time_chunk)
    logging.info('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_albedo(filin_4c, filin_pi, cart_out, cart_ck, cart_k, use_climatology, time_chunk)
    logging.info('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_wv(filin_4c1, filin_pi1, cart_out, cart_ck, cart_k, pressure_directory, use_climatology, time_chunk)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()
    gtas=xr.open_dataarray(cart_out+"gtas"+cos+".nc",  use_cftime=True)
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()
    logging.info('calcolo feedback')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ cos+".nc",  use_cftime=True)
            feedback=feedbacks.groupby((feedbacks.year-1) // 10 * 10).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res

    logging.info("Function completed in %.2f seconds.", time.time() - start_time)
    
    return(fb_coef)


# CLOUD FEEDBACK shell 2008
def fb_cloud(filin_4c:str, filin_4c1:str, filin_pi:str, filin_pi1:str, cart_out:str, cart_ck:str, cart_k:str, pressure_directory, use_climatology=True, time_chunk=12):
    """
    Computes the cloud feedback coefficient based on the Shell (2008) approach,
    using radiative fluxes, clear-sky fluxes, and feedback components. 

    Parameters:
    -----------
    filin_4c : str
        Path template for NetCDF files containing radiative flux variables (e.g., `rlut`, `rsut`).
        The template should include `{}` for formatting with variable names.

    filin_4c1 : str
        Path template for NetCDF files containing clear-sky radiative flux variables (`rsutcs`, `rlutcs` or `rsdt`,`rsntcs`, `rlntcs`).
        
        rsutcs: clear-sky reflected (upward) shortwave radiation at TOA (e.g., sunlight reflected back to space without clouds).
        rlutcs: clear-sky outgoing (upward) longwave radiation at TOA (thermal energy radiated to space without clouds). 
        rsdt: total incoming solar radiation at TOA.
        rsntcs: clear-sky net shortwave radiation at TOA -> net shortwave radiation (incoming minus reflected) under clear-sky conditions at TOA. (rsntcs = rsdt − rsutcs)
        rlntcs: clear-sky net longwave radiation at TOA -> net thermal radiation under clear-sky conditions at TOA. rlntcs= − rlutcs 
        (generally no downward (incoming from space) longwave radiation at TOA)
        
        The template should include `{}` for formatting with variable names.

        !!! NEL MIO CASO TUTTE LE VARIABILI SONO NELLO STESSO FILE
    
    filin_pi : str
        Path template for the preindustrial (PI) temperature files (required to compute anomalies) or for the standard simulation.

    cart_out : str
        Output directory where intermediate and precomputed datasets (e.g., `k.p`, `gtas.nc`) are stored.
        
    cart_k : str
        Path template for kernel dataset files, used by the `ker()` function if kernel data is missing.

    pressure_directory : str
        Path to the pressure-level data directory, used for additional computations such as water vapor feedback.

    use_climatology : bool, optional (default=True)
        If True, computes anomalies relative to the mean climatology from precomputed PI files.
        If False, computes anomalies relative to running mean PI data.

    time_chunk : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    tuple
        A tuple containing:
        - `fb_cloud` (float): The computed cloud feedback coefficient, representing the change in radiative
          forcing due to cloud processes.
        - `fb_cloud_err` (float): The estimated uncertainty (standard error) of the cloud feedback coefficient.

    References:
    -----------
    - Shell et al. (2008): Framework for decomposing climate feedbacks.
    - Caldwell et al. (2016): Analysis of cloud feedback uncertainties.

    """

    logging.info("Starting fb_cloud function.")
    start_time = time.time()

    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_ck, 'k.p')
    if not os.path.exists(k_file_path):
        logging.info("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_ck)  # Ensure that ker() is properly defined elsewhere in the code   
    k=pickle.load(open(cart_ck + 'k.p', 'rb'))

    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    pimean=dict()
    fb_coef=dict()
    fb_coef=calc_fb(filin_4c, filin_4c1, filin_pi, filin_pi1, cart_out, cart_ck, cart_k, pressure_directory, use_climatology, time_chunk)
    
    logging.info("Loading input files.") 
    filist = glob.glob(filin_4c.format('rlut'))
    filist.sort()
    rlut = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)['rlut']
    rlut = rlut.sel(time_counter = slice('1991-01-01', '2001-12-31')) # To modify (try to adjust it)
 
    filist = glob.glob(filin_4c.format('rsut'))
    filist.sort()
    rsut = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)['rsut']
    rsut = rsut.sel(time_counter = slice('1991-01-01', '2001-12-31')) # To modify (try to adjust it)

    filist = glob.glob(filin_4c.format('rsdt')) 
    filist.sort()
    rsdt = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)['rsdt']
    rsdt = rsdt.sel(time_counter = slice('1991-01-01', '2001-12-31')) # To modify (try to adjust it)

    filist = glob.glob(filin_4c.format('rsntcs')) 
    filist.sort()
    rsntcs = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)['rsntcs']
    rsntcs = rsntcs.sel(time_counter = slice('1991-01-01', '2001-12-31')) # To modify (try to adjust it)
    filist = glob.glob(filin_4c.format('rlntcs'))
    filist.sort()
    rlntcs = xr.open_mfdataset(filist, chunks = {'time_counter': time_chunk}, use_cftime=True)['rlntcs']
    rlntcs = rlntcs.sel(time_counter = slice('1991-01-01', '2001-12-31')) # To modify (try to adjust it)


    N = - rlut - rsut
    # ADDED to obtain rsutcs and rlutcs
    rsutcs = rsdt - rsntcs
    rlutcs = - rlntcs 
    N0 = - rsutcs - rlutcs # net energy flux at the top of the atmosphere in clear-sky conditions
    
    crf = (N0 - N) # how much clouds modify the Earth's radiation budget.
    crf = crf.groupby('time_counter.year').mean('time_counter')


    N = N.groupby('time_counter.year').mean('time_counter')
    N0 = N0.groupby('time_counter.year').mean('time_counter')

    crf_glob = ctl.global_mean(crf).compute()
    N_glob = ctl.global_mean(N).compute()
    N0_glob = ctl.global_mean(N0).compute()

    crf_glob= crf_glob.groupby((crf_glob.year-1) // 10 * 10).mean(dim='year')
    N_glob=N_glob.groupby((N_glob.year-1) // 10 * 10).mean(dim='year')
    N0_glob=N0_glob.groupby((N0_glob.year-1) // 10 * 10).mean(dim='year')

    gtas=xr.open_dataarray(cart_out+"gtas_climatology.nc",  use_cftime=True)
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()
    res_N = stats.linregress(gtas, N_glob)
    res_N0 = stats.linregress(gtas, N0_glob)
    res_crf = stats.linregress(gtas, crf_glob)

    allvars = 'rlntcs rsntcs rlut rsut'.split()
    for vnams in allvars:
        filist = glob.glob(filin_pi.format(vnams))
        filist.sort()
        var = xr.open_mfdataset(filist, chunks={'time_counter': time_chunk}, use_cftime=True)
        var_mean = var.mean('time_counter').mean()
        var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)

        # ADDED
        # Store variables in pimean
        if vnams in ['rsntcs', 'rlntcs']:
            # Save these for later computation of rsutcs and rlutcs
            pimean[vnams] = var_mean[vnams].compute()
        elif vnams in ['rlut', 'rsut']:
            # Save rlut and rsut directly
            pimean[vnams] = var_mean[vnams].compute()

    # ADDED
    # Compute rsutcs and rlutcs from rsntcs and rlntcs
    logging.info("Compunting rsutcs and rlutcs.") 
    pimean['rsutcs'] = rsdt.mean('time_counter') - pimean['rsntcs']
    pimean['rlutcs'] = -pimean['rlntcs']
    
    F0 = res_N0.intercept + pimean[('rlutcs')] + pimean[('rsutcs')] 
    F = res_N.intercept + pimean[('rlut')] + pimean[('rsut')]
    F0.compute()
    F.compute()

    F_glob = ctl.global_mean(F)
    F0_glob = ctl.global_mean(F0)
    F_glob = F_glob.compute()
    F0_glob = F0_glob.compute()
    
    logging.info("Cloud feedback coefficient and uncertainty calculation")
    fb_cloud = -res_crf.slope + np.nansum([fb_coef[( 'clr', fbn)].slope - fb_coef[('cld', fbn)].slope for fbn in fbnams]) # from Caldwell 2016
    fb_cloud_err = np.sqrt(res_crf.stderr**2 + np.nansum([fb_coef[('cld', fbn)].stderr**2 for fbn in fbnams]))

    logging.info("Function completed in %.2f seconds.", time.time() - start_time)
        
    return(fb_cloud, fb_cloud_err)