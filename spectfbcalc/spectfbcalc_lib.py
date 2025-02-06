#!/usr/bin/python
# -*- coding: utf-8 -*-

### Our library!

##### Package imports

import sys
import os
import glob

import numpy as np
import xarray as xr

from climtools import climtools_lib as ctl
from matplotlib import pyplot as plt
import matplotlib.cbook as cbook
from scipy import stats
import pickle
import dask.array as da

######################################################################
### Functions

def mytestfunction():
    print('test!')
    return


###### INPUT/OUTPUT SECTION: load kernels, load data ######

def load_spectral_kernel():

    ## per ste

    return allkers

#Definire una funzione di check che apra i file, controlli i nomi delle variabili, 
# ##e nel caso non siano uniformi agli standard del codice chieda all'utente di cambiarli

#PRENDERE I KERNEL
def load_kernel_ERA5(cart_k:str, cart_out:str):
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
    allkers = dict()
    finam='ERA5_kernel_{}_TOA.nc'
    tips = ['clr', 'cld']
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

    k = allkers[('cld', 't')]
    vlevs = xr.load_dataset( cart_k+'dp_era5.nc')
    vlevs=vlevs.rename({'level': 'player', 'latitude': 'lat', 'longitude': 'lon'})
    cose = 100*vlevs.player
    pickle.dump(vlevs, open(cart_out + 'vlevs_ERA5.p', 'wb')) #save vlevs
    pickle.dump(k, open(cart_out + 'k_ERA5.p', 'wb')) #save k
    pickle.dump(cose, open(cart_out + 'cose_ERA5.p', 'wb'))
    pickle.dump(allkers, open(cart_out + 'allkers_ERA5.p', 'wb'))
    return allkers


def load_kernel_HUANG(cart_k:str, cart_out:str):
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
    finam = 'RRTMG_{}_toa_{}_highR.nc'
    allkers = dict()

    for tip in tips:
        for vna in vnams:
            ker = xr.load_dataset(cart_k+ finam.format(vna, tip))

            allkers[(tip, vna)] = ker.assign_coords(month = np.arange(1, 13))
            if vna in ('ts', 't', 'wv_lw'):
                allkers[(tip, vna)]=allkers[(tip, vna)].lwkernel
            else:
                allkers[(tip, vna)]=allkers[(tip, vna)].swkernel


    vlevs = xr.load_dataset( cart_k + 'dp.nc') 
    k = allkers[('cld', 't')]
    pickle.dump(vlevs, open(cart_out + 'vlevs_HUANG.p', 'wb')) #save vlevs
    pickle.dump(k, open(cart_out + 'k_HUANG.p', 'wb')) #save k
    cose = 100*vlevs.player
    pickle.dump(cose, open(cart_out + 'cose_HUANG.p', 'wb'))
    pickle.dump(allkers, open(cart_out + 'allkers_HUANG.p', 'wb'))
    return allkers


def read_data(config):
    """
    Reads path of files from config.yml, read all vars and put them in a standardized dataset.
    """

    # read config

    # read data

    ds = standardize_names(ds)

    return ds

def read_data_ref(config):
    """
    Same for reference experiment.
    """
    return ds_ref

def standardize_names(ds):
    """
    standardizes variable and coordinate names
    """
    return ds


######################################################################################
#### Aux functions

def ref_clim(ds_ref):

    return ds_clim


def climatology(filin_pi:str, ker:str, cart_k:str, cart_out:str, allvars:str, use_climatology=True, time_chunk=12):
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
            
    ker : str
        Specifies which kernel to use: `'HUANG'` or `'ERA5'`. The function will load or preprocess the corresponding kernel.

    cart_k : str
        Path to the directory containing kernel dataset files.
    
    cart_out : str
        Path to the directory where processed kernel files (e.g., 'kHUANG.p', 'kERA5.p') are stored or will be saved.

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
    # Check if the kernel file exists, if not, call the ker() function
    if ker=='HUANG':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_HUANG(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out

    if ker=='ERA5':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_ERA5(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out

    pimean = dict()
    if allvars=='alb':
        allvars='rsus rsds'.split()
        if use_climatology==True:
            for vnam in allvars:
                filist = glob.glob(filin_pi.format(vnam))
                filist.sort()

                var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
                var_mean = var.groupby('time.month').mean()
                var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)
                pimean[vnam] = var_mean[vnam].compute()

            piok = pimean[('rsus')]/pimean[('rsds')]

        else:
            piok=dict()
            for vnam in allvars:
                filist = glob.glob(filin_pi.format(vnam))
                filist.sort()
                pivar = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
                pivar = pivar.sel(time = slice('2540-01-01', '2689-12-31')) #MODIFICA
                piok[vnam] = ctl.regrid_dataset(pivar[vnam], k.lat, k.lon)
     
            pivar['alb']=piok[('rsus')]/piok[('rsds')]
            piok = ctl.regrid_dataset(pivar['alb'], k.lat, k.lon)
            piok=ctl.running_mean(piok, 252)
     
    else:
        if use_climatology==True:
    
            filist = glob.glob(filin_pi.format(allvars))
            filist.sort()
            var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
            var_mean = var.groupby('time.month').mean()
            var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)
            piok = var_mean[allvars].compute()

        
        else:
            filist = glob.glob(filin_pi.format(allvars))
            filist.sort()
            pivar = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
            pivar = pivar.sel(time = slice('2540-01-01', '2689-12-31')) #MODIFICA
            piok = ctl.regrid_dataset(pivar[allvars], k.lat, k.lon)
            piok=ctl.running_mean(piok, 252)
        
    return(piok)
     

##calcolare tropopausa (Reichler 2003) 

def mask_atm(filin_4c:str, time_chunk=12):
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

    filist = glob.glob(filin_4c.format('ta'))
    filist.sort()
    temp = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)

    A=(temp.plev/temp['ta'])*(9.81/1005)
    laps1=(temp['ta'].diff(dim='plev'))*A  #derivata sulla verticale = laspe-rate

    laps1=laps1.where(laps1<=-2)
    mask = laps1/laps1
    return mask

### Mask for surf pressure

def mask_pres(pressure_directory:str, cart_out:str, cart_k:str):
    """
    Computes a "width mask" for atmospheric pressure levels based on surface pressure and kernel data.

    The function determines which pressure levels are above or below the surface pressure (`ps`) 
    and generates a mask that includes NaN values for levels below the surface pressure and 
    interpolated values near the surface. It supports kernels from HUANG and ERA5 datasets.

    Parameters:
    -----------
    cart_k : str
        Path to the directory containing kernel dataset files.
    
    cart_out : str
        Path to the directory where processed kernel files (e.g., 'kHUANG.p', 'kERA5.p', 'vlevsHUANG.p', 'vlevsERA5.p') 
        are stored or will be saved.

    pressure_directory : str
        Path to the directory containing surface pressure (`ps`) datasets in NetCDF format. 
        Example: `/path/to/ps/files/ps_Amon_*.nc`.

    Returns:
    --------
    wid_mask : xarray.DataArray
        A mask indicating the vertical pressure distribution for each grid point. 
        Dimensions depend on the kernel data and regridded surface pressure:
        - For HUANG: [`player`, `lat`, `lon`]
        - For ERA5: [`player`, `month`, `lat`, `lon`]

    Notes:
    ------
    - Kernels (`k`) and vertical levels (`vlevs`) are loaded from `cart_out`. If missing, they are computed 
      using `load_kernel_HUANG` or `load_kernel_ERA5`.
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
    
    # Check if the kernel file exists, if not, call the ker() function

    k_file_path = os.path.join(cart_out, 'k_HUANG.p')
    if not os.path.exists(k_file_path):
        allkers=dict()
        allkers= load_kernel_HUANG(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
    k=pickle.load(open(cart_out + 'k_HUANG.p', 'rb'))#prendi da cart_out
    vlevs=pickle.load(open(cart_out + 'vlevs_HUANG.p', 'rb'))

    ps = xr.open_mfdataset(pressure_directory)
    psclim = ps.groupby('time.month').mean()
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

#calcolo ts_anom e gtas e plank surf


def fb_planck_surf_wrapper(config):

    ds = read_data()
    ds_ref = read_data_ref()
    ds_clim = ref_clim()

    kernels = load_kernel()

    fb_planck_surf_core(ds, ds_clim)

    return


def fb_planck_surf_core(ds, ds_clim, kernels):
    return


def fb_planck_surf(filin_4c:str, filin_pi:str, cart_out:str, ker:str, cart_k:str, use_climatology=True, time_chunk=12):
    """
    Computes the surface Planck feedback using temperature anomalies and precomputed kernels.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files with a placeholder for the variable name, 
        such as 'ts'. Use `{}` for placeholders to enable string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) temperature files, required for computing anomalies.

    cart_out : str
        Path where output files will be saved.

    ker : str
        Specifies the kernel dataset to use: `'HUANG'` or `'ERA5'`.

    cart_k : str
        Path to the directory containing the kernel dataset files.

    use_climatology : bool, optional (default=True)
        If True, uses mean climatology from the precomputed PI files. 
        If False, computes a running mean from PI files.

    time_chunk : int, optional (default=12)
        Chunk size for loading data with xarray to optimize memory usage.

    Returns:
    --------
    feedbacks : dict
        A dictionary containing the computed global annual mean surface Planck feedbacks for clear-sky 
        (`clr`) and all-sky (`cld`) conditions. Keys of the dictionary:
        - `('clr', 'planck-surf')`: Clear-sky surface Planck feedback.
        - `('cld', 'planck-surf')`: All-sky surface Planck feedback.

    Notes:
    ------
    - Kernels (`k`) are loaded from `cart_out`. If missing, they are computed using 
      `load_kernel_HUANG` or `load_kernel_ERA5`.
    - Surface temperature anomalies (`ts_anom`) are computed relative to the PI climatology.
    - Global annual mean temperature anomalies (`gtas`) are saved.
    - Computed feedbacks are saved as NetCDF files for each kernel type and sky condition.

    Outputs Saved to `cart_out`:
    ----------------------------
    - `ts_anom{suffix}.nc`: Temperature anomalies relative to the PI climatology.
    - `gtas{suffix}.nc`: Global temperature anomaly series, grouped by year.
    - `dRt_planck-surf_global_{tip}{suffix}.nc`: Global surface Planck feedback for clear (`clr`) and 
      all (`cld`) sky conditions.

      Here, `{suffix}` = `"_climatology-{ker}kernels"` or `"_21yearmean-{ker}kernels"`, based on the 
      `use_climatology` flag and kernel type (`ker`).

    """   
    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    if ker=='HUANG':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_HUANG(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out
        

    if ker=='ERA5':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_ERA5(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out

    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    filist = glob.glob(filin_4c.format('ts'))
    filist.sort()
    var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
    var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
    var = ctl.regrid_dataset(var['ts'], k.lat, k.lon)
    allvars='ts'
    piok=climatology(filin_pi, ker, cart_k, cart_out, allvars, use_climatology, time_chunk)

    if use_climatology == False:
        piok=piok.drop('time')
        piok['time'] = var['time']
        piok = piok.chunk(var.chunks)
        anoms = var - piok
    else:
        anoms = var.groupby('time.month') - piok
    
    ts_anom = anoms.compute()
    ts_anom.to_netcdf(cart_out+ "ts_anom"+cos+"-"+ker+"kernels.nc", format="NETCDF4")
    gtas = ctl.global_mean(anoms).groupby('time.year').mean('time')
    gtas.to_netcdf(cart_out+ "gtas"+cos+"-"+ker+"kernels.nc", format="NETCDF4")
 
    for tip in ['clr', 'cld']:
        kernel = allkers[(tip, 'ts')]

        dRt = xr.apply_ufunc(lambda x, ker: x*ker, anoms.groupby('time.month'), kernel, dask = 'allowed').groupby('time.year').mean('time')
        dRt_glob = ctl.global_mean(dRt)
        planck= dRt_glob.compute()
        feedbacks[(tip, 'planck-surf')]=planck
        planck.to_netcdf(cart_out+ "dRt_planck-surf_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        
    return(feedbacks)


#CALCOLO PLANK-ATMO E LAPSE RATE CON TROPOPAUSA VARIABILE (DA CONTROLLARE)

def fb_plank_atm_lr(filin_4c:str, filin_pi:str, cart_out:str, ker:str, cart_k:str, pressure_directory, use_climatology = True, time_chunk=12):
    """
    Computes atmospheric feedbacks, including Planck and lapse-rate feedbacks, using atmospheric temperature anomalies
    and precomputed kernels.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files with a placeholder for the variable name, such as `'ta'`.
        Use `{}` for placeholders to enable string formatting.

    filin_pi : str
        Path template for preindustrial (PI) temperature files used to compute anomalies.

    cart_out : str
        Directory where output files will be saved.
    
    ker : str
        Kernel dataset to use, either `'HUANG'` or `'ERA5'`.
    
    cart_k : str
        Directory containing the kernel dataset files.

    pressure_directory : str, optional
        Directory containing pressure-level datasets for mask computation.

    use_climatology : bool, optional (default=True)
        If True, use mean climatology from the precomputed PI files. 
        If False, compute running mean from PI files.

    time_chunk : int, optional (default=12)
        Chunk size for loading data with xarray, optimizing memory usage.


    Returns:
    --------
    feedbacks : dict
        A dictionary containing the computed global annual mean atmospheric feedbacks for clear-sky (`clr`) and all-sky (`cld`) conditions.
        Keys of the dictionary:
        - `('clr', 'planck-atmo')`: Clear-sky atmospheric Planck feedback.
        - `('cld', 'planck-atmo')`: All-sky atmospheric Planck feedback.
        - `('clr', 'lapse-rate')`: Clear-sky lapse-rate feedback.
        - `('cld', 'lapse-rate')`: All-sky lapse-rate feedback.

    Notes:
    ------
    - Kernels (`k`) are loaded from `cart_out`. If missing, they are computed using `load_kernel_HUANG` or `load_kernel_ERA5`.
    - Atmospheric temperature anomalies (`anoms_ok`) are computed relative to the PI climatology.
    - Pressure-level masks are applied to ensure accurate feedback computation.
    - Computed feedbacks are saved as NetCDF files for each kernel type and sky condition.

    Outputs Saved to `cart_out`:
    ----------------------------
    - `ta_abs_pi{suffix}.nc`: Preindustrial absolute temperature interpolated to pressure levels.
    - `dRt_planck-atmo_global_{tip}{suffix}.nc`: Global atmospheric Planck feedback for clear (`clr`) and all (`cld`) sky conditions.
    - `dRt_lapse-rate_global_{tip}{suffix}.nc`: Global lapse-rate feedback for clear (`clr`) and all (`cld`) sky conditions.

      Here, `{suffix}` = `"_climatology-{ker}kernels"` or `"_21yearmean-{ker}kernels"`, based on the `use_climatology` flag and kernel type (`ker`).

    """
   
    allkers=dict()
    feedbacks=dict()

    if ker=='HUANG':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_HUANG(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out
        cose=pickle.load(open(cart_out + 'cose_'+ker+'.p', 'rb'))
        wid_mask=mask_pres(pressure_directory, cart_out, cart_k)

    if ker=='ERA5':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_ERA5(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out
        cose=pickle.load(open(cart_out + 'cose_'+ker+'.p', 'rb'))
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
   
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    filist = glob.glob(filin_4c.format('ta'))
    filist.sort()
    var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
    var = var['ta']
    var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    allvars='ta'
    piok=climatology(filin_pi, ker, cart_k, cart_out, allvars, use_climatology, time_chunk)

    if use_climatology==False:
        piok=piok.drop('time')
        piok['time'] = var['time']
        anoms_ok = var - piok
    else:
        anoms_ok=var.groupby('time.month') - piok

    ta_abs_pi = piok.interp(plev = cose)
    ta_abs_pi.to_netcdf(cart_out+ "ta_abs_pi"+cos+"-"+ker+"kernels.nc", format="NETCDF4")
    mask=mask_atm(filin_4c, time_chunk)
    anoms_ok = (anoms_ok*mask).interp(plev = cose)
    ts_anom=xr.open_dataarray(cart_out+"ts_anom"+cos+"-"+ker+"kernels.nc", chunks = {'time': time_chunk}, use_cftime=True) 

    for tip in ['clr','cld']:
        kernel = allkers[(tip, 't')]
        anoms_lr = (anoms_ok - ts_anom)  
        anoms_unif = (anoms_ok - anoms_lr)
        if ker=='HUANG':
            dRt_unif = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_unif.groupby('time.month'), kernel, wid_mask/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')
            dRt_lr = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_lr.groupby('time.month'), kernel, wid_mask/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')

        if ker=='ERA5':
            dRt_unif = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_unif.groupby('time.month'), kernel, vlevs.dp/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')
            dRt_lr = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_lr.groupby('time.month'), kernel, vlevs.dp/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')


        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        feedbacks[(tip,'planck-atmo')]=feedbacks_atmo
        feedbacks[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        
    return(feedbacks)

#CONTO ALBEDO

def fb_albedo(filin_4c:str, filin_pi:str, cart_out:str, ker:str, cart_k:str, use_climatology=True, time_chunk=12):
    """
    Computes albedo feedbacks using surface albedo anomalies and precomputed kernel datasets.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files with placeholders for variable names, e.g., `'rsus'` and `'rsds'`.
        Use `{}` as a placeholder for variable names in the string.

    filin_pi : str
        Path template for preindustrial (PI) files used to compute surface albedo climatology.

    cart_out : str
        Directory where output files will be saved.
    
    ker : str
        Kernel dataset to use, either `'HUANG'` or `'ERA5'`.
    
    cart_k : str
        Directory containing the kernel dataset files.

    use_climatology : bool, optional (default=True)
        If True, uses mean climatology from the precomputed PI files.
        If False, computes running mean from the PI files.

    time_chunk : int, optional (default=12)
        Chunk size for loading data with xarray to optimize memory usage.

    Returns:
    --------
    feedbacks : dict
        A dictionary containing the computed global annual mean albedo feedbacks for clear-sky (`clr`) and all-sky (`cld`) conditions.
        Keys of the dictionary:
        - `('clr', 'albedo')`: Clear-sky albedo feedback.
        - `('cld', 'albedo')`: All-sky albedo feedback.

    Notes:
    ------
    - Surface albedo is computed as `rsus / rsds` (upwelling solar radiation divided by downwelling solar radiation).
    - Anomalies are computed relative to the PI climatology (`piok`).
    - Kernels (`k`) are loaded from `cart_out`. If missing, they are computed using `load_kernel_HUANG` or `load_kernel_ERA5`.
    - Negative or zero albedo values are masked to ensure validity.

    Outputs Saved to `cart_out`:
    ----------------------------
    - `dRt_albedo_global_{tip}{suffix}.nc`: Global annual mean albedo feedback for clear (`clr`) and all (`cld`) sky conditions.
    
      Here, `{suffix}` = `"_climatology-{ker}kernels"` or `"_21yearmean-{ker}kernels"`, based on the `use_climatology` flag and kernel type (`ker`).

    """
   
    allkers=dict()
    # Check if the kernel file exists, if not, call the ker() function
    if ker=='HUANG':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_HUANG(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out

    if ker=='ERA5':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_ERA5(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out

    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"
  
    feedbacks=dict()
    filist_1 = glob.glob(filin_4c.format('rsus'))
    filist_1.sort()
    var_rsus = xr.open_mfdataset(filist_1, chunks = {'time': time_chunk}, use_cftime=True)['rsus']
    filist_2 = glob.glob(filin_4c.format('rsds'))
    filist_2.sort()
    var_rsds = xr.open_mfdataset(filist_2, chunks = {'time': time_chunk}, use_cftime=True)['rsds']
    var = var_rsus/var_rsds
    var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    allvars='alb'
    piok=climatology(filin_pi, ker, cart_k, cart_out, allvars, use_climatology, time_chunk)

    if use_climatology==False:
        piok=piok.drop('time')
        piok['time'] = var['time']

    # Removing inf and nan from alb
    piok = piok.where(piok > 0., 0.)
    var = var.where(var > 0., 0.)
    if use_climatology==False:
        anoms =  var - piok
    else:
        anoms =  var.groupby('time.month') - piok

    for tip in [ 'clr','cld']:
        kernel = allkers[(tip, 'alb')]

        dRt = xr.apply_ufunc(lambda x, ker: x*ker, anoms.groupby('time.month'), kernel, dask = 'allowed').groupby('time.year').mean('time')
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        feedbacks[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        
    return(feedbacks)

##CALCOLO W-V
def fb_wv(filin_4c:str, filin_pi:str, cart_out:str, ker:str, cart_k:str, pressure_directory:str,  use_climatology=True, time_chunk=12):
    
    """
    Computes the water vapor feedback using specific humidity (hus) anomalies, precomputed kernels,
    and vertical integration.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files containing climate data, with a placeholder for the variable
        name (e.g., 'hus'). Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for preindustrial (PI) climate files, required for computing anomalies.

    cart_out : str
        Output directory where precomputed files (e.g., `k.p`, `cose.p`, `vlevs.p`) and results are stored.
    
    cart_k : str
        Directory containing the kernel dataset files.
        
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
    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    if ker=='HUANG':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_HUANG(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out
        cose=pickle.load(open(cart_out + 'cose_'+ker+'.p', 'rb'))

    if ker=='ERA5':
        k_file_path = os.path.join(cart_out, 'k_'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_ERA5(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        else:
            allkers=pickle.load(open(cart_out + 'allkers_'+ker+'.p', 'rb'))
        k=pickle.load(open(cart_out + 'k_'+ker+'.p', 'rb'))#prendi da cart_out
        cose=pickle.load(open(cart_out + 'cose_'+ker+'.p', 'rb'))
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
   
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    ta_abs_pi=xr.open_dataarray(cart_out+"ta_abs_pi"+cos+"-"+ker+"kernels.nc",  chunks = {'time': time_chunk},  use_cftime=True)
    mask=mask_atm(filin_4c)
    wid_mask=mask_pres(pressure_directory, cart_out, cart_k)
    filist = glob.glob(filin_4c.format('hus')) 
    filist.sort()
    var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
    var = var['hus']
    var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    allvars='hus'
    piok=climatology(filin_pi, ker, cart_k, cart_out, allvars, use_climatology, time_chunk)
    Rv = 487.5 # gas constant of water vapor
    Lv = 2.5e+06 # latent heat of water vapor

    if use_climatology==False:
        piok=piok.drop('time')
        piok['time'] = var['time']
 

    var_int = (var*mask).interp(plev = cose)
    piok_int = piok.interp(plev = cose)


    
    if ker=='HUANG':
        if use_climatology==True:
            anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int.groupby('time.month'), piok_int , dask = 'allowed')
            coso3= xr.apply_ufunc(lambda x, ta: x*ta, anoms_ok3.groupby('time.month'), dlnws(ta_abs_pi), dask = 'allowed') #(using dlnws)
       
        else:
            anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int, piok_int , dask = 'allowed')
            coso3= xr.apply_ufunc(lambda x, ta: x*ta, anoms_ok3, dlnws(ta_abs_pi), dask = 'allowed') #(using dlnws)
    
    if ker=='ERA5': #AGGIUSTA LA COSA GROUPBY PER CLIMATOLOGY
        if use_climatology==False:
            anoms= var_int-piok_int
            coso = (anoms/piok_int) * (ta_abs_pi**2) * Rv/Lv
        else:
            anoms= var_int.groupby('time.month')-piok_int
            coso = (anoms.groupby('time.month')/piok_int).groupby('time.month') * (ta_abs_pi**2) * Rv/Lv #dlnws(ta_abs_pi) #va bene anche dlnws, vedi tu quello che vuoi usare
    
    for tip in ['clr','cld']:
        kernel_lw = allkers[(tip, 'wv_lw')]
        kernel_sw = allkers[(tip, 'wv_sw')]
        kernel = kernel_lw + kernel_sw
        
        if ker=='HUANG':
            dRt = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, coso3.groupby('time.month'), kernel, wid_mask/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')

        if ker=='ERA5':
            dRt = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, coso.groupby('time.month'), kernel, vlevs.dp / 100 , dask = 'allowed')).sum('player').groupby('time.year').mean('time')
        
        dRt_glob = ctl.global_mean(dRt)
        wv= dRt_glob.compute()
        feedbacks[(tip, 'water-vapor')]=wv
        wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" +tip+cos +"-"+ker+"kernels.nc", format="NETCDF4")
        
    return(feedbacks)

##CALCOLO EFFETTIVO DEI FEEDBACK
def calc_fb(filin_4c:str, filin_pi:str, cart_out:str, ker:str, cart_k:str, pressure_directory:str, use_climatology=True, time_chunk=12):
    """
    Computes climate feedback coefficients by combining feedback components (Planck, albedo, water vapor, etc.)
    with global temperature anomalies and performs a linear regression analysis.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files containing climate data, with a placeholder for the variable
        name (e.g., 'ta'). Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for preindustrial (PI) climate files, required for computing anomalies.

    cart_out : str
        Output directory where intermediate feedback data and results are stored.

     ker : str
        Kernel dataset to use, either `'HUANG'` or `'ERA5'`.

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
    if use_climatology==True:
        cos="_climatology"
        print(cos)
    else:
        cos="_21yearmean"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        fb_planck_surf(filin_4c, filin_pi, cart_out,ker,  cart_k, use_climatology, time_chunk)
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        fb_plank_atm_lr(filin_4c, filin_pi, cart_out, ker, cart_k, pressure_directory, use_climatology, time_chunk)
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        fb_albedo(filin_4c, filin_pi, cart_out, ker, cart_k, use_climatology, time_chunk)
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        fb_wv(filin_4c, filin_pi, cart_out, ker, cart_k, pressure_directory, use_climatology, time_chunk)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()
    gtas=xr.open_dataarray(cart_out+"gtas"+cos+"-"+ker+"kernels.nc",  use_cftime=True)
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()
    print('calcolo feedback')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ cos+"-"+ker+"kernels.nc",  use_cftime=True)
            feedback=feedbacks.groupby((feedbacks.year-1) // 10 * 10).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res
    
    return(fb_coef)

# ###CLOUD FEEDBACK shell 2008
def fb_cloud(filin_4c:str, filin_4c1:str, filin_pi:str, ker:str, cart_out:str, cart_k:str, pressure_directory, use_climatology=True, time_chunk=12):
    """
    Computes the cloud feedback coefficient based on the Shell (2008) approach,
    using radiative fluxes, clear-sky fluxes, and feedback components. 

    Parameters:
    -----------
    filin_4c : str
        Path template for NetCDF files containing radiative flux variables (e.g., `rlut`, `rsut`).
        The template should include `{}` for formatting with variable names.

    filin_4c1 : str
        Path template for NetCDF files containing clear-sky radiative flux variables (`rsutcs`, `rlutcs`).
        The template should include `{}` for formatting with variable names.
    
    filin_pi : str
        Path template for preindustrial (PI) climate files, required for computing anomalies.


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
    # Check if the kernel file exists, if not, call the ker() function
    if ker=='HUANG':
        k_file_path = os.path.join(cart_out, 'k'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_HUANG(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        k=pickle.load(open(cart_out + 'k'+ker+'.p', 'rb'))#prendi da cart_out

    if ker=='ERA5':
        k_file_path = os.path.join(cart_out, 'k'+ker+'.p')
        if not os.path.exists(k_file_path):
            allkers=dict()
            allkers= load_kernel_ERA5(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
        k=pickle.load(open(cart_out + 'k'+ker+'.p', 'rb'))#prendi da cart_out

    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    pimean=dict()
    fb_coef=dict()
    fb_coef=calc_fb(filin_4c, filin_pi, cart_out, cart_k, pressure_directory, use_climatology, time_chunk)
    
    filist = glob.glob(filin_4c.format('rlut'))
    filist.sort()
    rlut = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)['rlut']
    rlut = rlut.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
 
    filist = glob.glob(filin_4c.format('rsut'))
    filist.sort()
    rsut = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)['rsut']
    rsut = rsut.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA

    filist = glob.glob(filin_4c1.format('rsutcs')) 
    filist.sort()
    rsutcs = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)['rsutcs']
    rsutcs= ctl.regrid_dataset(rsutcs, k.lat, k.lon)
    rsutcs = rsutcs.sel(time = slice('1850-01-01', '1999-12-31'))#MODIFICA

    filist = glob.glob(filin_4c1.format('rlutcs'))
    filist.sort()
    rlutcs = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)['rlutcs']
    rlutcs = ctl.regrid_dataset(rlutcs, k.lat, k.lon)
    rlutcs = rlutcs.sel(time = slice('1850-01-01', '1999-12-31'))#MODIFICA


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

    gtas=xr.open_dataarray(cart_out+"gtas.nc",  use_cftime=True)
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()
    res_N = stats.linregress(gtas, N_glob)
    res_N0 = stats.linregress(gtas, N0_glob)
    res_crf = stats.linregress(gtas, crf_glob)

    allvars= 'rlutcs rsutcs rlut rsut'.split()
    for vnams in allvars:
            filist = glob.glob(filin_pi.format(vnams))
            filist.sort()
            var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
            var_mean = var.mean('time')
            var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)
            pimean[vnams] = var_mean[vnams].compute()


    F0 = res_N0.intercept + pimean[('rlutcs')] + pimean[('rsutcs')] 
    F = res_N.intercept + pimean[('rlut')] + pimean[('rsut')]
    F0.compute()
    F.compute()

    F_glob = ctl.global_mean(F)
    F0_glob = ctl.global_mean(F0)
    F_glob = F_glob.compute()
    F0_glob = F0_glob.compute()

    fb_cloud = -res_crf.slope + np.nansum([fb_coef[( 'clr', fbn)].slope - fb_coef[('cld', fbn)].slope for fbn in fbnams]) #letto in Caldwell2016

    fb_cloud_err = np.sqrt(res_crf.stderr**2 + np.nansum([fb_coef[('cld', fbn)].stderr**2 for fbn in fbnams]))
    return(fb_cloud, fb_cloud_err)


###### Spectral kernels ######



###### General functions ######



###### Plotting ######

