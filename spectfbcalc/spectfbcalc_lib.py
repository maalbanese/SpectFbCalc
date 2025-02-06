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

def load_kernel(cart_k:str, cart_out:str):
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


    vnams = ['t', 'ts', 'wv_lw', 'wv_sw', 'alb']
    tips = ['clr', 'cld']

    allkers = dict()

    for tip in tips:
        for vna in vnams:
            ker = xr.load_dataset(cart_k.format(vna, tip))

            allkers[(tip, vna)] = ker.assign_coords(month = np.arange(1, 13))

    vlevs = xr.load_dataset( '/data-hobbes/fabiano/radiative_kernels/Huang/toa/dp.nc') #aggiusta
    k = allkers[('cld', 't')].lwkernel
    pickle.dump(vlevs, open(cart_out + 'vlevs.p', 'wb')) #save vlevs
    pickle.dump(k, open(cart_out + 'k.p', 'wb')) #save k
    cose = 100*vlevs.player
    pickle.dump(cose, open(cart_out + 'cose.p', 'wb'))
    pickle.dump(allkers, open(cart_out + 'allkers.p', 'wb'))
    return allkers


def read_data(config):
    return ds

def read_data_ref(config):
    return ds_ref

def standardize_names():
    return


######################################################################################
#### Aux functions

def ref_clim(ds_ref):

    return ds_clim


def climatology(filin_pi:str, cart_k:str, cart_out:str, allvars:str, use_climatology=True, time_chunk=12):
    """
    Computes climatological means or running means for specified variables, processes data using kernels, and 
    saves the results to netCDF files.

    Parameters:
    -----------
    filin_pi : str
        Path template to input data files for the pre-industrial control run. 
        Should contain placeholders for variable names formatted as `{}`.
        
    cart_k : str
        Path template to the kernel dataset files. Placeholders should be formatted as `{}`.

    cart_out : str
        Path to save output files such as kernel objects and processed datasets.

    allvars : str
        Variable name(s) to process. For the `alb` case, it automatically processes the shortwave 
        components `rsus` and `rsds`.

    use_climatology : bool, optional, default=True
        Determines the processing type:
        - `True`: Computes the climatological mean over the entire time period.
        - `False`: Computes a 21-year running mean over a specific time slice.

    time_chunk : int, optional, default=12
        Chunk size for processing data with xarray for improved performance.

    Returns:
    --------
    None
        Outputs are saved as netCDF files in the specified `cart_out` directory.

    Outputs:
    --------
    The function saves processed datasets as netCDF files, including:
      - Climatological mean or running mean for each variable.
      - Regridded datasets aligned with the kernel latitude and longitude.
      - Albedo calculated from `rsus` and `rsds` if `allvars` is `alb`.
    """  
    # Check if the kernel file exists, if not, call the ker() function
    pimean = dict()
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        allkers=dict()
        print("Kernel file not found. Running ker() to generate it.")
        allkers= load_kernel(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out

    if allvars=='alb':
        allvars='rsus rsds'.split()
        if use_climatology==True:
            for vnam in allvars:
                filist = glob.glob(filin_pi.format(vnam))
                filist.sort()

                var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
                var_mean = var.mean('time')
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
            var_mean = var.mean('time')
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
    
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        allkers=dict()
        print("Kernel file not found. Running ker() to generate it.")
        allkers= load_kernel(cart_k,cart_out)  # Ensure that ker() is properly defined elsewhere in the code
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
    vlevs=pickle.load(open(cart_out + 'vlevs.p', 'rb'))#prendi da cart_out
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

#calcolo ts_anom e gtas e plank surf


def fb_planck_surf_from_file(config):

    ds = read_data()
    ds_ref = read_data_ref()
    ds_clim = ref_clim()

    kernels = load_kernel()

    fb_planck_surf(ds, ds_clim)

    return


def fb_planck_surf_core(ds, ds_clim, kernels):
    return


def fb_planck_surf(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, use_climatology=True, time_chunk=12):
    """
    Computes the surface Planck feedback using temperature anomalies and precomputed kernels.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'ts').
        Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) temperature files, required to compute anomalies.

    cart_out : str
        Path template to save the outputs. 

    cart_k : str
        Path template to the kernel dataset files, used by the `ker()` function if kernel data is missing.

    use_climatology : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_ts.nc`).
        If False, uses running mean data from precomputed PI files (`piok_ts_21y.nc`).

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
    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers= load_kernel(cart_k, cart_out)   # Ensure that ker() is properly defined elsewhere in the code
    else:
       allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out

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
    piok=climatology(filin_pi, cart_k, cart_out, allvars, use_climatology, time_chunk)

    if use_climatology == False:
        piok=piok.drop('time')
        piok['time'] = var['time']
        piok = piok.chunk(var.chunks)
    
     
    anoms = var - piok
    ts_anom = anoms.compute()
    ts_anom.to_netcdf(cart_out+ "ts_anom"+cos+".nc", format="NETCDF4")
    gtas = ctl.global_mean(anoms).groupby('time.year').mean('time')
    gtas.to_netcdf(cart_out+ "gtas"+cos+".nc", format="NETCDF4")
 
    for tip in ['clr', 'cld']:
        kernel = allkers[(tip, 'ts')].lwkernel

        dRt = xr.apply_ufunc(lambda x, ker: x*ker, anoms.groupby('time.month'), kernel, dask = 'allowed').groupby('time.year').mean('time')
        dRt_glob = ctl.global_mean(dRt)
        planck= dRt_glob.compute()
        feedbacks[(tip, 'planck-surf')]=planck
        planck.to_netcdf(cart_out+ "dRt_planck-surf_global_" +tip +cos+".nc", format="NETCDF4")
        
    return(feedbacks)


#CALCOLO PLANK-ATMO E LAPSE RATE CON TROPOPAUSA VARIABILE (DA CONTROLLARE)

def fb_plank_atm_lr(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, pressure_directory, use_climatology = True, time_chunk=12):
    """
    Computes atmospheric Planck and lapse-rate feedbacks using temperature anomalies, kernels, and masking.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'ta').
        Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) temperature files, required to compute anomalies.

    cart_out : str
        Output directory where precomputed files (`k.p`, `cose.p`, `vlevs.p`) and results are stored.

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
   
    allkers=dict()
    feedbacks=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
    else:
       allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
    cose=pickle.load(open(cart_out + 'cose.p', 'rb'))
   
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
    piok=climatology(filin_pi, cart_k, cart_out, allvars, use_climatology, time_chunk)

    if use_climatology==False:
        piok=piok.drop('time')
        piok['time'] = var['time']
    anoms_ok = var - piok

    ta_abs_pi = piok.interp(plev = cose)
    ta_abs_pi.to_netcdf(cart_out+ "ta_abs_pi"+cos+".nc", format="NETCDF4")
    mask=mask_atm(filin_4c, time_chunk)
    wid_mask=mask_pres(pressure_directory, cart_out, cart_k)
    anoms_ok = (anoms_ok*mask).interp(plev = cose)
    ts_anom=xr.open_dataarray(cart_out+"ts_anom"+cos+".nc", chunks = {'time': time_chunk}, use_cftime=True) 

    for tip in ['clr','cld']:
        kernel = allkers[(tip, 't')].lwkernel
        anoms_lr = (anoms_ok - ts_anom)  
        anoms_unif = (anoms_ok - anoms_lr)
    
        dRt_unif = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_unif.groupby('time.month'), kernel, wid_mask/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')
        dRt_lr = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, anoms_lr.groupby('time.month'), kernel, wid_mask/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')

        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        feedbacks[(tip,'planck-atmo')]=feedbacks_atmo
        feedbacks[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip +cos+".nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  +cos+".nc", format="NETCDF4")
        
    return(feedbacks)

#CONTO ALBEDO

def fb_albedo(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, use_climatology=True, time_chunk=12):
    """
    Computes the albedo feedback using surface albedo anomalies and precomputed kernels.

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'rsus', 'rsds').
        Placeholders should be formatted as `{}` to allow string formatting.

    filin_pi : str
        Path template for the preindustrial (PI) albedo files, required to compute anomalies.

    cart_out : str
        Output directory where precomputed files (`k.p`) and results are stored.

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
   
    allkers=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
    else:
        allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
    
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
    piok=climatology(filin_pi, cart_k, cart_out, allvars, use_climatology, time_chunk)

    if use_climatology==False:
        piok=piok.drop('time')
        piok['time'] = var['time']

    # Removing inf and nan from alb
    piok = piok.where(piok > 0., 0.)
    var = var.where(var > 0., 0.)
    anoms =  var - piok

    for tip in [ 'clr','cld']:
        kernel = allkers[(tip, 'alb')].swkernel

        dRt = xr.apply_ufunc(lambda x, ker: x*ker, anoms.groupby('time.month'), kernel, dask = 'allowed').groupby('time.year').mean('time')
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        feedbacks[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip +cos+".nc", format="NETCDF4")
        
    return(feedbacks)

##CALCOLO W-V
def fb_wv(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, pressure_directory:str,  use_climatology=True, time_chunk=12):
    
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
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
    else:
        allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
    cose=pickle.load(open(cart_out + 'cose.p', 'rb'))
    
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    ta_abs_pi=xr.open_dataarray(cart_out+"ta_abs_pi"+cos+".nc",  chunks = {'time': time_chunk},  use_cftime=True)
    mask=mask_atm(filin_4c)
    wid_mask=mask_pres(pressure_directory, cart_out, cart_k)
    filist = glob.glob(filin_4c.format('hus')) 
    filist.sort()
    var = xr.open_mfdataset(filist, chunks = {'time': time_chunk}, use_cftime=True)
    var = var['hus']
    var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
    var = ctl.regrid_dataset(var, k.lat, k.lon)
    allvars='hus'
    piok=climatology(filin_pi, cart_k, cart_out, allvars, use_climatology, time_chunk)

    if use_climatology==False:
        piok=piok.drop('time')
        piok['time'] = var['time']
 

    var_int = (var*mask).interp(plev = cose)
    piok_int = piok.interp(plev = cose)
    anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int, piok_int , dask = 'allowed')
    coso3= xr.apply_ufunc(lambda x, ta: x*ta, anoms_ok3, dlnws(ta_abs_pi), dask = 'allowed') #(using dlnws)
    for tip in ['clr','cld']:
        kernel_lw = allkers[(tip, 'wv_lw')].lwkernel
        kernel_sw = allkers[(tip, 'wv_sw')].swkernel
        kernel = kernel_lw + kernel_sw
        dRt = (xr.apply_ufunc(lambda x, ker, wid: x*ker*wid, coso3.groupby('time.month'), kernel, wid_mask/100., dask = 'allowed')).sum('player').groupby('time.year').mean('time')
        dRt_glob = ctl.global_mean(dRt)
        wv= dRt_glob.compute()
        feedbacks[(tip, 'water-vapor')]=wv
        wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" +tip+cos +".nc", format="NETCDF4")
        
    return(feedbacks)

##CALCOLO EFFETTIVO DEI FEEDBACK
def calc_fb(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, pressure_directory:str, use_climatology=True, time_chunk=12):
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
    else:
        cos="_21yearmean"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_planck_surf(filin_4c, filin_pi, cart_out, cart_k, use_climatology, time_chunk)
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_plank_atm_lr(filin_4c, filin_pi, cart_out, cart_k, pressure_directory, use_climatology, time_chunk)
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_albedo(filin_4c, filin_pi, cart_out, cart_k, use_climatology, time_chunk)
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+".nc")
    if not os.path.exists(path):
        fb_wv(filin_4c, filin_pi, cart_out, cart_k, pressure_directory, use_climatology, time_chunk)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()
    gtas=xr.open_dataarray(cart_out+"gtas"+cos+".nc",  use_cftime=True)
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()
    print('calcolo feedback')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ cos+".nc",  use_cftime=True)
            feedback=feedbacks.groupby((feedbacks.year-1) // 10 * 10).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res
    
    return(fb_coef)

# ###CLOUD FEEDBACK shell 2008
def fb_cloud(filin_4c:str, filin_4c1:str, filin_pi:str, cart_out:str, cart_k:str, pressure_directory, use_climatology=True, time_chunk=12):
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
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=load_kernel(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code   
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out

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

