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

    
    vlevs = xr.load_dataset( cart_k+'dp_era5.nc')
    vlevs=vlevs.rename({'level': 'player', 'latitude': 'lat', 'longitude': 'lon'})
    cose = 100*vlevs.player
    pickle.dump(vlevs, open(cart_out + 'vlevs_ERA5.p', 'wb'))
    pickle.dump(cose, open(cart_out + 'cose_ERA5.p', 'wb'))
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
            ker = xr.load_dataset(cart_k+ finam.format(vna, tip)) #cambiare

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

def load_kernel(ker, cart_k, cart_out):
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

    if ker=='ERA5':
        allkers=load_kernel_ERA5(cart_k, cart_out)
    if ker=='HUANG':
        allkers=load_kernel_HUANG(cart_k, cart_out)
    return allkers

def read_data(config):
    """
    Reads path of files from config.yml, read all vars and put them in a standardized dataset.
    """
    # read config

    # read data

    ds = standardize_names(ds)
    #deve essere un dataset non un datarray con tutte le variabili cosi possiamo chiamare var= ds['']

    return ds


def standardize_names(ds):
    """
    standardizes variable and coordinate names
    """
    return ds

def check_data(ds, piok):
    if len(ds["time"]) != len(piok["time"]):
        raise ValueError("Error: The 'time' columns in 'ds' and 'piok' must have the same length. To fix use variable 'time_range' of the function")
    return

######################################################################################
#### Aux functions

def ref_clim(config, allvars): 

    #open files

    piok=climatology()
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

    k=allkers[('cld', 't')]

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
                if time_range is not None:
                    pivar = pivar.sel(time = slice(time_range[0], time_range[1]))
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
            if time_range is not None:
                pivar = pivar.sel(time = slice(time_range[0], time_range[1]))
            piok = ctl.regrid_dataset(pivar[allvars], k.lat, k.lon)
            piok=ctl.running_mean(piok, 252)
        
    return(piok)
     

##calcolare tropopausa (Reichler 2003) 

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

def mask_pres(surf_pressure, cart_out:str, allkers):
    """
    Computes a "width mask" for atmospheric pressure levels based on surface pressure and kernel data.

    The function determines which pressure levels are above or below the surface pressure (`ps`) 
    and generates a mask that includes NaN values for levels below the surface pressure and 
    interpolated values near the surface. It supports kernels from HUANG and ERA5 datasets.

    Parameters:
    -----------
    surf_pressure : xr.Dataset
        An xarray dataset containing surface pressure (`ps`) values.
        The function computes a climatology based on mean monthly values.

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
    k=allkers[('cld', 't')]
    vlevs=pickle.load(open(cart_out + 'vlevs_HUANG.p', 'rb'))

    psclim = surf_pressure.groupby('time.month').mean()
    psye = psclim['ps'].mean('month')
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


def Rad_anomaly_planck_surf_wrapper(config):

    radiation=dict()
    ds = read_data()
    # filin_pi=
    # cart_k=
    # ker=
    # time_range=
    # time_chunk=
    # use_climatology=
    # cart_out=
    allkers=dict()
    allkers=load_kernel(ker, cart_k, cart_out)
    piok=dict()
    piok['ts']=climatology(filin_pi, allkers, 'ts', use_climatology, time_chunk)
    radiation = Rad_anomaly_planck_surf(ds, piok,  ker, allkers, cart_out, use_climatology, time_range)

    return (radiation)


def Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, use_climatology=True, time_range=None):
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

    Returns:
    - dict: A dictionary containing computed Planck surface radiation anomalies for clear-sky ('clr') and cloud ('cld') conditions.
    
    Outputs Saved to `cart_out`:
    - `gtas{suffix}.nc`: Global temperature anomaly series, grouped by year.
    - `dRt_planck-surf_global_{tip}{suffix}.nc`: Global surface Planck feedback for clear (`clr`) and 
      all (`cld`) sky conditions.

      Here, `{suffix}` = `"_climatology-{ker}kernels"` or `"_21yearmean-{ker}kernels"`, based on the 
      `use_climatology` flag and kernel type (`ker`).   
    """

    radiation=dict()
    k=allkers[('cld', 't')]
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    if time_range is not None:
        ds['ts'] = ds['ts'].sel(time = slice(time_range[0], time_range[1]))
    var = ctl.regrid_dataset(ds['ts'], k.lat, k.lon)   

    if use_climatology == False:
        check_data(ds['ts'], piok['ts'])
        piok=piok['ts'].drop('time')
        piok['time'] = var['time']
        piok = piok.chunk(var.chunks)
        anoms = var['ts'] - piok
    else:
        anoms = var['ts'].groupby('time.month') - piok['ts']
 
    for tip in ['clr', 'cld']:
        kernel = allkers[(tip, 'ts')]

        dRt = (anoms.groupby('time.month')* kernel).groupby('time.year').mean('time') 
        dRt_glob = ctl.global_mean(dRt)
        planck= dRt_glob.compute()
        radiation[(tip, 'planck-surf')]=planck
        planck.to_netcdf(cart_out+ "dRt_planck-surf_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        
    return(radiation)


#CALCOLO PLANK-ATMO E LAPSE RATE CON TROPOPAUSA VARIABILE (DA CONTROLLARE)

def Rad_anomaly_planck_atm_lr_wrapper(config):

    radiation=dict()
    ds = read_data()
    # filin_pi=
    # cart_k=
    # ker=
    # time_range=
    # time_chunk=
    # use_climatology=
    # cart_out=
    # pressure=
    allkers=dict()
    allkers=load_kernel(ker, cart_k, cart_out)
    piok=dict()
    allvars= 'ts ta'.split()
    for vnams in allvars:  
        piok[vnams]=climatology(filin_pi, allkers, vnams, True, time_chunk)
    
    # piok=climatology(filin_pi, ker, allkers, cart_out, 'ta', use_climatology, time_chunk)
    # piok_ts=climatology(filin_pi, ker, allkers, cart_out, 'ts', use_climatology, time_chunk)

    radiation = Rad_anomaly_planck_atm_lr(ds, piok, cart_out, ker, allkers, surf_pressure, time_range, use_climatology)

    return (radiation)

def Rad_anomaly_planck_atm_lr(ds, piok, cart_out:str, ker:str, allkers, surf_pressure=None, time_range=None, use_climatology = True):

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
        wid_mask=mask_pres(surf_pressure, cart_out, allkers)#forse qui si deve cambiare la pressure directory?

    if ker=='ERA5':
        vlevs=pickle.load(open(cart_out + 'vlevs_'+ker+'.p', 'rb'))
   
    if use_climatology==True:
        cos="_climatology"
    else:
        cos="_21yearmean"

    if time_range is not None:
        ds['ta'] = ds['ta'].sel(time = slice(time_range[0], time_range[1])) 
        ds['ts'] = ds['ts'].sel(time = slice(time_range[0], time_range[1])) 
    var = ctl.regrid_dataset(ds['ta'], k.lat, k.lon)
    var_ts = ctl.regrid_dataset(ds['ts'], k.lat, k.lon)

    if use_climatology==False:
        check_data(ds['ta'], piok['ta'])
        piok_ta=piok['ta'].drop('time')
        piok_ta['time'] = var['time']
        piok_ts=piok['ts'].drop('time')
        piok_ts['time'] = var['time']
        anoms_ok = var['ta'] - piok_ta
        ts_anom = var_ts['ts'] - piok_ts
    else:
        anoms_ok=var['ta'].groupby('time.month') - piok['ta']
        ts_anom=var_ts['ts'].groupby('time.month') - piok['ts']

    mask=mask_atm(var['ta'])
    anoms_ok = (anoms_ok*mask).interp(plev = cose) 

    for tip in ['clr','cld']:
        kernel = allkers[(tip, 't')]
        anoms_lr = (anoms_ok - ts_anom)  
        anoms_unif = (anoms_ok - anoms_lr)
        if ker=='HUANG':
            dRt_unif = (anoms_unif.groupby('time.month')*kernel*wid_mask/100).sum('player').groupby('time.year').mean('time')  
            dRt_lr = (anoms_lr.groupby('time.month')*kernel*wid_mask/100).sum('player').groupby('time.year').mean('time')   

        if ker=='ERA5':
            dRt_unif =(anoms_unif.groupby('time.month')*(kernel*vlevs.dp/100)).sum('player').groupby('time.year').mean('time')  
            dRt_lr = (anoms_lr.groupby('time.month')*(kernel*vlevs.dp/100)).sum('player').groupby('time.year').mean('time')  


        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        radiation[(tip,'planck-atmo')]=feedbacks_atmo
        radiation[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        
    return(radiation)

#CONTO ALBEDO

def Rad_anomaly_albedo_wrapper(config):
    
    radiation=dict()
    ds = read_data()
    # filin_pi=
    # cart_k=
    # ker=
    # time_range=
    # time_chunk=
    # use_climatology=
    # cart_out=
    allkers=dict()
    piok=dict()
    allkers=load_kernel(ker, cart_k, cart_out)
    piok['alb']=climatology(filin_pi,  allkers, 'alb', use_climatology, time_chunk)
    radiation = Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, use_climatology, time_range)

    return (radiation)

def Rad_anomaly_albedo(ds, piok, ker:str, allkers, cart_out:str, use_climatology=True, time_range=None):

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
    var = var_rsus['rsus']/var_rsds['rsds'] 
    if time_range is not None:
        var = var.sel(time = slice(time_range[0], time_range[1])) 
    var = ctl.regrid_dataset(var, k.lat, k.lon)

    # Removing inf and nan from alb
    piok = piok['alb'].where(piok['alb'] > 0., 0.)
    var = var.where(var > 0., 0.)
        
    if use_climatology==False:
        check_data(var, piok)
        piok=piok.drop('time')
        piok['time'] = var['time']
        anoms =  var - piok
    else:
        anoms =  var.groupby('time.month') - piok

    for tip in [ 'clr','cld']:
        kernel = allkers[(tip, 'alb')]

        dRt = (anoms.groupby('time.month')* kernel).groupby('time.year').mean('time') 
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        radiation[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip +cos+"-"+ker+"kernels.nc", format="NETCDF4")
        
    return(radiation)

##CALCOLO W-V

def Rad_anomaly_wv_wrapper(config):

    ds = read_data()
    # filin_pi=
    # cart_k=
    # ker=
    # time_range=
    # time_chunk=
    # use_climatology=
    # cart_out=
    # pressure=
    allkers=dict()
    allkers=load_kernel(ker, cart_k, cart_out)
    piok=dict()
    allvars= 'hus ta'.split()
    for vnams in allvars:  
        piok[vnams]=climatology(filin_pi, allkers, vnams, True, time_chunk)
    

    # piok=climatology(filin_pi, allkers, 'hus', use_climatology, time_chunk)
    # piok_ta=climatology(filin_pi,  allkers, 'ta', use_climatology, time_chunk)

    radiation = Rad_anomaly_wv(ds, piok, cart_out, ker, allkers, surf_pressure, time_range, use_climatology, time_chunk)

    return (radiation)

def Rad_anomaly_wv(ds, piok,  cart_out:str, ker:str, allkers, surf_pressure, time_range=None, use_climatology=True):
    
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

    
    if time_range is not None:
        ds['hus'] = ds['hus'].sel(time = slice(time_range[0], time_range[1])) 
        ds['ta'] = ds['ta'].sel(time = slice(time_range[0], time_range[1]))
    var = ctl.regrid_dataset(ds['hus'], k.lat, k.lon)
    var_ta=ds['ta']
    mask=mask_atm(var_ta['ta'])

    Rv = 487.5 # gas constant of water vapor
    Lv = 2.5e+06 # latent heat of water vapor

    if use_climatology==False:
        check_data(ds['ta'], piok['ta'])
        piok_hus=piok['hus'].drop('time')
        piok_hus['time'] = var['time']
        piok_ta=piok['ta'].drop('time')
        piok_ta['time'] = var['time']
 
    ta_abs_pi = piok_ta.interp(plev = cose)
    var_int = (var['hus']*mask).interp(plev = cose)
    piok_int = piok_hus.interp(plev = cose)
  
    if ker=='HUANG':
        wid_mask=mask_pres(surf_pressure, cart_out, allkers)
        if use_climatology==True:
            anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int.groupby('time.month'), piok_int , dask = 'allowed')
            coso3= anoms_ok3.groupby('time.month') *dlnws(ta_abs_pi)
       
        else:
            anoms_ok3 = xr.apply_ufunc(lambda x, mean: np.log(x) - np.log(mean), var_int, piok_int , dask = 'allowed')
            coso3= anoms_ok3*dlnws(ta_abs_pi)
    
    if ker=='ERA5': 
        if use_climatology==False:
            anoms= var_int-piok_int
            coso = (anoms/piok_int) * (ta_abs_pi**2) * Rv/Lv
        else:
            anoms= var_int.groupby('time.month')-piok_int
            coso = (anoms.groupby('time.month')/piok_int).groupby('time.month') * (ta_abs_pi**2) * Rv/Lv #dlnws(ta_abs_pi) you can also use the function
    
    for tip in ['clr','cld']:
        kernel_lw = allkers[(tip, 'wv_lw')]
        kernel_sw = allkers[(tip, 'wv_sw')]
        kernel = kernel_lw + kernel_sw
        
        if ker=='HUANG':
            dRt = (coso3.groupby('time.month')* kernel* wid_mask/100).sum('player').groupby('time.year').mean('time')

        if ker=='ERA5':
            dRt = (coso.groupby('time.month')*( kernel* vlevs.dp / 100) ).sum('player').groupby('time.year').mean('time')
        
        dRt_glob = ctl.global_mean(dRt)
        wv= dRt_glob.compute()
        radiation[(tip, 'water-vapor')]=wv
        wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" +tip+cos +"-"+ker+"kernels.nc", format="NETCDF4")
        
    return radiation

##CALCOLO EFFETTIVO DEI FEEDBACK


def calc_fb_wrapper(config):

    ds=read_data() #unire anche quelli di fili_4c1
    # filin_pi= 
    # ker=
    # cart_k=
    # cart_out=
    # pressure=
    # use_climatology=
    # time_range=
    # time_chunk=
    allkers=dict()
    allkers=load_kernel(ker, cart_k, cart_out)
    piok=dict()
    allvars= 'ts tas hus alb ta'.split()
    for vnams in allvars:  
        piok[vnams]=climatology(filin_pi, allkers, vnams, time_range, use_climatology, time_chunk)
    allvars1= 'rlutcs rsutcs rlut rsut'.split()
    for vnams in allvars1:  
        piok[vnams]=climatology(filin_pi, allkers, vnams, time_range, True, time_chunk)
    
    fb_coef, fb_cloud, fb_cloud_err=calc_fb(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology, time_range)

    return fb_coef, fb_cloud, fb_cloud_err

def calc_fb(ds, piok, ker, allkers, cart_out, surf_pressure, use_climatology=True, time_range=None):
    """
    
    """
    if use_climatology==True:
        cos="_climatology"
        print(cos)
    else:
        cos="_21yearmean"

    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_surf(ds, piok, ker, allkers, cart_out, use_climatology, time_range)
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_atm_lr(ds, piok, cart_out, ker, allkers, surf_pressure, time_range, use_climatology)
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_albedo(ds, piok, ker, allkers, cart_out, use_climatology, time_range)
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr"+cos+"-"+ker+"kernels.nc")
    if not os.path.exists(path):
        Rad_anomaly_wv(ds, piok, cart_out, ker, allkers, surf_pressure, time_range, use_climatology)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()

    #compute gtas
    k=allkers[('cld', 't')]
    var_tas= ctl.regrid_dataset(ds['tas'], k.lat, k.lon) 

    if use_climatology == False:
        piok_tas=piok['tas'].drop('time')
        piok_tas['time'] = var_tas['time']
        piok_tas = piok_tas.chunk(var_tas.chunks)
        anoms_tas = var_tas['tas'] - piok_tas
    else:
        anoms_tas = var_tas['tas'].groupby('time.month') - piok['tas']
        
    gtas = ctl.global_mean(anoms_tas).groupby('time.year').mean('time') 
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()

    print('calcolo feedback')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ cos+"-"+ker+"kernels.nc",  use_cftime=True)
            feedback=feedbacks.groupby((feedbacks.year-1) // 10 * 10).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res
    
    #cloud
    fb_cloud, fb_cloud_err = feedback_cloud(ds, piok, fb_coef, gtas, time_range)
    
    return fb_coef, fb_cloud, fb_cloud_err 

# ###CLOUD FEEDBACK shell 2008

def feedback_cloud_wrapper(config):

    ds = read_data() #questo da filin4c e filin4c1
    # filin_pi=
    # cart_k=
    # ker=
    # time_range=
    # time_chunk=
    # cart_out=
    # use_climatology=

    allkers=dict()
    allkers=load_kernel(ker, cart_k, cart_out)
    piok=dict()
    allvars= 'rlutcs rsutcs rlut rsut'.split()
    for vnams in allvars:  
        piok[vnams]=climatology(filin_pi, allkers, vnams, True, time_chunk)

    fb_coef = calc_fb(config)

    #compute gtas here? 
    #ds_cloud has to be regridded before the function(let it here?)
    k=allkers[('cld', 't')]

    nomi='rlutcs rsutcs'.split()
    for nom in nomi:
        ds[nom] = ctl.regrid_dataset(ds[nom], k.lat, k.lon) #no mettiamo anche questi in ds
        #ma da capire quando regriddarli


    var_tas= ctl.regrid_dataset(ds['tas'], k.lat, k.lon) 

    if use_climatology == False:
        piok_tas=piok['tas'].drop('time')
        piok_tas['time'] = var_tas['time']
        piok_tas = piok_tas.chunk(var_tas.chunks)
        anoms_tas = var_tas['tas'] - piok_tas
    else:
        anoms_tas = var_tas['tas'].groupby('time.month') - piok['tas']
        
    gtas = ctl.global_mean(anoms_tas).groupby('time.year').mean('time') 
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()

    fb_cloud, fb_cloud_err = feedback_cloud(ds, piok, fb_coef, gtas, time_range)

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
    if not (ds['rlut'].dims["lon"] == ds['rsutcs'].dims["lon"] and ds['rlut'].dims["lat"] == ds['rsutcs'].dims["lat"]):
        raise ValueError("Error: The spatial grids ('lon' and 'lat') datasets must match.")
    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    
    if time_range is not None:
        nomi='rlut rsut rlutcs rsutcs'.split()
        for nom in nomi:
            ds[nom] = ds[nom].sel(time = slice(time_range[0], time_range[1]))
    
    rlut=ds['rlut']
    rsut=ds['rsut']
    rsutcs = ds['rsutcs']
    rlutcs = ds['rlutcs']

    N = - rlut['rlut'] - rsut['rsut']
    N0 = - rsutcs['rsutcs'] - rlutcs['rlutcs']

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


###### Spectral kernels ######



###### General functions ######



###### Plotting ######

