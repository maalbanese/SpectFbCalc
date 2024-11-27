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


###### Broad-band kernels ######

#PRENDERE I KERNEL

def ker(cart_k:str, cart_out:str):
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

##CALCOLO PIMEAN

def clim(filin_pi:str, cart_k:str, cart_out:str, allvars:str, MEAN=True, chu=12):
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

    MEAN : bool, optional, default=True
        Determines the processing type:
        - `True`: Computes the climatological mean over the entire time period.
        - `False`: Computes a 21-year running mean over a specific time slice.

    chu : int, optional, default=12
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
        print("Kernel file not found. Running ker() to generate it.")
        ker(cart_k)  # Ensure that ker() is properly defined elsewhere in the code
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out

    if allvars=='alb':
       allvars='rsus rsds'.split()
       if MEAN==True:
        for vnam in allvars:
            filist = glob.glob(filin_pi.format(vnam))
            filist.sort()

            var = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)

            var_mean = var.mean('time')

            var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)

            pimean[vnam] = var_mean[vnam].compute()

        piok = pimean[('rsus')]/pimean[('rsds')]

       else:
        piok=dict()
        for vnam in allvars:
            filist = glob.glob(filin_pi.format(vnam))
            filist.sort()
            pivar = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)
            pivar = pivar.sel(time = slice('2540-01-01', '2689-12-31')) #MODIFICA
            piok[vnam] = ctl.regrid_dataset(pivar[vnam], k.lat, k.lon)
     
        pivar['alb']=piok[('rsus')]/piok[('rsds')]
        piok = ctl.regrid_dataset(pivar['alb'], k.lat, k.lon)
        piok=ctl.running_mean(piok, 252)
     
    else:
     if MEAN==True:
    
        filist = glob.glob(filin_pi.format(allvars))
        filist.sort()

        var = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)

        var_mean = var.mean('time')

        var_mean = ctl.regrid_dataset(var_mean, k.lat, k.lon)

        piok = var_mean[allvars].compute()

        
     else:
        filist = glob.glob(filin_pi.format(allvars))
        filist.sort()
        pivar = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)
        pivar = pivar.sel(time = slice('2540-01-01', '2689-12-31')) #MODIFICA
        piok = ctl.regrid_dataset(pivar[allvars], k.lat, k.lon)
        piok=ctl.running_mean(piok, 252)
        
    return(piok)
     

##calcolare tropopausa (Reichler 2003) 

def mask_atm(filin_4c:str, chu=12):
    """
    Generates a mask for atmospheric temperature data based on the lapse rate threshold.
    as in (Reichler 2003) 

    Parameters:
    -----------
    filin_4c : str
        Path template for input NetCDF files, with a placeholder for the variable name (e.g., 'ta').
        Placeholders should be formatted as `{}` to allow string formatting.

    chu : int, optional (default=12)
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
    temp = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)

    A=(temp.plev/temp['ta'])*(9.81/1005)
    laps1=(temp['ta'].diff(dim='plev'))*A  #derivata sulla verticale = laspe-rate

    laps1=laps1.where(laps1<=-2)
    mask = laps1/laps1
    return mask

### Mask for surf pressure

def mask_pres(dir:str, cart_out:str, cart_k:str):
    """
    Generates a pressure mask based on climatological surface pressure and vertical levels.

    Parameters:
    -----------
    dir : str
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
        print("Kernel file not found. Running ker() to generate it.")
        ker(cart_k)  # Ensure that ker() is properly defined elsewhere in the code
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
    vlevs=pickle.load(open(cart_out + 'vlevs.p', 'rb'))#prendi da cart_out
    ps = xr.open_mfdataset(dir)
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

def fb_planck_surf(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, MEAN=True, chu=12):
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

    MEAN : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_ts.nc`).
        If False, uses running mean data from precomputed PI files (`piok_ts_21y.nc`).

    chu : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    None
        The function does not return a value but saves the following outputs to the `cart_out` directory:
        - Global annual mean surface Planck feedback for clear-sky and all-sky conditions.
        - Temperature anomalies (`ts_anom`) relative to the PI climatology as NetCDF files.
        - Global temperature anomaly series (`gtas`) as NetCDF files.

   
    """   
    allkers=dict()
    # Check if the kernel file exists, if not, call the ker() function
    k_file_path = os.path.join(cart_out, 'k.p')
    if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=ker(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
    else:
       allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
    k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
    vlevs=pickle.load(open(cart_out + 'vlevs.p', 'rb'))#prendi da cart_out

    filist = glob.glob(filin_4c.format('ts'))
    filist.sort()
    var = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)
    var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
    var = ctl.regrid_dataset(var['ts'], k.lat, k.lon)
    allvars='ts'
    piok=clim(filin_pi, cart_k, cart_out, allvars, MEAN, chu)

    if MEAN==False:
     piok=piok.drop('time')
     piok['time'] = var['time']
    
    anoms = var - piok
    anoms.compute()
    ts_anom = anoms
    ts_anom.to_netcdf(cart_out+ "ts_anom.nc", format="NETCDF4")
    gtas = ctl.global_mean(anoms).groupby('time.year').mean('time')
    gtas.to_netcdf(cart_out+ "gtas.nc", format="NETCDF4")
 
    for tip in ['clr', 'cld']:
     kernel = allkers[(tip, 'ts')].lwkernel

     dRt = xr.apply_ufunc(lambda x, ker: x*ker, anoms.groupby('time.month'), kernel, dask = 'allowed').groupby('time.year').mean('time')
     dRt_glob = ctl.global_mean(dRt)
     feedbacks = dRt_glob.compute()
     feedbacks.to_netcdf(cart_out+ "dRt_planck-surf_global_" +tip +".nc", format="NETCDF4")


#CALCOLO PLANK-ATMO E LAPSE RATE CON TROPOPAUSA VARIABILE (DA CONTROLLARE)

def fb_plank_atm_lr(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, dir, MEAN = True, chu=12):
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

    dir : str
        Directory containing surface pressure (`ps`) data required for pressure masking.

    MEAN : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_ta.nc`).
        If False, uses running mean data from precomputed PI files (`piok_ta_21y.nc`).

    chu : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    None
        The function does not return a value but saves the following outputs to the `cart_out` directory:
        - Global annual mean Planck and lapse-rate feedbacks for clear-sky and all-sky conditions.
        - Preindustrial absolute temperature profile interpolated to the kernel levels.
        - Anomaly and mask data for further analysis.

    """
   
   allkers=dict()
    # Check if the kernel file exists, if not, call the ker() function
   k_file_path = os.path.join(cart_out, 'k.p')
   if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=ker(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
   else:
       allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
   k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
   cose=pickle.load(open(cart_out + 'cose.p', 'rb'))
   
   filist = glob.glob(filin_4c.format('ta'))
   filist.sort()
   var = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)
   var = var['ta']
   var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
   var = ctl.regrid_dataset(var, k.lat, k.lon)
   allvars='ta'
   piok=clim(filin_pi, cart_k, cart_out, allvars, MEAN, chu)

   if MEAN==False:
      piok=piok.drop('time')
      piok['time'] = var['time']
   anoms_ok = var - piok

   ta_abs_pi = piok.interp(plev = cose)
   ta_abs_pi.to_netcdf(cart_out+ "ta_abs_pi.nc", format="NETCDF4")
   mask=mask_atm(filin_4c, chu)
   wid_mask=mask_pres(dir, cart_out, cart_k)
   anoms_ok = (anoms_ok*mask).interp(plev = cose)
   ts_anom=xr.open_dataarray(cart_out+"ts_anom.nc", chunks = {'time': chu}, use_cftime=True) 

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
    
     feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip +".nc", format="NETCDF4")
     feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip +".nc", format="NETCDF4")

#CONTO ALBEDO

def fb_albedo(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, MEAN=True, chu=12):
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

    MEAN : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_alb.nc`).
        If False, uses running mean data from precomputed PI files (`piok_alb_21y.nc`).

    chu : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    None
        The function does not return a value but saves the following outputs to the `cart_out` directory:
        - Global annual mean albedo feedback for clear-sky and all-sky conditions.
        - Albedo anomaly data for further analysis.

  """
   
  allkers=dict()
    # Check if the kernel file exists, if not, call the ker() function
  k_file_path = os.path.join(cart_out, 'k.p')
  if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=ker(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
  else:
       allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
  k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out

  
  feedbacks=dict()
  filist_1 = glob.glob(filin_4c.format('rsus'))
  filist_1.sort()
  var_rsus = xr.open_mfdataset(filist_1, chunks = {'time': chu}, use_cftime=True)['rsus']
  filist_2 = glob.glob(filin_4c.format('rsds'))
  filist_2.sort()
  var_rsds = xr.open_mfdataset(filist_2, chunks = {'time': chu}, use_cftime=True)['rsds']
  var = var_rsus/var_rsds
  var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
  var = ctl.regrid_dataset(var, k.lat, k.lon)
  allvars='alb'
  piok=clim(filin_pi, cart_k, cart_out, allvars, MEAN, chu)

  if MEAN==False:
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
    feedbacks = 100*dRt_glob
    feedbacks.to_netcdf(cart_out+ "dRt_albedo_global_" +tip +".nc", format="NETCDF4")

##CALCOLO W-V
def fb_wv(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, dir:str, MEAN=True, chu=12):
    
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

    dir : str
        Path to the pressure-level data directory, used for creating masks and vertical integration.

    MEAN : bool, optional (default=True)
        If True, uses mean climatological data from precomputed PI files (`amoc_all_1000_hus.nc`).
        If False, uses running mean data from precomputed PI files (`piok_hus_21y.nc`).

    chu : int, optional (default=12)
        Chunk size for loading data using xarray to optimize memory usage.

    Returns:
    --------
    None
        The function does not return a value but saves the following outputs to the `cart_out` directory:
        - Global annual mean water vapor feedback for clear-sky and all-sky conditions.
        - Intermediate water vapor anomalies for further analysis.

   """
   allkers=dict()
    # Check if the kernel file exists, if not, call the ker() function
   k_file_path = os.path.join(cart_out, 'k.p')
   if not os.path.exists(k_file_path):
        print("Kernel file not found. Running ker() to generate it.")
        allkers=ker(cart_k, cart_out)  # Ensure that ker() is properly defined elsewhere in the code
   else:
       allkers=pickle.load(open(cart_out + 'allkers.p', 'rb'))
   k=pickle.load(open(cart_out + 'k.p', 'rb'))#prendi da cart_out
   cose=pickle.load(open(cart_out + 'cose.p', 'rb'))
   ta_abs_pi=xr.open_dataarray(cart_out+"ta_abs_pi.nc",  chunks = {'time': chu},  use_cftime=True)
   mask=mask_atm(filin_4c)
   wid_mask=mask_pres(dir, cart_out, cart_k)
   filist = glob.glob(filin_4c.format('hus')) 
   filist.sort()
   var = xr.open_mfdataset(filist, chunks = {'time': chu}, use_cftime=True)
   var = var['hus']
   var = var.sel(time = slice('1850-01-01', '1999-12-31')) #MODIFICA
   var = ctl.regrid_dataset(var, k.lat, k.lon)
   allvars='hus'
   piok=clim(filin_pi, cart_k, cart_out, allvars, MEAN, chu)

   if MEAN==False:
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
      feedbacks= dRt_glob.compute()
      feedbacks.to_netcdf(cart_out+ "dRt_water-vapor_global_" +tip +".nc", format="NETCDF4")

##CALCOLO EFFETTIVO DEI FEEDBACK
def calc_fb(filin_4c:str, filin_pi:str, cart_out:str, cart_k:str, dir:str, MEAN=True, chu=12):
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

    dir : str
        Path to the pressure-level data directory, used for additional computations such as water vapor feedback.

    MEAN : bool, optional (default=True)
        If True, computes anomalies relative to the mean climatology from precomputed PI files.
        If False, computes anomalies relative to running mean PI data.

    chu : int, optional (default=12)
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
    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr.nc")
    if not os.path.exists(path):
        fb_planck_surf(filin_4c, filin_pi, cart_out, cart_k, MEAN, chu)
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr.nc")
    if not os.path.exists(path):
        fb_plank_atm_lr(filin_4c, filin_pi, cart_out, cart_k, dir, MEAN, chu)
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr.nc")
    if not os.path.exists(path):
        fb_albedo(filin_4c, filin_pi, cart_out, cart_k, MEAN, chu)
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr.nc")
    if not os.path.exists(path):
        fb_wv(filin_4c, filin_pi, cart_out, cart_k, dir, MEAN, chu)    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    fb_coef = dict()
    gtas=xr.open_dataarray(cart_out+"gtas.nc",  use_cftime=True)
    gtas= gtas.groupby((gtas.year-1) // 10 * 10).mean()
    print('calcolo feedback')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:

            feedbacks=xr.open_dataarray(cart_out+"dRt_" +fbn+"_global_"+tip+ ".nc",  use_cftime=True)
            feedback=feedbacks.groupby((feedbacks.year-1) // 10 * 10).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res
    
    return(fb_coef)


###### Spectral kernels ######



###### General functions ######



###### Plotting ######

