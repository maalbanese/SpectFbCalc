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
from xarray import unify_chunks
 
from pathlib import Path
from cdo import Cdo

######################################################################
### Functions 
time_coder = xr.coders.CFDatetimeCoder(use_cftime = True)

STD_VARS = {"hus", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_LOGQ = {"hus_log", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_NOALB = {"hus", "rlut", "rsdt", "rlutcs", "rsut", "rsutcs", "ta", "tas", "ts", "rsds", "rsus"}


def regrid(ds, target_ds):
    """
    Wrapper to ctl.regrid, to avoid data array loses name
    """
    coso = ctl.regrid_dataset(ds, target_ds.lat, target_ds.lon)
    coso.name = ds.name
    return coso



class Kernel:
    """
    Loads the kernels and the additional information needed.  
    
    ----------
    name : one among "HUANG", "ERA5", "SPECTRAL"
    """
 
    def __init__(self, name, config = None, path_input = None, filename_template = None) -> None:
        if name not in ['HUANG', 'ERA5', 'SPECTRAL']:
            raise ValueError(f'kernel name {name} not supported')
            
        self.name = name
        if config is not None:
            self.cart_k = config['kernels'][name]['path_input']
            self.filename_template = config['kernels'][name]['filename_template']
        else:
            if path_input is None: raise ValueError(f"config is None and path_input not provided for kernel {self.name}")
            if filename_template is None: raise ValueError(f"config is None and filename_template not provided for kernel {self.name}")
            self.cart_k = path_input
            self.filename_template = filename_template

        print(f"Loading kernel: {name}")
        self.kernel, self.dp = load_kernel(self.name, self.cart_k, finam = self.filename_template)

        self.use_log_wv = False
        if self.name in ['HUANG']: # Add here other kernels that need hus_log
            self.use_log_wv = True

        if self.name in ['HUANG', 'ERA5']:
            self.dp = self.dp/100. # in units of 100 hPa


    def recompute_dp(self, experiment) -> None:
        """
        Recomputes the atmospheric layers based on the model's surface pressure.

        experiment is an Experiment instance
        """

        if "surf_pressure" not in experiment.ds:
            print(f"Surface pressure not available in experiment {experiment.name}, using default dp for kernel {self.name}")
            return

        if self.dp is None:
            raise ValueError(f"Cannot recompute dp for kernel {self.name}: dp not available")
            
        # Surface mask
        wid_mask = np.empty([len(self.dp.plev)] + list(experiment.surf_pressure.shape))
        
        for ila in range(len(experiment.surf_pressure.lat)):
            for ilo in range(len(experiment.surf_pressure.lon)):
                ind = np.where((experiment.surf_pressure[ila, ilo].values - self.dp.plev.values) > 0)[0][0]
                wid_mask[:ind, ila, ilo] = np.nan
                wid_mask[ind, ila, ilo] = experiment.surf_pressure[ila, ilo].values - self.dp.plev.values[ind]
                wid_mask[ind+1:, ila, ilo] = self.dp.values[ind+1:]

        k = self.kernel[('clr', 't')]
        wid_mask = xr.DataArray(wid_mask, dims = k.dims[1:], coords = k.drop('month').coords)

        if self.name in ['HUANG', 'ERA5']:
            self.dp = wid_mask/100. # in units of 100 hPa
        else:
            self.dp = wid_mask


    def __repr__(self) -> str:
        return (
            f"Kernel(\n"
            f"  name      = {self.name!r}\n"
            f"  cart_k = {self.cart_k}\n"
            f"  available kernels = {self.kernel.keys()}\n"
            f")"
        )

 
class Experiment:
    """
    Loads the data of a specific experiment.  
    
    ----------
    name : str
        An identifying name for the experiment (e.g. {model}_{member}_{exp_type})
    variables : set[str]
        Climate variable names tracked by this instance.
    dataset : xr.Dataset
        Merged lazy dataset of remapped data populated by `load_remapped`.
    """
 
    def __init__(self, name, orig_dir, remap_dir = "remapped", raw_variables = STD_VARS_NOALB, time_chunk = 20, variable_mapping = None) -> None:
        self.name: str = name
        self.raw_variables: set[str] | list[str] | tuple[str] = raw_variables
        
        self.orig_dir: Path = Path(orig_dir)
        self.remap_dir: Path = Path(remap_dir)

        self.remap_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = {'time': time_chunk}
        self.chunks_remap = {'time': 120} # this is for remapped data

        if variable_mapping is None:
            self.variable_mapping = {var: var for var in self.raw_variables}
        else:
            self.variable_mapping = variable_mapping

        self.load_file_dict()
        self.raw_data = dict()
        self.ds = xr.Dataset()
        self.ds_anom = xr.Dataset()
        self.ds_clim = xr.Dataset()
 
    # ------------------------------------------------------------------
    # 1. Lazy loading of raw DataArrays
    # ------------------------------------------------------------------
    def load_file_dict(self):
        file_dict = {}
        for var in self.raw_variables:
            base_folder = self.orig_dir / self.variable_mapping[var]

            pattern = f"{self.variable_mapping[var]}*.nc"
            files = sorted(base_folder.glob(pattern))
        
            if not files:
                files = sorted(base_folder.glob(f"**/{pattern}"))
        
            file_dict[var] = files

        self.file_dict = file_dict

    # def load_file_dict(self):
    #     file_dict = {}
    #     for var in self.raw_variables:
    #         pattern = f"{self.variable_mapping[var]}/{self.variable_mapping[var]}*.nc"
    #         matches = []
    #         for d in self.orig_dir:  # orig_dir is now always a list
    #             matches.extend(d.glob(pattern))
    #         file_dict[var] = sorted(matches)
    #     self.file_dict = file_dict

 
    def load_raw(self) -> None:
        """
        Lazily open each file in *file_dict* as an xr.DataArray and store
        them in ``self.raw_data``.
 
        Parameters
        ----------
        file_dict : list of path-like
            Paths to NetCDF files.
        """
        print('Loading raw data...')

        # self.raw_data = {var: xr.open_mfdataset(self.file_dict[var], combine='by_coords', decode_times=time_coder, chunks = self.chunks, preprocess = preproc)[self.variable_mapping[var]]
        #     for var in self.raw_variables
        # }
        self.raw_data = {}
        for var in self.raw_variables:
            print(var)
            self.raw_data[var] = xr.open_mfdataset(self.file_dict[var], combine='by_coords', decode_times=time_coder, chunks = self.chunks, preprocess = preproc)[self.variable_mapping[var]]
 
    # ------------------------------------------------------------------
    # 2. CDO remapping to a target grid
    # ------------------------------------------------------------------
 
    def remap(self, target_ds: xr.Dataset, save_remapped = False) -> None:
        """
        Interpolates to target_ds (the kernel ds).
 
        Parameters
        ----------
        target_ds

        save_rempped
        """

        ### Improve! use smmregrid instead. compatibility with gaussian reduced/curvilinear?
        remapped = {var: regrid(self.raw_data[var], target_ds) for var in self.raw_variables}

        # compute and save
        if save_remapped:
            print("Saving remapped to disk")
            for var in remapped:
                print(var)
                remapped[var] = remapped[var].compute()
                remapped[var].to_netcdf(os.path.join(self.remap_dir, f'{var}_{self.name}_remapped.nc'))
            
        self.ds = xr.merge([remapped[var] for var in remapped])



    def remap_cdo(
        self,
        target_grid_file: str | Path,
        method: str = "remapbil",
    ) -> None:
        """
        Interpolate each file in *file_dict* to the grid of *target_grid_file*
        using CDO Python bindings and write results to ``self.remap_dir``.
 
        Parameters
        ----------
        file_dict : list of path-like
            Source files to be remapped.
        target_grid_file : path-like
            NetCDF file whose grid defines the target resolution.
        method : str
            CDO remapping operator: ``"remapbil"`` (default), ``"remapcon"``,
            ``"remapnn"``, etc.
        """
        target_grid_file = str(target_grid_file)

        cdo = Cdo()
        remap_fn = cdo(method)
 
        for var in self.file_dict:
            for src_file in self.file_dict[var]:
                dst_file = self.remap_dir / src_file.name
                remap_fn(target_grid_file, input=src_file, output=dst_file)

    
    def vertical_interp(self, target_ds):
        print('check vertical dimension')
        self.ds = check_vertical(self.ds)
        self.ds = self.ds.interp(plev = target_ds.plev)

    # ------------------------------------------------------------------
    # 3. Load remapped data into a merged, lazy Dataset
    # ------------------------------------------------------------------
 
    def load_remapped(self) -> None:
        """
        Lazily read all remapped files in ``self.remap_dir`` and merge them
        into a single xr.Dataset stored in ``self.dataset``.
 
        Parameters
        ----------
        pattern : str
            Glob pattern to discover files inside ``self.remap_dir``.
        """

        pattern = f"*_{self.name}_remapped.nc"
        files = sorted(self.remap_dir.glob(pattern))
        if len(files) > 0:
            self.ds = xr.open_mfdataset(files, combine = "by_coords", decode_times=time_coder, chunks = self.chunks_remap, preprocess = preproc)
        else:
            raise ValueError('No remapped dataset found on disk!')
            # Alternatively, can compute them from raw

    def check_remapped(self) -> None:
        """
        Check if remapped files are there.

        IMPROVE: check that all variables are there
        """

        pattern = f"*_{self.name}_remapped.nc"
        files = sorted(self.remap_dir.glob(pattern))
        if len(files) > 0:
            print(f'{self.name} already remapped')
            return True
        else:
            return False


    def load_surf_pressure(self, pressure_path, target_ds) -> None:
        """
        Loads surface pressure to compute atmospheric layers.
        Interpolates to kernel grid
        """
        if pressure_path:  # If pressure data is specified, load it
            print("Loading surface pressure data...")
            ps_files = sorted(glob.glob(pressure_path))  # Support for patterns like "*.nc"
            if not ps_files:
                raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
            
            surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords', decode_times=time_coder)
        
        psclim = surf_pressure.groupby('time.month').mean(dim='time')
        psye = psclim['ps'].mean('month')
        
        psye_rg = regrid(psye, target_ds).compute()

        self.surf_pressure = psye_rg
    
    def Net_TOA(self):
        print('Creating Net TOA variables')
        self.ds['Net'] = self.ds['rsdt'] - self.ds['rlut'] - self.ds['rsut'] #net_toa_allsky
        self.ds['Net0'] = self.ds['rsdt'] - self.ds['rlutcs'] - self.ds['rsutcs'] #net_toa_clr
 
    
    def check_albedo(self) -> None:
        """
        Checks if the albedo is loaded.
        """

        if not self.ds:
            raise ValueError('Remapped data not loaded (self.ds is empty)')
    
        if 'alb' in self.ds.data_vars:
            print('alb already in ds')
        else:
            if 'rsus' and 'rsds' in self.ds:
                print("Computing albedo from rsus and rsds")
                albedo = self.ds['rsus']/self.ds['rsds']
                self.ds['alb'] = albedo.where(albedo > 0., 0.)

                del self.ds['rsus']
                del self.ds['rsds']
            else:
                raise ValueError('alb or rsus or rsds not found in ds! Cannot compute albedo. Vars in ds: ', self.ds.data_vars)


    def check_hus_log(self) -> None:
        """
        Checks if hus_log is in ds.
        """

        if not self.ds:
            raise ValueError('Remapped data not loaded (self.ds is empty)')
    
        if 'hus_log' in self.ds.data_vars:
            print('hus_log already in ds')
        else:
            print('Applying log to hus')
            self.ds['hus_log'] = da.log(self.ds['hus'])
            del self.ds['hus']


    def check_vars(self, variables = STD_VARS_LOGQ) -> None:
        """
        Checks if all variables needed are loaded. 
        """

        if not self.ds:
            raise ValueError('Remapped data not loaded (self.ds is empty)')
        
        if 'alb' in variables: self.check_albedo()
        if 'hus_log' in variables: self.check_hus_log()
        self.Net_TOA()

        for var in variables:
            if var in self.ds:
                print(f"{var} loaded")
            else:
                print(f'missing {var}!')

        self.variables = variables
    
    def check_time_range(self, config):
        time_range=config['time_range_exp']
        if time_range is not None:
            self.ds = self.ds.sel(time=slice(time_range['start'], time_range['end']))
        


    def compute_clim(self, time_range = None, compute = True):
        """
        Computes climatology. If time_range is provided, select a period.
        """
        self.ds_clim = compute_climatology(self.ds, time_range=time_range)

        if compute:
            self.ds_clim = self.ds_clim.compute()


    def compute_runmean(self, window_years = 21, time_range = None, compute = True):
        """
        Computes running mean of dataset.
        """
        self.ds_clim = compute_running_mean(self.ds, window_years=window_years, time_range = time_range)

        if compute:
            self.ds_clim = self.ds_clim.compute()


    def compute_anom_clim(self, control):
        """
        Removes monthly climatology from a control run.
        """

        from importlib.util import find_spec
    
        if find_spec("flox") is not None:
            # Use groupby
            self.ds_anom = self.ds.groupby('time.month') - control.ds_clim
            self.ds_anom = self.ds_anom.chunk(self.ds.chunks)
        else:
            # Expand to avoid groupby
            self.ds_anom = self.ds - expand_clim(control.ds_clim, self.ds)


    def compute_anom_aligned(self, control):
        """
        Removes climatology from a control run, aligning the time axis (experiment and climatology need to have the same length!).
        """

        check_time_axis(self.ds, control.ds_clim)
        clim_aligned = control.ds_clim.drop("time")
        clim_aligned["time"] = self.ds["time"]
        self.ds_anom = self.ds - clim_aligned
        
    
    def check_lazy_loading(self):
        check_lazy_loading(self.ds)
        
 
    def __repr__(self) -> str:
        return (
            f"Experiment(\n"
            f"  name      = {self.name!r}\n"
            f"  variables (raw) = {self.raw_variables}\n"
            f"  variables = {self.variables}\n"
            f"  remap_dir  = {self.remap_dir}\n"
            f"  raw_data = {len(self.raw_data)} DataArray(s)\n"
            f"  ds    = {list(self.ds.data_vars)}\n"
            f"  ds_anom    = {list(self.ds_anom.data_vars)}\n"
            f"  ds_clim    = {list(self.ds_clim.data_vars)}\n"
            f")"
        )


def expand_clim(ds_clim, ds):
    """
    Expand a climatology along the time axis (to avoid groupby).
    without Flox, the groupby creates: PerformanceWarning: Slicing with an out-of-order index is generating 165 times more chunks
    """
    # Build a month coordinate from your data's time axis
    months = ds.time.dt.month
    # Index into the climatology and assign time as the new coordinate
    # ds_clim = ds_clim.chunk({'month': 12})
    expanded = ds_clim.sel(month=months).drop_vars("month").assign_coords(time=ds.time).chunk(ds.chunks)

    return expanded


def compute_climatology(ds, time_range = None):
    """
    Computes climatology. If time_range is provided, select a period.
    """
    if time_range is not None:
        ds_clim = ds.sel(time=slice(time_range['start'], time_range['end']))
    else:
        ds_clim = ds

    ds_clim = ds_clim.groupby('time.month').mean()

    return ds_clim


def compute_running_mean(ds, window_years = 21, time_range = None):
    """
    Computes running mean of dataset.
    """
    if time_range is not None:
        ds_clim = ds.sel(time=slice(time_range['start'], time_range['end']))

    ds_clim = ctl.running_mean(ds_clim, window_years*12)

    return ds_clim


def compute_anomalies(exp, control, method="climatology", window_years=21, time_range_clim = None):
    """
    Compute anomalies between a variable and its climatology/reference.

    Parameters
    ----------
    exp : Experiment instance for 4x
    control : Experiment instance for picontrol
    method : str, default "monthly"
        Method for anomaly computation:
        - "climatology"            : monthly averaged climatology to calculate the anomaly
        - "running_mean"              : running mean climatology to calculate the anomaly 
    window_years : int, default 21
        Num of years used for running mean
    """

    if method == "climatology":
        control.compute_clim(time_range = time_range_clim, compute = True)
        exp.compute_anom_clim(control)

    elif method == "running_mean":
        control.compute_runmean(window_years = window_years, compute = True)
        exp.compute_anom_aligned(control)
    else:
        raise ValueError(f"Unknown anomaly method {method}")





################################################################################################

def mytestfunction():
    print('test!')
    return

###### INPUT/OUTPUT SECTION: load kernels, load data ######
def load_spectral_kernel(cart_k: str):
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


    # mapping: filename tag → (output tag, subdirectory)
    tips = {
        "clear": ("clr", "clearsky_fluxes"),
        "cloudy":   ("cld", "allsky_fluxes"),
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
            fname = f"spectral_fluxes_kernel_longwave_{month:02d}_{tip_raw}.nc"
            fpath = os.path.join(sky_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing spectral kernel file: {fpath}")
            ds = xr.open_dataset(fpath, chunks={"freq":1, 'time':'auto'})
            # explicitly tag the month (temporary time-like dimension)
            ds = ds.expand_dims(time=[month])
            ds_months.append(ds)

        # --- concatenate months ---
        kernels = xr.concat(ds_months, dim="time")
        if kernels.sizes.get("time", 0) != 12:
            raise ValueError("Spectral kernel must have exactly 12 months")
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
                ker = ker.rename({"lev": "plev"})
            allkers[(tip_out, vna_out)] = ker

        # --- pressure levels (once is enough) ---
        if vlevs is None and "lev" in kernels.coords:
            vlevs = kernels["lev"].rename({"lev": "plev"})

    # --- save outputs ---
    ds_out = xr.Dataset()

    for (tip, vname), da in allkers.items():
        ds_out[f"{tip}_{vname}"] = da
    
    return allkers, None # no dp


def load_kernel_ERA5(cart_k, finam):
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
                ker=ker.rename({'level': 'plev'})
                vna='t'
            if vna=='wv_lw_dp':
                ker=ker.rename({'level': 'plev'})
                vna='wv_lw'
            if vna=='wv_sw_dp':
                ker=ker.rename({'level': 'plev'})
                vna='wv_sw'
                
            if tip=='clr':
                stef=ker.TOA_clr
            else:
                stef=ker.TOA_all
            allkers[(tip, vna)] = stef.assign_coords(month = np.arange(1, 13))

    
    vlevs = xr.load_dataset( cart_k+'dp_era5.nc')
    vlevs=vlevs.rename({'level': 'plev', 'latitude': 'lat', 'longitude': 'lon'})

    return allkers, vlevs.dp


def load_kernel_HUANG(cart_k, finam):
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
                if vna in ('t', 'wv_lw', 'wv_sw'):
                    ker=ker.rename({'player': 'plev'})

            allkers[(tip, vna)] = ker.assign_coords(month = np.arange(1, 13))
            if vna in ('ts', 't', 'wv_lw'):
                allkers[(tip, vna)]=allkers[(tip, vna)].lwkernel
            else:
                allkers[(tip, vna)]=allkers[(tip, vna)].swkernel

    vlevs = xr.load_dataset( cart_k + 'dp.nc')  
    vlevs=vlevs.rename({'player': 'plev'})
    
    return allkers, vlevs.dp


def load_kernel(ker, cart_k, finam=None):
    """
    Selects and loads radiative kernels from different sources: ERA5, HUANG, and SPECTRAL.

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

    finam : str
        Filename pattern used to locate the kernel files. It must include
        a formatting placeholder (e.g., `'ERA5_kernel_{}_TOA.nc'`,
        `'spectral_kernel_ste_{}.nc'`) that will be filled with the
        variable name or kernel type.

    Returns:
    --------
    allkers : dict
        A dictionary containing the kernels. Keys are: `(tip, variable)`.
        
    ### IMPROVE: return a dataset instead of a dict

    dp : xr.DataArray
        The width of the atmospheric layers in units of 100 hPa.
    """
    if ker == 'ERA5':
        return load_kernel_ERA5(cart_k, finam)

    elif ker == 'HUANG':
        return load_kernel_HUANG(cart_k, finam)

    elif ker == 'SPECTRAL':
        return load_spectral_kernel(cart_k)
    else:
        raise ValueError(f"Unsupported kernel type: {ker}")

#################################################################################################

def load_config(config_file, variable_mapping_file = None):
    """
    Loads the configuration from config_file, returns a dict. Creates output directory.
    """
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    reference_dataset = config['file_paths']['reference_dataset']
    # if not isinstance(reference_dataset, list):
    #     reference_dataset = [reference_dataset]
    # config['file_paths']['reference_dataset'] = [Path(p) for p in reference_dataset]

    experiment_dataset = config['file_paths']['experiment_dataset']
    # if not isinstance(experiment_dataset, list):
    #     experiment_dataset = [experiment_dataset]
    # config['file_paths']['experiment_dataset'] = [Path(p) for p in experiment_dataset]

    exp_name = config['exp_name']
    # if not isinstance(exp_name, list):
    #     exp_name = [exp_name]
    # config['exp_name'] = exp_name
    
    cart_out = config['file_paths'].get("output")
    ## Create dirs
    cart_out_exp = cart_out + f'/{exp_name}/'
    os.makedirs(cart_out_exp, exist_ok=True)
    config['cart_out_exp'] = cart_out_exp
    
    method = config.get("anomaly_method")
    if method is None:
        method = "climatology"
        print("The config file does not specify 'anomaly_method' (e.g., 'climatology', 'running_mean'), assuming 'climatology'.")
    
    config["anomaly_method"] = method

    #### qui mkdir e cart_out differenziate
    use_atm_mask=config.get("use_atm_mask", True)
    save_pattern = config.get("save_pattern", False)
    
    config['use_atm_mask'] = bool(use_atm_mask)
    config['save_pattern'] = bool(save_pattern)
    config['num_year_regr'] = config.get("num_year_regr", 10)

    # Read time ranges from config
    time_range_clim = config.get("time_range_clim", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    
    print(f"Time range for climatology: {time_range_clim if time_range_clim else "all"}")
    print(f"Time range for experiment: {time_range_exp if time_range_exp else "all"}")
    config['time_range_exp'] = time_range_exp
    config['time_range_clim'] = time_range_clim

    # Surface pressure management
    config['pressure_path'] = config['file_paths'].get('pressure_data', None)

    # Load variable mapping for non-CMOR names
    if variable_mapping_file is not None:
        variable_map = load_variable_mapping(variable_mapping_file, config['exp_name'])['rename_map']
        config['variable_mapping'] = variable_map
    else:
        config['variable_mapping'] = None

    return config


def preprocess_data(config_file, ker = "HUANG", raw_variables = STD_VARS_NOALB, save_remapped = True, variable_mapping_file = None):
    """
    All the preprocessing needed before calling the feedback calculations:
        - Reads experiment, control and kernels;
        - Remaps to kernel resolution;
        - Computes climatology;
        - Computes anomalies;
        - Computes width of atm layers for kernel;
    """
    
    # load config
    config = load_config(config_file, variable_mapping_file = variable_mapping_file)

    # load kernel
    kernel = Kernel(ker, config = config)
    k = kernel.kernel[('clr', 't')]

    if kernel.use_log_wv:
        variables = STD_VARS_LOGQ
    else:
        variables = STD_VARS
    
    # load picontrol (+ remap)
    print('\n -------> Loading control')
    control = Experiment('PI', config['file_paths']['reference_dataset'], remap_dir = config['cart_out_exp'] + f"remapped_{ker}/", raw_variables = raw_variables, variable_mapping = config['variable_mapping'])

    if control.check_remapped():
        control.load_remapped()
    else:
        print(f"Remapped data not found in: {control.remap_dir}, computing from raw..")
        control.load_raw()

        control.remap(target_ds = k, save_remapped = True)
    
    control.check_vars(variables = variables)
    control.vertical_interp(k)

    # load 4x (+ remap)
    print('\n -------> Loading experiment')
    experiment = Experiment('4x', config['file_paths']['experiment_dataset'], remap_dir = config['cart_out_exp'] + f"remapped_{ker}/", raw_variables = raw_variables, variable_mapping = config['variable_mapping'])
    if experiment.check_remapped():
        experiment.load_remapped()
    else:
        print(f"Remapped data not found in: {experiment.remap_dir}, computing from raw..")
        experiment.load_raw()
        experiment.remap(target_ds = k, save_remapped = True)

    experiment.check_vars(variables = variables)
    experiment.vertical_interp(k)
    experiment.check_time_range(config)

    # compute climatology and anomaly
    method = config['anomaly_method']
    print(f"\n -------> Computing {method} and anomalies")
    
    if method == 'climatology':
        control.compute_clim(time_range = config['time_range_clim'], compute = True)
        experiment.compute_anom_clim(control)
    elif method == 'running_mean':
        control.compute_runmean(window_years = config['num_running_years_trend'], time_range=config['time_range_clim'], compute = True)
        experiment.compute_anom_aligned(control)
    else:
        raise ValueError(f"Unknown anomaly method {method}")

    if ker == 'HUANG':
        print(f"\n -------> Recomputing atm dp with surface pressure\n")
        experiment.load_surf_pressure(config['pressure_path'], k)
        kernel.recompute_dp(experiment)

    print(f"\n ----------> Preprocessing complete for {config['exp_name']} <------------\n")

    ### up to this point: everything still lazy!
    return control, experiment, kernel


def main_loop(config_file, ker = 'HUANG'):
    control, experiment, kernel = preprocess_data(config_file, ker)

    ### Calc feedbacks

    return


###############################################################################################


def check_lazy_loading(ds):
    """Quick diagnostic for lazy loading status"""
    import dask.array as da
    
    print("=== Lazy Loading Status ===")
    for var in ds.data_vars:
        is_lazy = isinstance(ds[var].data, da.Array)
        status = "✓ Lazy" if is_lazy else "✗ LOADED"
        if is_lazy:
            size_mb = ds[var].nbytes / 1e6
            n_chunks = ds[var].data.npartitions
            print(f"{var:20s} {status:10s} ({size_mb:8.1f} MB, {n_chunks:4d} chunks)")
        else:
            print(f"{var:20s} {status:10s}")

    try:
        print(f"\nChunk structure:\n{ds.chunks}")
    except ValueError as err:
        print('Problem with chunks!')
        print(err)
        ds = xr.unify_chunks(ds)
        print(f"\nChunk structure:\n{ds.chunks}")

    return


def load_variable_mapping(configvar_file, dataset_type):
    """Load variable mappings for the specified dataset type from YAML."""
    with open(configvar_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get(dataset_type, {})



def check_time_axis(ds, piok):
    if len(ds["time"]) != len(piok["time"]):
        raise ValueError("Error: The 'time' columns in 'ds' and 'piok' must have the same length. To fix use variable 'time_range' of the function")
    return

def check_vertical(da):
    plev = da.coords["plev"]
    #units = plev.attrs.get("units")
 
    min_val = float(plev.min())

    # Se minimo > 50 → probabilmente in Pa → convertiamo
    if min_val > 50:
        new_plev = plev / 100.0
        new_plev.attrs["units"] = "hPa"
        print('Rewriting vertical coordinates from Pa to hPa')
        # Riassegna la coordinata convertita
        da = da.assign_coords(plev=new_plev)

    return da

def preproc(ds):
    ds = ds.assign_coords(lat = ds.lat.round(4))
    #ds = ds.assign_coords(lon = ds.lon.round(4))
    if 'lat_bnds' in ds:
        ds['lat_bnds'] = ds['lat_bnds'].round(4)
    
    return ds

######################################################################################
#### 

##tropopause computation (Reichler 2003) 
def mask_strato(ta):
    """
    Generates a mask for atmospheric temperature data based on the lapse rate threshold.
    as in (Reichler 2003) 

    Parameters:
    -----------
    ta: xarray.DataArray
    Atmospheric temperature dataset with pressure levels ('plev') as a coordinate. 

    Returns:
    --------
    mask : xarray.DataArray
        A mask array where:
        - Values are 1 where the lapse rate (`laps`) is less than or equal to -2 K/km.
        - Values are NaN elsewhere.
    """
    p=ta.plev
    n=ta.sizes['plev']
    #costanti
    g=9.81 #gravity
    cp=1005 #specific heat capacity of air at costant pressure
    R= 0.2870 #gas costant for dry air
    k=R/cp

    result=[]
    plevs=[]
    for i in range (0, n-1):
        lev1_P=(p.isel(plev=i)).item()
        lev1_T=ta.sel(plev= lev1_P)
        lev2_P=(p.isel(plev=i+1)).item()
        lev2_T=ta.sel(plev= lev2_P)
        a=(lev2_T-lev1_T)
        b=((lev2_P*100)**k -(lev1_P*100)**k) #*100 bc datas are in hPa
        c=((lev1_P*100)**k+(lev2_P*100)**k)
        d=(lev1_T+lev2_T)
        lapse=(a/b)*(c/d)*(k*g/R)
    
        plevs.append(lev1_P)
        result.append(lapse)
    lapse_da = xr.concat(result, dim="plev").assign_coords(plev=plevs)
    cond = xr.where(lapse_da.plev < 100, lapse_da <= -2, True)

    mask = cond.astype(int).cumprod("plev")
    mask = mask.where(mask == 1)

    # re-adding the last level with all zeros
    zero_slice = xr.zeros_like(ta.isel(plev=0)).assign_coords(plev=ta.plev[-1]).expand_dims('plev')
    mask = xr.concat([mask, zero_slice], dim='plev').transpose(*ta.dims)
    
    if ta.chunks:
        chunk_dic = {dim: chu for dim, chu in zip(ta.dims, ta.chunks)}
        mask = mask.chunk(chunk_dic)
        
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
        - For HUANG: [`plev`, `lat`, `lon`]
        - For ERA5: [`plev`, `month`, `lat`, `lon`]

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

    psmax = float(psye.max())
    if psmax > 2000:     # likely Pa
        psye = psye / 100.0
    psye_rg = ctl.regrid_dataset(psye, k.lat, k.lon).compute()
    wid_mask = np.empty([len(vlevs.plev)] + list(psye_rg.shape))

    for ila in range(len(psye_rg.lat)):
        for ilo in range(len(psye_rg.lon)):
            ind = np.where((psye_rg[ila, ilo].values - vlevs.plev.values) > 0)[0][0]
            wid_mask[:ind, ila, ilo] = np.nan
            wid_mask[ind, ila, ilo] = psye_rg[ila, ilo].values - vlevs.plev.values[ind]
            wid_mask[ind+1:, ila, ilo] = vlevs.dp.values[ind+1:]
        

    wid_mask = xr.DataArray(wid_mask, dims = k.dims[1:], coords = k.drop('month').coords)
    return wid_mask

###################################################

def pliq(T):
    pliq = 0.01 * np.exp(54.842763 - 6763.22 / T - 4.21 * np.log(T) + 0.000367 * T + np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))
    return pliq

def pice(T):
    pice = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T) / 100.0
    return pice

def dlnws_old(T):
    """
    Calculates 1/(dlnq/dT_1K) using Huang et al. (2017) formulas.
    """
    pliq0 = pliq(T)
    pice0 = pice(T)

    T1 = T + 1.0
    pliq1 = pliq(T1)
    pice1 = pice(T1)
    
    # Use np.where to choose between pliq and pice based on the condition T >= 273
    if isinstance(T, xr.DataArray):# and isinstance(T.data, da.core.Array):
        print('qui')
        ws = xr.where(T >= 273, pliq0, pice0)    # Dask equivalent of np.where is da.where
        ws1 = xr.where(T1 >= 273, pliq1, pice1)
    else:
        print('qua')
        ws = np.where(T >= 273, pliq0, pice0)
        ws1 = np.where(T1 >= 273, pliq1, pice1)
    
    # Calculate the inverse of the derivative dws
    dws = ws / (ws1 - ws)

    if isinstance(dws, np.ndarray):
        dws = ctl.transform_to_dataarray(T, dws, 'dlnws')
   
    return dws


def q2(e, p = 1e3):
    """
    Saturation specific humidity, given saturation vapor pressure and P (in hPa).
    """
    return 0.622*e/(p - 0.378*e)


def calc_qsat(temp, pres = 1.e3):
    """
    Computes the saturation specific humidity.
    temp in K
    pres in hPa
    """

    # Following Murphy and Koop 2005
    # ew = 0.01*np.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T + np.tanh(0.0415*(T - 218.8)) * (53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T))
    # ei = 0.01*np.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)

    # simplified formulas (WMO)
    t = temp - 273.15
    ew = 6.112 * np.exp(17.62 * t / (243.12 + t))
    ei = 6.112 * np.exp(22.46 * t / (272.62 + t))

    # use ice vs water
    if isinstance(temp, xr.DataArray):
        e = xr.where(t > 0, ew, ei)
    else:
        e = np.where(t > 0, ew, ei)

    qsat = q2(e, pres)

    return qsat


def Kq_fact(temp, method, pres = None):
    """
    Factor used to normalize the water vapor kernel, which usually corresponds to a change in specific humidity due to an increase of atm temp by 1 K, keeping RH constant.
    """
    if pres is None:
        if isinstance(temp, xr.DataArray):
            pres = temp.plev
        else:
            raise ValueError('pres not specified in input to Kq_fact')
    
    Rv = 461.5 # gas constant of water vapor
    Lv = 2.5e+06 # latent heat of water vapor

    qs0 = calc_qsat(temp, pres = pres)
    qs1K = calc_qsat(temp + 1, pres = pres)

    if method == 'log': # Huang, 2017 (HUANG)
        cos = 1/np.log(qs1K/qs0)
    elif method == 'linear': # Huang and Huang, 2023 (ERA5)
        cos = qs0/(qs1K - qs0)
    elif method == 'CC': # approximation using CC, as in Huang and Huang, eq. A6
        cos = temp**2 * Rv / Lv
    else:
        raise ValueError(f'method {method} not recognized')

    return cos


# function dlnws(T)
# begin

# pliq=0.01*exp(54.842763- 6763.22/T-4.21*log(T) + 0.000367*T+tanh(0.0415*(T-218.8))*\
#                 (53.878 -1331.22/T-9.44523*log(T) + 0.014025*T))
# pice=exp(9.550426-5723.265/T+3.53068*log(T)-0.00728332*T)/100.
# T1=T+1.
# pliq1=0.01*exp(54.842763- 6763.22 / T1-4.21*log(T1) + 0.000367*T1+tanh(0.0415*(T1 - 218.8))*\
#                 (53.878 -1331.22/T1-9.44523*log(T1) + 0.014025*T1))
# pice1=exp(9.550426-5723.265/T1+3.53068*log(T1)-0.00728332*T1)/100.

# ws=where(T.ge.273,pliq,pice)
# ws1=where(T1.ge.273,pliq1,pice1)

# dws=ws/(ws1-ws)
# return(dws)
# end


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

       
# FUNCTION FOR WV ANOMALIES
# From Mass mixing Ratio (kg/kg to ppmv)
def q_to_ppmv(q_inp):
    Ma = 28.97  # Molecular weight of dry air
    Mw = 18.02  # Molecular weight of water vapor
    vw_ppmv = q_inp / (1 - q_inp) * (Ma / Mw) * 10**6
    return vw_ppmv

############ RADIATIVE ANOMALY FUNCTIONS #############
#PLANCK SURFACE

def Rad_anomaly_planck_surf(experiment, kernel, cart_out, save_pattern=False):
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
    for tip in ['clr', 'cld']:
        print(f"Processing {tip}")  
        try:
            k = kernel.kernel[(tip, 'ts')]
            # print("Kernel loaded successfully")  
        except Exception as e:
            print(f"Error loading kernel for {tip}: {e}")  
            continue  

        dRt = (experiment.ds_anom['ts'].groupby("time.month") * k).groupby("time.year").mean("time")

        #Save full dRt pattern before global averaging
        if save_pattern: 
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} surface Planck dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_planck-surf_pattern_" + tip +".nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt)
        planck = dRt_glob.compute()
        planck.name='planck-surf'
        radiation[(tip, 'planck-surf')] = planck
        planck.to_netcdf(cart_out + "dRt_planck-surf_global_" + tip + ".nc", format="NETCDF4")
        planck.close()
        
    return radiation

#PLANK-ATMO AND LAPSE RATE WITH VARYING TROPOPAUSE

def Rad_anomaly_planck_atm_lr(experiment,  kernel, cart_out, use_strat_mask=True, save_pattern=False):

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
    radiation=dict()
    if use_strat_mask==True:
        mask = mask_strato(experiment.ds['ta'])
        ta_anom = (experiment.ds_anom['ta'] * mask).sel(plev = mask.plev)
    else:
        ta_anom = experiment.ds_anom['ta']
    
    anoms_lr = ta_anom - experiment.ds_anom['ts'] 
    anoms_unif = ta_anom - anoms_lr

    for tip in ['clr', 'cld']:
        print(f"Processing {tip}")  
        try:
            k = kernel.kernel[(tip, 't')]
            # print("Kernel loaded successfully")  
        except Exception as e:
            print(f"Error loading kernel for {tip}: {e}")  
            continue  

        if kernel.name=='SPECTRAL':
            dRt_unif = (anoms_unif.groupby('time.month')*k).sum(dim="plev").groupby("time.year").mean("time")
            dRt_lr = (anoms_lr.groupby('time.month')*k).sum(dim="plev").groupby("time.year").mean("time")
        else:
            dRt_unif = (anoms_unif.groupby('time.month') * (k * kernel.dp)).sum("plev").groupby("time.year").mean("time")
            dRt_lr = (anoms_lr.groupby('time.month') * (k * kernel.dp)).sum("plev").groupby("time.year").mean("time")


        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt_unif.name = "dRt_atmo"
            dRt_unif.attrs["description"] = f"{tip} atmosperic Planck dRt pattern"
            dRt_unif.to_netcdf(cart_out + "dRt_planck-atmo_pattern_" + tip + ".nc", format="NETCDF4")

            dRt_lr.name = "dRt_lr"
            dRt_lr.attrs["description"] = f"{tip} lapse-rate dRt pattern"
            dRt_lr.to_netcdf(cart_out + "dRt_lapse-rate_pattern_" + tip + ".nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_unif_glob = ctl.global_mean(dRt_unif)
        dRt_lr_glob = ctl.global_mean(dRt_lr)
        feedbacks_atmo = dRt_unif_glob.compute()
        feedbacks_lr = dRt_lr_glob.compute()
        feedbacks_atmo.name='planck-atmo'
        feedbacks_lr.name='lapse-rate'
        radiation[(tip,'planck-atmo')]=feedbacks_atmo
        radiation[(tip,'lapse-rate')]=feedbacks_lr 
        feedbacks_atmo.to_netcdf(cart_out+ "dRt_planck-atmo_global_" +tip + ".nc", format="NETCDF4")
        feedbacks_lr.to_netcdf(cart_out+ "dRt_lapse-rate_global_" +tip  + ".nc", format="NETCDF4")
        feedbacks_atmo.close()
        feedbacks_lr.close()

    return(radiation)

#ALBEDO

def Rad_anomaly_albedo(experiment, kernel, cart_out, save_pattern=False):
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

    if kernel.name == "SPECTRAL":
        print("Skipping albedo feedback for SPECTRAL kernels (not defined).")
        return radiation

    for tip in [ 'clr','cld']:
        k = kernel.kernel[(tip, 'alb')]
        dRt = (experiment.ds_anom['alb'].groupby("time.month") * k).groupby("time.year").mean("time")
            
        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} albedo dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_albedo_pattern_" + tip + ".nc", format="NETCDF4")

        #Then compute and save global mean
        dRt_glob = ctl.global_mean(dRt).compute()
        alb = 100*dRt_glob
        alb.name='albedo'
        radiation[(tip, 'albedo')]= alb
        alb.to_netcdf(cart_out+ "dRt_albedo_global_" +tip + ".nc", format="NETCDF4")
        alb.close()

    return(radiation)

#W-V COMPUTATION
def Rad_anomaly_wv(experiment, control, kernel, cart_out, use_strat_mask=True, save_pattern=False):
    
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
    radiation=dict()
    
    if kernel.use_log_wv:
        wv_name='hus_log'
    else:
        wv_name='hus'
        
    if use_strat_mask==True:
        mask=mask_strato(experiment.ds['ta'])
        anoms_hus= (experiment.ds_anom[wv_name] * mask).sel(plev = mask.plev)
    else:
        anoms_hus=experiment.ds_anom[wv_name]
            
    if kernel.name=='HUANG':
        dln = Kq_fact(control.ds_clim['ta'], method = 'CC')
        if 'time' in dln.dims:
            coso = anoms_hus * dln.values
        elif 'month' in dln.dims:
            coso = anoms_hus.groupby('time.month') * dln
    elif kernel.name=='ERA5':
        dq_norm = (anoms_hus.groupby('time.month') / control.ds_clim['hus'])
        print(dq_norm.shape)
        coso = dq_norm.groupby('time.month') * Kq_fact(control.ds_clim['ta'], method = 'CC')
        # coso = (anoms_hus.groupby('time.month') / control.ds_clim['hus']) * Kq_fact(control.ds_clim['ta'], method = 'linear')
    elif kernel.name == "SPECTRAL":
        coso = q_to_ppmv(anoms_hus)
    

    for tip in ['clr','cld']:
        print(f"Processing {tip}") 
        kernel_lw = kernel.kernel[(tip, 'wv_lw')]
        if kernel.name!= 'SPECTRAL':
            kernel_sw = kernel.kernel[(tip, 'wv_sw')]

        if kernel.name=='SPECTRAL':
            dRt_lw = (coso.groupby('time.month')* kernel_lw).sum('plev').groupby('time.year').mean('time')
        else:
            dRt_lw = (coso.groupby('time.month')* (kernel_lw*kernel.dp)).sum('plev').groupby('time.year').mean('time')
            dRt_sw = (coso.groupby('time.month')* (kernel_sw*kernel.dp)).sum('plev').groupby('time.year').mean('time')
            dRt = dRt_lw + dRt_sw
                
                
                
        #Save full dRt pattern before global averaging
        if save_pattern:
            dRt.name = "dRt"
            dRt.attrs["description"] = f"{tip} water vapor dRt pattern"
            dRt.to_netcdf(cart_out + "dRt_water-vapor_pattern_" + tip + ".nc", format="NETCDF4")
            if kernel.name != 'SPECTRAL':
                dRt_lw.name = "dRt_lw"
                dRt_lw.attrs["description"] = f"{tip} water vapor dRt_lw pattern"
                dRt_lw.to_netcdf(cart_out + "dRt_lw_water-vapor_pattern_" + tip +  ".nc", format="NETCDF4")
                dRt_sw.name = "dRt_sw"
                dRt_sw.attrs["description"] = f"{tip} water vapor dRt_sw pattern"
                dRt_sw.to_netcdf(cart_out + "dRt_sw_water-vapor_pattern_" + tip +  ".nc", format="NETCDF4")

        
        dRt_glob_lw = ctl.global_mean(dRt_lw)
        wv_lw= dRt_glob_lw.compute()
        wv_lw.name='water-vapor_lw'
        wv_lw.to_netcdf(cart_out+ "dRt_lw_water-vapor_global_" +tip+ ".nc", format="NETCDF4")
        radiation[(tip, 'water-vapor_lw')] = wv_lw
        if kernel.name != 'SPECTRAL':
            dRt_glob_sw = ctl.global_mean(dRt_sw)
            wv_sw= dRt_glob_sw.compute()
            wv_sw.name='water-vapor_sw'
            radiation[(tip, 'water-vapor_sw')] = wv_sw
            wv_sw.to_netcdf(cart_out+ "dRt_sw_water-vapor_global_" +tip+ ".nc", format="NETCDF4")
  
            dRt_glob = ctl.global_mean(dRt)
            wv= dRt_glob.compute()
            wv.name='water-vapor'
            radiation[(tip, 'water-vapor')] = wv
            wv.to_netcdf(cart_out+ "dRt_water-vapor_global_" + tip + ".nc", format="NETCDF4")
            wv.close()
        
    return radiation

#CLOUD ANOMALY
def Rad_anomaly_cloud(experiment, control, cart_out):
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    dRt={}
    
    crf = experiment.ds_anom['Net0'] - experiment.ds_anom['Net']

    # lat_target = np.linspace(-90, 90, 73)
    # lon_target = np.linspace(0, 357.5, 144)
    # crf = ctl.regrid_dataset(crf, lat_target, lon_target)
    crf_glob= ctl.global_mean(crf).groupby('time.year').mean('time')

    dRt = open_dRt(cart_out, names=dRt_nocloud)

    dRt_cloud= -crf_glob + sum([dRt[( 'clr', fbn)] - dRt[('cld', fbn)] for fbn in dRt_nocloud])
    cloud = dRt_cloud.compute()
    cloud.name='cloud'
    cloud.to_netcdf(cart_out + "dRt_cloud_global.nc", format="NETCDF4")
    return cloud


#ALL RAD_ANOM COMPUTATION

def calc_anoms(experiment, control, kernel, cart_out, use_strat_mask=True, save_pattern=False, force_recompute=True):


    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_ps = Rad_anomaly_planck_surf(experiment, kernel, cart_out, save_pattern)
    else:
        print(f'Reading already computed anomaly from {path}')
        anom_ps = xr.open_dataset(path)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_pal = Rad_anomaly_planck_atm_lr(experiment, kernel, cart_out, use_strat_mask, save_pattern)
    else:
        print(f'Reading already computed anomaly from {path}')
        anom_pal = xr.open_dataset(path)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_a = Rad_anomaly_albedo(experiment, kernel, cart_out, save_pattern)
    else:
        print(f'Reading already computed anomaly from {path}')
        anom_a = xr.open_dataset(path)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_wv = Rad_anomaly_wv(experiment, control, kernel, cart_out, use_strat_mask, save_pattern)
    else:
        print(f'Reading already computed anomaly from {path}')
        anom_wv = xr.open_dataset(path) 

    print('cloud')
    path = os.path.join(cart_out, "dRt_cloud_global.nc")
    if not os.path.exists(path) or force_recompute:
        anom_cloud = Rad_anomaly_cloud(experiment, control, cart_out)
    else:
        print(f'Reading already computed anomaly from {path}')
        anom_cloud = xr.open_dataset(path) 

    return anom_ps, anom_pal, anom_a, anom_wv, anom_cloud 

##FEEDBACK COMPUTATION
        
dRt_all=['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo', 'cloud']
dRt_nocloud=['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']

def open_dRt(cart_out, names=dRt_all):
    dRt= {}
    for tip in ['clr', 'cld']:
        for i in names:
            if i != 'cloud':
                dRt[(tip, i)]=xr.open_dataarray(cart_out+"dRt_" + i +"_global_"+tip+ ".nc",  decode_times=time_coder)
    if 'cloud' in names:
        dRt[('cld', 'cloud')] = xr.open_dataarray(cart_out+"dRt_cloud_global.nc",  decode_times=time_coder)
    return dRt



def calc_fb(experiment, control, kernel, cart_out, use_strat_mask=True, save_pattern=False, num=10):
   
    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_surf(experiment, kernel, cart_out, save_pattern)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr.nc")
    if not os.path.exists(path):
        Rad_anomaly_albedo(experiment, kernel, cart_out, save_pattern)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_cld.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_atm_lr(experiment, kernel, cart_out, use_strat_mask, save_pattern)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr.nc")
    if not os.path.exists(path):
        Rad_anomaly_wv(experiment, control, kernel, cart_out, use_strat_mask, save_pattern)   

    print('cloud')
    path = os.path.join(cart_out, "dRt_cloud_global.nc")
    if not os.path.exists(path):
        Rad_anomaly_cloud(experiment, control, cart_out)
    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    dRt={}
    fb_coef = dict()
    fb_pattern = dict()
    dRt=open_dRt(cart_out)

    #compute gtas

    gtas = ctl.global_mean(experiment.ds_anom['tas']).groupby('time.year').mean('time')
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
            start_year = int(dRt[(tip, fbn)].year.min())
            feedback=dRt[(tip, fbn)].groupby((dRt[(tip, fbn)].year-start_year) // num * num).mean()

            res = stats.linregress(gtas, feedback)
            fb_coef[(tip, fbn)] = res

            if save_pattern:
                print(f"Computing spatial feedback pattern for {tip}-{fbn}...")
                # Open the dRt pattern
                feedbacks_pattern = xr.open_dataarray(cart_out+"dRt_"+fbn+"_pattern_"+tip +".nc", decode_times=time_coder) 
                start_year = int(feedbacks_pattern.year.min())
                feedbacks_pattern_dec = feedbacks_pattern.groupby((feedbacks_pattern.year - start_year) // num * num).mean('year')
                feedbacks_pattern_dec = feedbacks_pattern_dec.chunk({'year': -1})
                # Perform regression at each grid point
                slope, stderr = regress_pattern_vectorized(feedbacks_pattern_dec, gtas)
                fb_pattern[(tip, fbn)] = (slope, stderr)
                slope.to_netcdf(cart_out + "feedback_pattern_"+ fbn +"_" + tip + ".nc", format="NETCDF4")
                stderr.to_netcdf(cart_out + "feedback_pattern_error_"+ fbn +"_" + tip + ".nc", format="NETCDF4")
    
    start_year = int(dRt[('cld', 'cloud')].year.min())
    feedback=dRt[('cld', 'cloud')].groupby((dRt[('cld', 'cloud')].year-start_year) // num * num).mean()
    fb_coef[('cld', 'cloud')] = stats.linregress(gtas, feedback)
    
    return {
        "fb_coeffs": fb_coef,
        "fb_pattern": fb_pattern if save_pattern else None,
    }


def calc_inter(ds, running_years):
    med = ctl.running_mean(ds, running_years)
    trend=ds-med
    return trend


def calc_fb_interannual(experiment, control, kernel, cart_out, use_strat_mask=True, save_pattern=False, running_years=25):   
   
    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_surf(experiment, kernel, cart_out, save_pattern)
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr.nc")
    if not os.path.exists(path):
        Rad_anomaly_albedo(experiment, kernel, cart_out, save_pattern)
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_cld.nc")
    if not os.path.exists(path):
        Rad_anomaly_planck_atm_lr(experiment, kernel, cart_out, use_strat_mask, save_pattern)
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr.nc")
    if not os.path.exists(path):
        Rad_anomaly_wv(experiment, control, kernel, cart_out, use_strat_mask, save_pattern) 

    print('cloud')
    path = os.path.join(cart_out, "dRt_cloud_global.nc")
    if not os.path.exists(path):
        Rad_anomaly_cloud(experiment, control, cart_out)
    
    fbnams = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    dRt={}
    fb_coef = dict()
    fb_pattern = dict()
    dRt=open_dRt(cart_out)

    #compute gtas
    gtas = ctl.global_mean(experiment.ds_anom['tas']).groupby('time.year').mean('time')
    temp= calc_inter(gtas, running_years)

    if save_pattern:
        fb_pattern = {}
    else:
        fb_pattern = None

    print('feedback calculation...')
    for tip in ['clr', 'cld']:
        for fbn in fbnams:
            inter=calc_inter(dRt[(tip, fbn)], running_years)

            res = stats.linregress(temp,inter)
            fb_coef[(tip, fbn)] = res
            if save_pattern:
                print(f"Computing spatial feedback pattern for {tip}-{fbn}...")
                # Open the dRt pattern
                feedbacks_pattern = xr.open_dataarray(cart_out+"dRt_"+fbn+"_pattern_"+tip+ ".nc", decode_times=time_coder)
                feedbacks_pattern_dec=calc_inter(feedbacks_pattern, running_years)              

                feedbacks_pattern_dec = feedbacks_pattern_dec.chunk({'year': -1})
                gtas1 = temp.chunk({'year': -1})
                # Perform regression at each grid point
                slope, stderr = regress_pattern_vectorized(feedbacks_pattern_dec, gtas1)
                fb_pattern[(tip, fbn)] = (slope, stderr)
                slope.to_netcdf(cart_out + "feedback_pattern_"+ fbn +"_" + tip + ".nc", format="NETCDF4")
                stderr.to_netcdf(cart_out + "feedback_pattern_error_"+ fbn +"_" + tip +  ".nc", format="NETCDF4")

    #cloud
    inter=calc_inter(dRt[('cld', 'cloud')], running_years)
    res = stats.linregress(temp,inter)
    fb_coef['cld', 'cloud'] = res
    
    return {
        "fb_coeffs": fb_coef,
        "fb_pattern": fb_pattern if save_pattern else None,
    }    


def single_feedback(name, experiment, kernel, cart_out, control=None, use_strat_mask=True, save_pattern=False, num=10):
    
    gtas = ctl.global_mean(experiment.ds_anom['tas']).groupby('time.year').mean('time')
    start_year = int(gtas.year.min()) 
    gtas = gtas.groupby((gtas.year-start_year) // num * num).mean()
    
    if name != 'cloud':
        path = os.path.join(cart_out, "dRt_"+name+"_global_clr.nc")
    else:
        path = os.path.join(cart_out, "dRt_"+name+"_global.nc")
    if not os.path.exists(path):
        print('Using '+name + ' radiation anomalies function')
        if name=='albedo':
            Rad_anomaly_albedo(experiment, kernel, cart_out, save_pattern)

        elif name=='planck-surf':
            Rad_anomaly_planck_surf(experiment, kernel, cart_out, save_pattern)

        elif name=='planck-atmo':
            Rad_anomaly_planck_atm_lr(experiment, kernel, cart_out, use_strat_mask, save_pattern)

        elif name=='water-vapor':
            Rad_anomaly_wv(experiment, control, kernel, cart_out, use_strat_mask, save_pattern)

        elif name=='lapse-rate':
            Rad_anomaly_planck_atm_lr(experiment, kernel, cart_out, use_strat_mask, save_pattern)
        
        elif name == 'cloud':
            Rad_anomaly_cloud(experiment, control, cart_out)
    
    fb=dict()
    if name!='cloud':
        for tip in ['clr', 'cld']:
            feedbacks=xr.open_dataarray(cart_out+"dRt_" +name+"_global_"+tip+".nc",  decode_times=time_coder)
            start_year = int(feedbacks.year.min())
            feedback=feedbacks.groupby((feedbacks.year-start_year) // num * num).mean()

            res = stats.linregress(gtas, feedback)
            fb[(tip, name)] = res
    else:
        feedbacks=xr.open_dataarray(cart_out+"dRt_" +name+"_global.nc",  decode_times=time_coder)
        start_year = int(feedbacks.year.min())
        feedback=feedbacks.groupby((feedbacks.year-start_year) // num * num).mean()
        fb = stats.linregress(gtas, feedback)

    return fb