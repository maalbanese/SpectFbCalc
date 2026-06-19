#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Radiative Anomalies and Climate Feedback Calculation Library

This module provides a comprehensive suite of tools for processing climate model 
outputs, applying radiative kernels, and computing climate feedbacks. It includes 
classes and functions for lazy loading of large NetCDF datasets, horizontal and vertical 
remapping, computation of climatologies and anomalies, and linear regression 
analysis for feedback quantification.

Supported Radiative Kernels:
    * HUANG (Huang et Huang, 2023)
    * ERA5 (Soden et al., 2008)
    * SPECTRAL

Dependencies:
    * xarray, dask, numpy, scipy, pandas
    * cdo, smmregrid, climtools
"""

##### Package imports

from __future__ import annotations
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
from typing import Literal, Any

######################################################################
### Functions 
time_coder = xr.coders.CFDatetimeCoder(use_cftime = True)

STD_VARS = {"hus", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_LOGQ = {"hus_log", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_NOALB = {"hus", "rlut", "rsdt", "rlutcs", "rsut", "rsutcs", "ta", "tas", "ts", "rsds", "rsus"}
STD_VARS_ECE4 = {"hus", "rlut", "rsdt", "rlntcs", "rsut", "rsntcs", "alb", "ta", "tas", "ts"}


def regrid(ds: xr.DataArray | xr.Dataset, target_ds: xr.Dataset | xr.DataArray) -> xr.DataArray | xr.Dataset:
    """
    Wrapper to ctl.regrid_dataset to ensure the data array does not lose its name
    during the horizontal remapping process.

    Parameters
    ----------
    ds
        The input data to be regridded.
    target_ds
        The target dataset containing the destination coordinates (`lat` and `lon`).

    Returns
    -------
    coso
        The regridded data preserving the original name.
    """
    coso = ctl.regrid_dataset(ds, target_ds.lat, target_ds.lon)
    coso.name = ds.name
    return coso


class Kernel:
    """
    Loads radiative kernels and the additional information needed for feedback calculations.  
    
    Parameters
    ----------
    name 
        Identifier of the kernel to load. Must be one of: "HUANG", "ERA5", "SPECTRAL".
    config
        Configuration dictionary loaded from the YAML file. If provided, paths are extracted from it.
    path_input
        Direct path to the kernel directory. Required if `config` is not provided.
    filename_template
        Template string for the kernel filenames (e.g., 'ERA5_kernel_{}_TOA.nc'). 
        Required if `config` is not provided.
    
    Attributes
    ----------
    name : str
        The name of the loaded kernel.
    kernel : dict or xarray.Dataset
        The actual kernel data loaded into memory.
    dp : xarray.DataArray
        The width of the atmospheric layers (in hPa or Pa depending on the kernel).
    use_log_wv : bool
        Flag indicating if the kernel requires logarithmic water vapor (e.g., True for HUANG).
    """
    def __init__(self, name: Literal['HUANG', 'ERA5', 'SPECTRAL'], config: dict[str, Any] | None = None, path_input: str | None = None, filename_template: str | None = None) -> None:
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


    def recompute_dp(self, experiment: Experiment) -> None:
        """
        Recomputes the width of atmospheric layers (dp) based on the model's 
        actual surface pressure, dynamically masking layers below the surface.

        Parameters
        ----------
        experiment 
            An instance of the Experiment class containing the model's surface pressure data.
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
    Loads and manages the data of a specific climate experiment.  
    
    Parameters
    ----------
    name
        An identifying name for the experiment (e.g. {model}_{member}_{exp_type}).
    orig_dir
        Directory containing the original raw NetCDF files.
    remap_dir
        Directory where remapped datasets are stored or will be saved.
    raw_variables
        Climate variable names tracked by this instance.
    time_chunk : int, default 20
        Chunk size for the time dimension during lazy loading.
    variable_mapping
        Dictionary mapping internal standard variables to specific model variable names.
    file_dict
        Dictionary of raw file paths. If None, it will be generated automatically.
    
    Attributes
    ----------
    ds : xarray.Dataset
        Merged dataset containing the remapped variables.
    ds_clim : xarray.Dataset
        Dataset containing the computed climatology.
    ds_anom : xarray.Dataset
        Dataset containing the computed anomalies.
    raw_data : dict
        Dictionary holding the raw xarray.DataArray objects before remapping.
    surf_pressure : xarray.DataArray
        Remapped and time-averaged surface pressure data (if loaded).
    """
    def __init__(self, name: str, orig_dir: str | Path, remap_dir: str | Path = "remapped", raw_variables: set[str] | list[str] | tuple[str] = STD_VARS_NOALB, time_chunk: int = 20, variable_mapping: dict[str, str] | None = None, file_dict: dict[str, Any] | None = None) -> None:
        self.name = name
        self.raw_variables = raw_variables
        
        self.orig_dir = Path(orig_dir)
        self.remap_dir = Path(remap_dir)

        self.remap_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = {'time': time_chunk}
        self.chunks_remap = {'time': 120} # this is for remapped data

        if variable_mapping is None:
            self.variable_mapping = {var: var for var in self.raw_variables}
        else:
            self.variable_mapping = variable_mapping

        if file_dict is None:
            self.load_file_dict()
        else:
            self.file_dict = file_dict

        self.raw_data = dict()
        self.ds = xr.Dataset()
        self.ds_anom = xr.Dataset()
        self.ds_clim = xr.Dataset()
 
    # ------------------------------------------------------------------
    # 1. Lazy loading of raw DataArrays
    # ------------------------------------------------------------------
    def load_file_dict(self) -> None:
        """
        Loads the file paths for each variable in `self.raw_variables` from `self.orig_dir` and stores them in `self.file_dict`.
        """
        file_dict = {}
        for var in self.raw_variables:
            base_folder = self.orig_dir / self.variable_mapping[var]

            pattern = f"{self.variable_mapping[var]}*.nc"
            files = sorted(base_folder.glob(pattern))
        
            if not files:
                files = sorted(base_folder.glob(f"**/{pattern}"))
            if not files:
                raise FileNotFoundError(f"No files found for variable {var}")
            
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

 
    # def load_raw(self) -> None:
    #     """
    #     Lazily open each file in *file_dict* as an xr.DataArray and store
    #     them in ``self.raw_data``.
 
    #     Parameters
    #     ----------
    #     file_dict : list of path-like
    #         Paths to NetCDF files.
    #     """
    #     print('Loading raw data...')

    #     # self.raw_data = {var: xr.open_mfdataset(self.file_dict[var], combine='by_coords', decode_times=time_coder, chunks = self.chunks, preprocess = preproc)[self.variable_mapping[var]]
    #     #     for var in self.raw_variables
    #     # }
    #     self.raw_data = {}
    #     for var in self.raw_variables:
    #         print(var)
    #         self.raw_data[var] = xr.open_mfdataset(self.file_dict[var], combine='by_coords', decode_times=time_coder, chunks = self.chunks, preprocess = preproc)[self.variable_mapping[var]]
 
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
        self.raw_data = {}
        for var in self.raw_variables:
            print(var)
            ds_tmp = xr.open_mfdataset(self.file_dict[var], combine='by_coords', decode_times=time_coder, chunks = self.chunks, preprocess = preproc)
            
            rename_coords = {}
            if 'time_counter' in ds_tmp.coords or 'time_counter' in ds_tmp.dims:
                rename_coords['time_counter'] = 'time'
            if 'pressure_levels' in ds_tmp.coords or 'pressure_levels' in ds_tmp.dims:
                rename_coords['pressure_levels'] = 'plev'
                
            if rename_coords:
                ds_tmp = ds_tmp.rename(rename_coords)

            self.raw_data[var] = ds_tmp[self.variable_mapping[var]]
    
    # ------------------------------------------------------------------
    # 2. CDO remapping to a target grid
    # ------------------------------------------------------------------
    def prepare_input_dataset(self, target_grid_ds: xr.Dataset | xr.DataArray, cdo_method: str | None = None) -> None:
        """
        Orchestrates the data availability. Checks if remapped data exists on disk.
        If not, processes it using the best tool available (smmregrid/CDO or internal remap).

        Parameters
        ----------
        target_grid_ds 
            The target dataset (the kernel) whose grid will be used for interpolation.
        cdo_method 
            If specified, forces the use of a particular CDO remapping method (e.g., "remapbil", "remapcon", "ycon", etc.) when smmregrid is used for unstructured grids. 
            If None, defaults to "ycon" for unstructured grids.
        """
        if self.check_remapped():
            self.load_remapped()
            return

        print(f"Remapped data not found in: {self.remap_dir}, computing...")
        
        try:
            first_var = next(v for v in self.file_dict if self.file_dict[v])
        except StopIteration:
            raise ValueError("No variables with files found in file_dict!")
        with xr.open_dataset(self.file_dict[first_var][0]) as ds_check:
            is_unstructured = 'cell' in ds_check.dims

        if is_unstructured:
            if cdo_method is None:
                cdo_method = "ycon"
            print(f"Unstructured grid detected, using smmregrid with method: {cdo_method}")
            self.remap_cdo(target_grid_file=target_grid_ds, method=cdo_method)
            self.load_remapped()
        else:
            self.load_raw()
            self.remap(target_ds=target_grid_ds, save_remapped=True)

    def _resolve_target_grid(self, target_grid_file: str | Path | xr.Dataset | xr.DataArray) -> str:
        """
        Resolves the target grid to a CDO-compatible string or file path.
        For xr.Dataset/DataArray (e.g. kernel), builds a CDO grid string from lat/lon coords.

        Parameters
        ----------
        target_grid_file
            The target grid specification. Can be a file path or an xarray object containing lat/lon coordinates.
        """
        import subprocess

        if isinstance(target_grid_file, (xr.Dataset, xr.DataArray)):
            # Estrai lat/lon dalla DataArray/Dataset del kernel
            if isinstance(target_grid_file, xr.DataArray):
                ds = target_grid_file.to_dataset(name=target_grid_file.name or 'data')
            else:
                ds = target_grid_file

            if 'lat' in ds.coords and 'lon' in ds.coords:
                nlat = len(ds.lat)
                nlon = len(ds.lon)
                cdo_grid = f"r{nlon}x{nlat}"
                print(f"Target grid resolved to CDO string: {cdo_grid}")
                return cdo_grid

            temp_target = self.remap_dir / "temp_target_grid.nc"
            if not temp_target.exists():
                ds.to_netcdf(temp_target, format='NETCDF4_CLASSIC')
                temp_nc3 = self.remap_dir / "temp_target_grid_nc3.nc"
                subprocess.run(
                    ["cdo", "-f", "nc", "copy", str(temp_target), str(temp_nc3)],
                    check=True
                )
                temp_target = temp_nc3
            return str(temp_target)

        else:
            return str(target_grid_file)

    def remap(self, target_ds: xr.Dataset | xr.DataArray, save_remapped: bool = False) -> None:
        """
        Interpolates the raw data to the horizontal grid of the target dataset.
 
        Parameters
        ----------
        target_ds 
            The target dataset (the kernel) whose grid will be used for interpolation.
        save_remapped 
            If True, saves the newly remapped variables to disk as NetCDF files.
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


    def remap_cdo(self, target_grid_file: str | Path | xr.Dataset | xr.DataArray, method: str = "remapbil") -> None:
        """
        Interpolate each file in *file_dict* to the grid of *target_grid_file*.
        Uses CDO for regular grids, and smmregrid for EC-Earth4 unstructured grids.
        
        Parameters
        ----------
        target_grid_file 
            NetCDF file or xarray object whose grid defines the target resolution.
        method
            CDO remapping operator: ``"remapbil"`` (default), ``"remapcon"``,
            ``"remapnn"``, etc.
        """
        from smmregrid import Regridder, cdo_generate_weights

        target_path = self._resolve_target_grid(target_grid_file)

        weights_cache = {}

        def _get_cache_key(filepath: Path) -> str:
            """removed years from filename to get a cache key for weights"""
            import re
            return re.sub(r'\d{4}', 'YYYY', filepath.name)

        for var in self.file_dict:
            if not self.file_dict[var]:
                continue

            dst_file = self.remap_dir / f"{var}_{self.name}_remapped.nc"
            if dst_file.exists():
                print(f"File already remapped: {dst_file.name}")
                continue

            tmp_files = []

            for i, src_file in enumerate(self.file_dict[var]):
                src_file = Path(src_file)
                sub_dst = (
                    self.remap_dir / f"tmp_{i}_{var}_{self.name}.nc"
                    if len(self.file_dict[var]) > 1
                    else dst_file
                )

                cache_key = _get_cache_key(src_file)

                if cache_key not in weights_cache:
                    print(f"Generating weights ({method}) for grid type: {cache_key}")
                    ds_ref = xr.open_dataset(src_file)
                    time_dim = next(
                        (d for d in ['time_counter', 'time'] if d in ds_ref.dims), None
                    )
                    source_grid = ds_ref.isel({time_dim: 0}) if time_dim else ds_ref
                    for enc_key in list(source_grid.encoding.keys()):
                        if 'time' in enc_key:
                            source_grid.encoding.pop(enc_key, None)
                    weights_cache[cache_key] = cdo_generate_weights(
                        source_grid, target_grid=target_path, method=method
                    )
                    ds_ref.close()

                ds_file = xr.open_dataset(src_file)
                regridder = Regridder(weights=weights_cache[cache_key])
                regridded_ds = regridder.regrid(ds_file)
                var_name = self.variable_mapping.get(var, var)
                if var_name in regridded_ds.data_vars:
                    ds_out = regridded_ds[[var_name]]
                else:
                    data_vars = list(regridded_ds.data_vars)
                    raise ValueError(f"Variable {var_name} not found in regridded dataset. Available: {data_vars}")

                try:
                    ds_out.to_netcdf(sub_dst)
                except Exception as e:
                    if sub_dst.exists():
                        sub_dst.unlink()
                    raise
                finally:
                    ds_file.close()
                    regridded_ds.close()

                tmp_files.append(sub_dst)

            if len(tmp_files) > 1:
                print(f"Merging chunks for {var} into {dst_file.name}")
                ds_chunked = xr.open_mfdataset(tmp_files, combine='by_coords', preprocess=preproc)
                ds_chunked.to_netcdf(dst_file)
                ds_chunked.close()
                for f in tmp_files:
                    f.unlink()
    
    def vertical_interp(self, target_ds: xr.Dataset | xr.DataArray) -> None:
        """
        Checks and converts the vertical pressure coordinates (if needed) 
        and interpolates the dataset to the vertical levels of the target kernel.

        Parameters
        ----------
        target_ds
            The target dataset (usually the kernel) containing the destination 
            pressure levels (`plev`).
        """
        print('check vertical dimension')
        self.ds = check_vertical(self.ds)
        self.ds = self.ds.interp(plev = target_ds.plev)

    # ------------------------------------------------------------------
    # 3. Load remapped data into a merged, lazy Dataset
    # ------------------------------------------------------------------
 
    def load_remapped(self) -> None:
        """
        Lazily reads all previously remapped files found in `self.remap_dir` 
        and merges them into a single xarray.Dataset stored in `self.ds`.
        """
        pattern = f"*_{self.name}_remapped.nc"
        files = sorted(self.remap_dir.glob(pattern))
        if len(files) > 0:
            self.ds = xr.open_mfdataset(files, combine = "by_coords", decode_times=time_coder, chunks = self.chunks_remap, preprocess = preproc)
        else:
            raise ValueError('No remapped dataset found on disk!')
            # Alternatively, can compute them from raw

    def check_remapped(self) -> bool:
        """
        Check if remapped files are there. If at least one file matching the pattern is found, assumes remapping has been done.
        """
        pattern = f"*_{self.name}_remapped.nc"
        files = sorted(self.remap_dir.glob(pattern))
        if len(files) > 0:
            print(f'{self.name} already remapped')
            return True
        else:
            return False


    def load_surf_pressure(self, pressure_path: str, target_ds: xr.Dataset | xr.DataArray) -> None:
        """
        Loads surface pressure data, interpolates it to the kernel grid, 
        and computes its climatology.

        Parameters
        ----------
        pressure_path
            File path pattern (e.g., glob pattern) to the surface pressure NetCDF files.
        target_ds
            The target dataset (the kernel) defining the destination grid for remapping.
        """
        if not pressure_path:
            print("No pressure path provided.")
            return

        print("Loading surface pressure data...")
        ps_files = sorted(glob.glob(pressure_path))
        if not ps_files:
            raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
        
        surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords', decode_times=time_coder, preprocess=preproc)
        ps_var_name = 'ps' if 'ps' in surf_pressure.data_vars else list(surf_pressure.data_vars)[0]
        is_ps_unstructured = 'cell' in surf_pressure.dims

        if is_ps_unstructured:
            print("Surface pressure unstructured grid detected. Using smmregrid...")
            from smmregrid import Regridder, cdo_generate_weights
            
            time_dim = next((d for d in ['time_counter', 'time'] if d in surf_pressure.dims), None)
            source_grid = surf_pressure.isel({time_dim: 0}) if time_dim else surf_pressure
            
            for enc_key in list(source_grid.encoding.keys()):
                if 'time' in enc_key:
                    source_grid.encoding.pop(enc_key, None)
            
            target_path = self._resolve_target_grid(target_ds)
            
            weights = cdo_generate_weights(source_grid, target_grid=target_path, method="ycon")
            regridder = Regridder(weights=weights)
            
            print("Remapping surface pressure time series...")
            ps_mapped_ds = regridder.regrid(surf_pressure)
            ps_mapped = ps_mapped_ds[ps_var_name]
            
        else:
            print("Surface pressure regular grid detected. Using standard regrid...")
            ps_mapped = regrid(surf_pressure[ps_var_name], target_ds)

        print("Computing climatology on remapped surface pressure...")
        psclim = ps_mapped.groupby('time.month').mean(dim='time')
        psye = psclim.mean('month')
        self.surf_pressure = psye.compute()
        print("Surface pressure successfully loaded, remapped and averaged.")
    
    def Net_TOA(self) -> None:
        """
        Computes Net TOA fluxes (all-sky and clear-sky) and adds them to the dataset.
        """
        print('Creating Net TOA variables')
        self.ds['Net'] = self.ds['rsdt'] - self.ds['rlut'] - self.ds['rsut'] #net_toa_allsky
        self.ds['Net0'] = self.ds['rsdt'] - self.ds['rlutcs'] - self.ds['rsutcs'] #net_toa_clr
 
    
    def check_albedo(self) -> None:
        """
        Checks if the albedo is loaded. If not, tries to compute it from rsus and rsds. 
        If neither is available, raises an error.
        """
        if not self.ds:
            raise ValueError('Remapped data not loaded (self.ds is empty)')
    
        if 'alb' in self.ds.data_vars:
            print('alb already in ds')
        else:
            if 'rsus' in self.ds and 'rsds' in self.ds:
                print("Computing albedo from rsus and rsds")
                albedo = self.ds['rsus']/self.ds['rsds']
                self.ds['alb'] = albedo.where(albedo > 0., 0.)

                del self.ds['rsus']
                del self.ds['rsds']
            else:
                raise ValueError('alb or rsus or rsds not found in ds! Cannot compute albedo. Vars in ds: ', self.ds.data_vars)


    def check_hus_log(self) -> None:
        """
        Checks if hus_log is in ds. If not, applies log to hus and adds it to the dataset, 
        removing the original hus. Raises an error if hus is not available.
        """
        if not self.ds:
            raise ValueError('Remapped data not loaded (self.ds is empty)')
    
        if 'hus_log' in self.ds.data_vars:
            print('hus_log already in ds')
        else:
            print('Applying log to hus')
            self.ds['hus_log'] = da.log(self.ds['hus'])
            del self.ds['hus']


    def check_vars(self, variables: set[str] | list[str] | tuple[str] = STD_VARS_LOGQ) -> None:
        """
        Checks if all variables needed are loaded. If some are missing, 
        tries to compute them from available variables (e.g., rsutcs from rsntcs and rsdt, or rlutcs from rlntcs). 
        If some variables are still missing after the computations, raises an error.

        Parameters
        ----------
        variables
            The set of variables that are required for the feedback calculations. Defaults to `STD_VARS_LOGQ`.
        """
        if not self.ds:
            raise ValueError('Remapped data not loaded (self.ds is empty)')
        
        if "rsntcs" in self.ds.data_vars and "rsutcs" not in self.ds.data_vars:
            print("Computed rsutcs from rsntcs and rsdt")
            self.ds["rsutcs"] = self.ds["rsdt"] - self.ds["rsntcs"]
            del self.ds["rsntcs"]

        if "rlntcs" in self.ds.data_vars and "rlutcs" not in self.ds.data_vars:
            print("Computed rlutcs from rlntcs")
            self.ds["rlutcs"] = -self.ds["rlntcs"]
            del self.ds["rlntcs"]

        if 'alb' in variables: self.check_albedo()
        if 'hus_log' in variables: self.check_hus_log()
    
        self.Net_TOA()

        missing_vars = []
        for var in variables:
            if var in self.ds.data_vars:
                print(f"-> {var} loaded")
            else:
                print(f"!!! {var} not found !!!")
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(
                f"Dataset missing variables: {missing_vars}"
            )

        self.variables = variables
    
    def check_time_range(self, config: dict[str, Any]) -> None:
        """
        If a time range is specified in the configuration, selects only that period from the dataset.
        
        Parameters
        ----------
        config
            Configuration dictionary loaded from the YAML file, 
            expected to contain a 'time_range_exp' key with 'start' and 'end' subkeys specifying the time slice to select.
        """
        time_range=config.get('time_range_exp')
        if time_range is not None:
            self.ds = self.ds.sel(time=slice(time_range['start'], time_range['end']))
        
    def check_coords(self) -> None:
        """
        Checks and renames coordinates to standard names (e.g., time_counter → time, pressure_levels → plev).
        """
        if not self.ds:
            raise ValueError('Remapped data not loaded (self.ds is empty)')
        
        rename_coords = {}
        if 'time_counter' in self.ds.coords or 'time_counter' in self.ds.dims:
            rename_coords['time_counter'] = 'time'
        if 'pressure_levels' in self.ds.coords or 'pressure_levels' in self.ds.dims:
            rename_coords['pressure_levels'] = 'plev'
                
        if rename_coords:
            self.ds = self.ds.rename(rename_coords)

    def compute_clim(self, time_range: dict[str, str] | None = None, compute: bool = True) -> None:
        """
        Computes climatology. If time_range is provided, select a period.

        Parameters
        ----------
        time_range
            Dictionary with 'start' and 'end' keys specifying the time slice 
            to consider before computing the mean.
        compute
            If True, explicitly computes the Dask array into memory.
        """
        self.ds_clim = compute_climatology(self.ds, time_range=time_range)

        if compute:
            self.ds_clim = self.ds_clim.compute()


    def compute_runmean(self, window_years: int = 21, time_range: dict[str, str] | None = None, compute: bool = True) -> None:
        """
        Computes the running mean of the experiment's dataset.

        Parameters
        ----------
        window_years
            Number of years used for the running mean window.
        time_range
            Dictionary with 'start' and 'end' keys specifying the time slice 
            to consider before computing the mean.
        compute
            If True, explicitly computes the Dask array into memory.
        """
        self.ds_clim = compute_running_mean(self.ds, window_years=window_years, time_range = time_range)

        if compute:
            self.ds_clim = self.ds_clim.compute()


    def compute_anom_clim(self, control: Experiment) -> None:
        """
        Removes monthly climatology from a control run.

        Parameters
        ----------
        control
            An instance of the Experiment class representing the control run, 
            which must have its climatology already computed and stored in `control.ds_clim`.
        """
        from importlib.util import find_spec
    
        if find_spec("flox") is not None:
            # Use groupby
            self.ds_anom = self.ds.groupby('time.month') - control.ds_clim
            self.ds_anom = self.ds_anom.chunk(self.ds.chunks)
        else:
            # Expand to avoid groupby
            self.ds_anom = self.ds - expand_clim(control.ds_clim, self.ds)


    def compute_anom_aligned(self, control: Experiment) -> None:
        """
        Removes climatology from a control run, aligning the time axis 
        (experiment and climatology need to have the same length!).

        Parameters
        ----------
        control
            An instance of the Experiment class representing the control run. 
            The time axis of `control.ds_clim`
        """
        check_time_axis(self.ds, control.ds_clim)
        clim_aligned = control.ds_clim.drop("time")
        clim_aligned["time"] = self.ds["time"]
        self.ds_anom = self.ds - clim_aligned
        
    
    def check_lazy_loading(self) -> None:
        """
        Checks if the dataset is lazily loaded (i.e., backed by Dask arrays) and prints a warning if it is not.
        """
        check_lazy_loading(self.ds)
        
 
    def __repr__(self) -> str:
        return (
            f"Experiment(\n"
            f"  name      = {self.name!r}\n"
            f"  variables (raw) = {self.raw_variables}\n"
            f"  variables = {getattr(self, 'variables', 'Not checked yet')}\n"
            f"  remap_dir  = {self.remap_dir}\n"
            f"  raw_data = {len(self.raw_data)} DataArray(s)\n"
            f"  ds    = {list(self.ds.data_vars)}\n"
            f"  ds_anom    = {list(self.ds_anom.data_vars)}\n"
            f"  ds_clim    = {list(self.ds_clim.data_vars)}\n"
            f")"
        )


def expand_clim(ds_clim: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
    """
    Expands a climatology along the time axis to match a target dataset.

    This avoids using `groupby`, which without Flox can create performance 
    warnings (e.g., slicing with out-of-order index generating massive chunk counts).

    Parameters
    ----------
    ds_clim
        The climatology dataset (must have a 'month' coordinate).
    ds
        The target dataset supplying the target 'time' axis.

    Returns
    -------
    expanded
        The climatology expanded along the full time axis of `ds`.
    """
    # Build a month coordinate from your data's time axis
    months = ds.time.dt.month
    # Index into the climatology and assign time as the new coordinate
    # ds_clim = ds_clim.chunk({'month': 12})
    expanded = ds_clim.sel(month=months).drop_vars("month").assign_coords(time=ds.time).chunk(ds.chunks)

    return expanded


def compute_climatology(ds: xr.Dataset, time_range: dict[str, str] | None = None) -> xr.Dataset:
    """
    Computes the monthly climatology of a dataset.

    Parameters
    ----------
    ds
        The input dataset.
    time_range
        Dictionary with 'start' and 'end' keys specifying the time slice 
        to consider before computing the mean. If None, uses the whole time axis.

    Returns
    -------
    ds_clim
        The monthly climatology dataset.
    """
    if time_range is not None:
        ds_clim = ds.sel(time=slice(time_range['start'], time_range['end']))
    else:
        ds_clim = ds

    ds_clim = ds_clim.groupby('time.month').mean()

    return ds_clim


def compute_running_mean(ds: xr.Dataset, window_years: int = 21, time_range: dict[str, str] | None = None) -> xr.Dataset:
    """
    Computes the running mean of a dataset. 
    Assumes the dataset has monthly temporal resolution.

    Parameters
    ----------
    ds
        The input dataset.
    window_years
        Number of years for the rolling window.
    time_range
        Dictionary with 'start' and 'end' keys to subset the data before computation.

    Returns
    -------
    ds_clim
        The dataset smoothed with a running mean.
    """
    if time_range is not None:
        ds_clim = ds.sel(time=slice(time_range['start'], time_range['end']))

    ds_clim = ctl.running_mean(ds_clim, window_years*12)

    return ds_clim


def compute_anomalies(exp: Experiment, control: Experiment, method: Literal["climatology", "running_mean"] = "climatology", window_years: int = 21, time_range_clim: dict[str, str] | None = None) -> None:
    """
    Computes anomalies for an experiment against a control reference.

    Parameters
    ----------
    exp
        Experiment instance (e.g., 4xCO2 run).
    control
        Experiment instance used as reference (e.g., piControl).
    method
        Method for anomaly computation:
        - "climatology" : Subtracts the monthly averaged climatology.
        - "running_mean" : Subtracts the running mean climatology.
    window_years
        Number of years used for the running mean (if method is "running_mean").
    time_range_clim
        Time slice to use for computing the control's reference climatology.
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
def load_spectral_kernel(cart_k: str) -> tuple[dict[tuple[str, str], xr.DataArray], None]:
    """
    Loads and preprocesses spectral kernels for further analysis.

    Spectral kernels are expected as monthly climatologies split into
    individual NetCDF files (01–12), for clear-sky and all-sky conditions.
    The function reconstructs a monthly kernel with dimension `month`
    (NOT `time`), ensuring compatibility with downstream anomaly computations.

    Parameters
    ----------
    cart_k
        Base path containing spectral kernel subdirectories ('clear_sky_fluxes/' 
        and 'all_sky_fluxes/').

    Returns
    -------
    allkers
        Dictionary mapping `(tip, variable)` to an xarray.DataArray with a `month` dimension.
        - `tip` ∈ {"clr", "cld"}
        - `variable` ∈ {"t", "ts", "wv_lw"}
    None
        Placeholder for the `dp` (pressure thickness) variable, which is not 
        returned by this specific loader.
    """
    tips = {
        "clear": ("clr", "clearsky_fluxes"),
        "cloudy":   ("cld", "allsky_fluxes"),
    }
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
    # ds_out = xr.Dataset()

    # for (tip, vname), da in allkers.items():
    #     ds_out[f"{tip}_{vname}"] = da
    
    return allkers, None # no dp


def load_kernel_ERA5(cart_k: str, finam: str) -> tuple[dict[tuple[str, str], xr.DataArray], xr.DataArray]:
    """
    Loads and preprocesses ERA5 radiative kernels

    Parameters
    ----------
    cart_k
        Path to the directory containing the ERA5 kernel NetCDF files.
    finam 
        Filename template, e.g. `ERA5_kernel_{}_TOA.nc`.

    Returns
    -------
    allkers
        A dictionary containing the preprocessed kernels. Keys are tuples `(tip, variable)`.
    dp
        The pressure levels difference (`dp`).

    Notes
    -----
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


def load_kernel_HUANG(cart_k: str, finam: str) -> tuple[dict[tuple[str, str], xr.DataArray], xr.DataArray]:
    """
    Loads and processes Huang (2017) climate kernel datasets.

    Parameters
    ----------
    cart_k
        Directory path containing the kernel NetCDF files.
    finam
        Filename template containing two placeholders: one for the variable 
        and one for the sky condition (e.g., 'huang_kernel_{}_{}.nc').

    Returns
    -------
    allkers
        Dictionary mapping `(tip, variable)` to an xarray.DataArray.
    dp
        The pressure thickness layers from the 'dp.nc' file.
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


def load_kernel(ker: str, cart_k: str, finam: str | None = None) -> tuple[dict[tuple[str, str], xr.DataArray], xr.DataArray | None]:
    """
    Selects and loads radiative kernels from specified sources.

    Parameters
    ----------
    ker
        Identifier of the kernel dataset ('ERA5', 'HUANG', or 'SPECTRAL').
    cart_k
        Directory path containing the kernel files.
    finam
        Filename pattern with placeholders. Required for ERA5 and HUANG.

    Returns
    -------
    allkers
        Dictionary containing the kernels.
    dp
        The width of the atmospheric layers, or None if not provided by the kernel.
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

def load_config(config_file: str | Path, variable_mapping_file: str | Path | None = None) -> dict[str, Any]:
    """
    Loads and validates the configuration from a YAML file.

    Parameters
    ----------
    config_file
        Path to the main YAML configuration file.
    variable_mapping_file
        Optional path to a YAML file mapping variable names.

    Returns
    -------
    config
        Parsed configuration dictionary.
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
    if not cart_out:
        raise ValueError("Key 'output' missing in 'file_paths' configuration.")
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
    
    config['use_atm_mask'] = bool(config.get("use_atm_mask", True))
    config['save_pattern'] = bool(config.get("save_pattern", False))
    config['num_year_regr'] = config.get("num_year_regr", 10)
    config['num_running_years_trend'] = config.get("num_running_years_trend", 21)

    # Read time ranges from config
    time_range_clim = config.get("time_range_clim", {})
    time_range_exp = config.get("time_range_exp", {})
    # Validate and clean time ranges
    time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None
    time_range_exp = time_range_exp if time_range_exp.get("start") and time_range_exp.get("end") else None
    
    print(f"Time range for climatology: {time_range_clim if time_range_clim else 'all'}")
    print(f"Time range for experiment: {time_range_exp if time_range_exp else 'all'}")
    config['time_range_exp'] = time_range_exp
    config['time_range_clim'] = time_range_clim

    # Surface pressure management
    config['pressure_path'] = config['file_paths'].get('pressure_data', None)

    # Load variable mapping for non-CMOR names
    if variable_mapping_file is not None:
        variable_map = load_variable_mapping(variable_mapping_file, config['exp_name']).get('rename_map')
        config['variable_mapping'] = variable_map
    else:
        config['variable_mapping'] = None

    return config


def preprocess_data(config_file: str | Path, ker: str = "HUANG", raw_variables: set[str] | list[str] | tuple[str] = STD_VARS_NOALB, save_remapped: bool = True, variable_mapping_file: str | Path | None = None, control_file_dict: dict | None = None, exp_file_dict: dict | None = None) -> tuple[Experiment, Experiment, Kernel]:
    """
    Orchestrates the data preparation sequence before feedback calculation.

    Steps include loading config, reading kernel/control/experiment data, 
    remapping to kernel resolution, and computing climatologies/anomalies.

    Parameters
    ----------
    config_file
        Path to the configuration YAML file.
    ker
        Kernel identifier.
    raw_variables
        Variables to extract from the raw datasets.
    save_remapped
        If True, saves remapped intermediate NetCDF files to disk.
    variable_mapping_file
        Path to a variable mapping YAML file.
    control_file_dict
        Optional pre-loaded dictionary of control files.
    exp_file_dict
        Optional pre-loaded dictionary of experiment files.

    Returns
    -------
    control
        The processed control `Experiment` object.
    experiment
        The processed forced `Experiment` object.
    kernel
        The loaded `Kernel` object.
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
    control = Experiment('PI', config['file_paths']['reference_dataset'], remap_dir = config['cart_out_exp'] + f"remapped_{ker}/", raw_variables = raw_variables, variable_mapping = config['variable_mapping'], file_dict = control_file_dict)

    control.prepare_input_dataset(target_grid_ds=k)
    control.check_coords() 
    control.check_vars(variables = variables)
    control.vertical_interp(k)

    # load 4x (+ remap)
    print('\n -------> Loading experiment')
    experiment = Experiment('4x', config['file_paths']['experiment_dataset'], remap_dir = config['cart_out_exp'] + f"remapped_{ker}/", raw_variables = raw_variables, variable_mapping = config['variable_mapping'], file_dict = exp_file_dict)
    
    experiment.prepare_input_dataset(target_grid_ds=k)
    experiment.check_coords() 
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


def check_lazy_loading(ds: xr.Dataset) -> xr.Dataset:
    """
    Checks the lazy-loading status of an xarray Dataset.
    If chunking is problematic, unifies chunks and returns the fixed dataset.

    Parameters
    ----------
    ds
        The dataset to inspect.
        
    Returns
    -------
    ds
        The original dataset, or the chunk-unified dataset if errors were caught.
    """
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

    return ds

def load_variable_mapping(configvar_file: str | Path, dataset_type: str) -> dict[str, Any]:
    """Load variable mappings for the specified dataset type from YAML."""
    with open(configvar_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get(dataset_type, {})

def check_time_axis(ds: xr.Dataset, piok: xr.Dataset) -> None:
    """Ensures two datasets share the same length along the time dimension."""
    if len(ds["time"]) != len(piok["time"]):
        raise ValueError("Error: The 'time' columns in 'ds' and 'piok' must have the same length. To fix use variable 'time_range' of the function")
    return

def check_vertical(da: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Inspects vertical coordinates and converts from Pascals to hPa if needed.
    
    Parameters
    ----------
    da
        Dataset or DataArray containing a 'plev' coordinate.

    Returns
    -------
    da
        Object with corrected 'plev' units.
    """
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

def preproc(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardizes coordinate names and rounds grid definitions to avoid float errors.
    """
    rename_coords = {}
    if 'time_counter' in ds.dims or 'time_counter' in ds.coords:
        rename_coords['time_counter'] = 'time'
    if 'pressure_levels' in ds.dims or 'pressure_levels' in ds.coords:
        rename_coords['pressure_levels'] = 'plev'
    if rename_coords:
        ds = ds.rename(rename_coords)

    ds = ds.assign_coords(lat = ds.lat.round(4))
    #ds = ds.assign_coords(lon = ds.lon.round(4))
    if 'lat_bnds' in ds:
        ds['lat_bnds'] = ds['lat_bnds'].round(4)
    
    return ds

######################################################################################

def mask_strato(ta: xr.DataArray) -> xr.DataArray:
    """
    Generates a mask for atmospheric temperature data based on the lapse rate threshold,
    following Reichler et al. (2003).

    The lapse rate $\Gamma$ is calculated between adjacent pressure levels.

    Parameters
    ----------
    ta
        Atmospheric temperature dataset with pressure levels ('plev') as a coordinate. 

    Returns
    -------
    mask
        A mask array where:
        - Values are 1 where the lapse rate is less than or equal to -2 K/km.
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

# def mask_pres(surf_pressure: xr.Dataset, cart_out: str, allkers: dict[tuple[str, str], xr.DataArray], config_file: str | None = None) -> xr.DataArray:
#     """
#     Computes a "width mask" for atmospheric pressure levels based on surface pressure and kernel data.

#     The function determines which pressure levels are above or below the surface pressure (`ps`) 
#     and generates a mask that includes NaN values for levels below the surface pressure and 
#     interpolated values near the surface.

#     Parameters
#     ----------
#     surf_pressure
#         An xarray dataset containing surface pressure (`ps`) values, or a string path to NetCDF file(s).
#     cart_out
#         Path to the directory where outputs are saved.
#     allkers
#         Dictionary containing radiative kernels for different conditions.
#     vlevs
#         The dataset or data array containing the pressure levels (`plev`) and layer thicknesses (`dp`).
#         (Must be passed in memory to avoid missing pickle file errors).
#     config_file
#         Optional path to configuration YAML file, required if `surf_pressure` is a path.

#     Returns
#     -------
#     wid_mask
#         A mask indicating the vertical pressure distribution for each grid point. 
#     """
#     # MODIFIED TO WORK BOTH WITH ARRAY AND FILE
#     k = allkers[('cld', 't')]
#     vlevs = pickle.load(open(os.path.join(cart_out, 'vlevs_HUANG.p'), 'rb'))

#     # If surf_pressure is an array:
#     if isinstance(surf_pressure, xr.Dataset):
#         psclim = surf_pressure.groupby('time.month').mean(dim='time')
#         psye = psclim['ps'].mean('month')

#     # If surf_pressure is a path, open config_file
#     elif isinstance(surf_pressure, str):
#         if pressure_path is None:
#             if config_file is None:
#                 raise ValueError("config_file must be provided when surf_pressure is a directory.")
#             with open(config_file, 'r') as f:
#                 config = yaml.safe_load(f)
#             pressure_path = config["file_paths"].get("pressure_data", None)
    
#         if not pressure_path:
#             raise ValueError("No pressure_data path specified in the configuration file, but surf_pressure was given as a path.")

#         ps_files = sorted(glob.glob(pressure_path))  
#         if not ps_files:
#             raise FileNotFoundError(f"No matching files found for pattern: {pressure_path}")

#         ps = xr.open_mfdataset(ps_files, combine='by_coords')

#         # Check that 'ps' exists
#         if 'ps' not in ps:
#             raise KeyError("The dataset does not contain the expected 'ps' variable.")

#         # Convert time variable to datetime if necessary
#         if not np.issubdtype(ps['time'].dtype, np.datetime64):
#             ps = ps.assign_coords(time=pd.to_datetime(ps['time'].values))
    
#         # Resample to monthly and calculate climatology
#         ps_monthly = ps.resample(time='M').mean()
#         psclim = ps_monthly.groupby('time.month').mean(dim='time')
#         psye = psclim['ps'].mean('month')
   
#     else:
#         raise TypeError("surf_pressure must be an xarray.Dataset or a path to NetCDF files.")

#     psmax = float(psye.max())
#     if psmax > 2000:     # likely Pa
#         psye = psye / 100.0
#     psye_rg = ctl.regrid_dataset(psye, k.lat, k.lon).compute()
#     wid_mask = np.empty([len(vlevs.plev)] + list(psye_rg.shape))

#     for ila in range(len(psye_rg.lat)):
#         for ilo in range(len(psye_rg.lon)):
#             ind = np.where((psye_rg[ila, ilo].values - vlevs.plev.values) > 0)[0][0]
#             wid_mask[:ind, ila, ilo] = np.nan
#             wid_mask[ind, ila, ilo] = psye_rg[ila, ilo].values - vlevs.plev.values[ind]
#             wid_mask[ind+1:, ila, ilo] = vlevs.dp.values[ind+1:]
        

#     wid_mask = xr.DataArray(wid_mask, dims = k.dims[1:], coords = k.drop('month').coords)
#     return wid_mask

###################################################

# def pliq(T: float | np.ndarray) -> float | np.ndarray:
#     """
#     Computes the saturation vapor pressure over liquid water.
#     """
#     pliq = 0.01 * np.exp(54.842763 - 6763.22 / T - 4.21 * np.log(T) + 0.000367 * T + np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))
#     return pliq

# def pice(T: float | np.ndarray) -> float | np.ndarray:
#     """
#     Computes the saturation vapor pressure over ice.
#     """
#     pice = np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T) / 100.0
#     return pice

# def dlnws_old(T: float | np.ndarray) -> float | np.ndarray:
#     """
#     Calculates 1/(dlnq/dT_1K) using Huang et al. (2017) formulas.
#     """
#     pliq0 = pliq(T)
#     pice0 = pice(T)

#     T1 = T + 1.0
#     pliq1 = pliq(T1)
#     pice1 = pice(T1)
    
#     # Use np.where to choose between pliq and pice based on the condition T >= 273
#     if isinstance(T, xr.DataArray):# and isinstance(T.data, da.core.Array):
#         print('qui')
#         ws = xr.where(T >= 273, pliq0, pice0)    # Dask equivalent of np.where is da.where
#         ws1 = xr.where(T1 >= 273, pliq1, pice1)
#     else:
#         print('qua')
#         ws = np.where(T >= 273, pliq0, pice0)
#         ws1 = np.where(T1 >= 273, pliq1, pice1)
    
#     # Calculate the inverse of the derivative dws
#     dws = ws / (ws1 - ws)

#     if isinstance(dws, np.ndarray):
#         dws = ctl.transform_to_dataarray(T, dws, 'dlnws')
   
#     return dws


def q2(e: float | np.ndarray | xr.DataArray, p: float | np.ndarray | xr.DataArray = 1e3) -> float | np.ndarray | xr.DataArray:
    """
    Computes saturation specific humidity given saturation vapor pressure and pressure.

    Parameters
    ----------
    e
        Saturation vapor pressure (hPa).
    p
        Atmospheric pressure (hPa). Defaults to 1000 hPa.
    """
    return 0.622*e/(p - 0.378*e)


def calc_qsat(temp: float | np.ndarray | xr.DataArray, pres: float | np.ndarray | xr.DataArray = 1.e3) -> float | np.ndarray | xr.DataArray:
    """
    Computes the saturation specific humidity using simplified WMO formulas.

    Parameters
    ----------
    temp
        Temperature (K).
    pres
        Pressure (hPa). Defaults to 1000 hPa.
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


def Kq_fact(temp: float | np.ndarray | xr.DataArray, method: str, pres: float | np.ndarray | xr.DataArray | None = None) -> float | np.ndarray | xr.DataArray:
    """
    Computes the factor used to normalize the water vapor kernel.
    
    Usually corresponds to a change in specific humidity due to an increase 
    of atmospheric temperature by 1 K, keeping relative humidity constant.

    Parameters
    ----------
    temp
        Atmospheric temperature data.
    method
        Methodology to use ('log', 'linear', or 'CC').
    pres
        Pressure levels (hPa). If None, extracted from `temp.plev`.
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
def regress_pattern_vectorized(feedback_data: xr.DataArray, gtas: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Performs a linear regression between feedback spatial data and a global 
    temperature anomaly index using `xarray.apply_ufunc` for parallelization.

    Parameters
    ----------
    feedback_data
        Feedback values with dimensions (time, lat, lon).
    gtas
        Global temperature anomaly over time with dimension (time).

    Returns
    -------
    slope_map
        The regression slope (feedback pattern) for each lat/lon point.
    stderr_map
        The standard error of the regression slope for each lat/lon point.
    """
    def linregress_1d(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
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
def q_to_ppmv(q_inp: float | np.ndarray | xr.DataArray) -> float | np.ndarray | xr.DataArray:
    """
    Converts Mass Mixing Ratio (kg/kg) to parts per million by volume (ppmv).

    Parameters
    ----------
    q_inp
        Mass mixing ratio (kg/kg).
        
    Returns
    -------
    vw_ppmv
        Mixing ratio in ppmv.
    """
    Ma = 28.97  # Molecular weight of dry air
    Mw = 18.02  # Molecular weight of water vapor
    vw_ppmv = q_inp / (1 - q_inp) * (Ma / Mw) * 10**6
    return vw_ppmv

############ RADIATIVE ANOMALY FUNCTIONS #############
#PLANCK SURFACE

def Rad_anomaly_planck_surf(experiment: Experiment, kernel: Kernel, cart_out: str, save_pattern: bool = False) -> dict[tuple[str, str], xr.DataArray]:
    """
    Compute the Planck surface radiation anomaly using climate model data and radiative kernels.

    Parameters
    ----------
    experiment
        Instance of the Experiment class containing the model datasets and anomalies.
    kernel
        Instance of the Kernel class containing the loaded radiative kernels.
    cart_out
        Output directory where results will be saved.
    save_pattern
        If True, save the full spatial anomaly patterns in addition to global means.

    Returns
    -------
    radiation
        Dictionary containing computed Planck surface radiation anomalies:
        - ('clr', 'planck-surf') : clear-sky Planck surface anomaly
        - ('cld', 'planck-surf') : all-sky Planck surface anomaly

    Saved Files
    -----------
    - dRt_planck-surf_global_{tip}.nc : 
        Global mean of the Planck surface anomaly for each condition (clear/cloudy).

    If `save_pattern` is True, also saves:
    - dRt_planck-surf_pattern_{tip}.nc : 
        Full spatial pattern of the Planck surface anomaly for each condition (clear/cloudy).
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

def Rad_anomaly_planck_atm_lr(experiment: Experiment, kernel: Kernel, cart_out: str, use_strat_mask: bool = True, save_pattern: bool = False) -> dict[tuple[str, str], xr.DataArray]:
    """
    Computes atmospheric Planck and lapse-rate radiation anomalies using climate model data and radiative kernels.

    Parameters
    ----------
    experiment
        Instance of the Experiment class containing the model datasets and anomalies.
    kernel
        Instance of the Kernel class containing the loaded radiative kernels.
    cart_out
        Output directory where results will be saved.
    use_strat_mask
        If True, apply a stratospheric mask to the temperature anomalies 
        to exclude stratospheric levels from the analysis.
    save_pattern
        If True, save the full spatial anomaly patterns in addition to global means.

    Returns
    -------
    radiation
        Dictionary containing computed Planck surface radiation anomalies:
        - ('clr', 'planck-atmo') : clear-sky Planck atmospheric anomaly
        - ('cld', 'planck-atmo') : all-sky Planck atmospheric anomaly
        - ('clr', 'lapse-rate') : clear-sky lapse-rate anomaly
        - ('cld', 'lapse-rate') : all-sky lapse-rate anomaly

    Saved Files
    -----------
    - dRt_planck-atmo_global_{tip}.nc : 
        Global mean of the Planck atmospheric anomaly for each condition (clear/cloudy).
    - dRt_lapse-rate_global_{tip}.nc : 
        Global mean of the lapse-rate anomaly for each condition (clear/cloudy).

    If `save_pattern` is True, also saves:
    - dRt_planck-atmo_pattern_{tip}.nc : 
        Full spatial pattern of the Planck atmospheric anomaly for each condition (clear/cloudy).
    - dRt_lapse-rate_pattern_{tip}.nc : 
        Full spatial pattern of the lapse-rate anomaly for each condition (clear/cloudy).
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

def Rad_anomaly_albedo(experiment: Experiment, kernel: Kernel, cart_out: str, save_pattern: bool = False) -> dict[tuple[str, str], xr.DataArray]:
    """
    Compute the albedo radiation anomaly using climate model output and radiative kernels.

    Parameters
    ----------
    experiment
        Instance of the Experiment class containing the model datasets and anomalies.
    kernel
        Instance of the Kernel class containing the loaded radiative kernels.
    cart_out
        Output directory where results will be saved.
    save_pattern
        If True, save the full spatial anomaly patterns in addition to global means.

    Returns
    -------
    radiation
        Dictionary containing computed albedo radiation anomalies:
        - ('clr', 'albedo') : clear-sky albedo anomaly
        - ('cld', 'albedo') : all-sky albedo anomaly

    Saved Files
    -----------
    - dRt_albedo_global_{tip}.nc : 
        Global mean of the albedo anomaly for each condition (clear/cloudy).

    If `save_pattern` is True, also saves:
    - dRt_albedo_pattern_{tip}.nc : 
        Full spatial pattern of the albedo anomaly for each condition (clear/cloudy).
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

def Rad_anomaly_wv(experiment: Experiment, control: Experiment, kernel: Kernel, cart_out: str, use_strat_mask: bool = True, save_pattern: bool = False) -> dict[tuple[str, str], xr.DataArray]:
    """
    Compute water vapor radiation anomalies using climate model output and radiative kernels.

    Parameters
    ----------
    experiment
        Instance of the Experiment class containing the model datasets and anomalies.
    control
        Instance of the Experiment class containing the control datasets.
    kernel : Kernel
        Instance of the Kernel class containing the loaded radiative kernels.
    cart_out
        Output directory where results will be saved.
    use_strat_mask
        If True, apply a stratospheric mask to the water vapor anomalies 
        to exclude stratospheric levels from the analysis.
    save_pattern
        If True, save the full spatial anomaly patterns in addition to global means.

    Returns
    -------
    radiation
        Dictionary containing computed water vapor radiation anomalies:
        - ('clr', 'water-vapor') : clear-sky water vapor anomaly
        - ('cld', 'water-vapor') : all-sky water vapor anomaly
    
    Saved Files
    -----------
    - dRt_water-vapor_global_{tip}.nc : 
        Global mean of the water vapor anomaly for each condition (clear/cloudy).

    If `save_pattern` is True, also saves:
    - dRt_water-vapor_pattern_{tip}.nc : 
        Full spatial pattern of the water vapor anomaly for each condition (clear/cloudy).
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

def Rad_anomaly_cloud(experiment: Experiment, cart_out: str) -> xr.DataArray:
    """
    Computes cloud radiative forcing (cloud feedback) anomalies using model output 
    and previously computed radiative anomalies for other feedbacks.

    Parameters
    ----------
    experiment
        Instance of the Experiment class containing radiative flux anomalies.
    cart_out
        Output directory where results will be saved.

    Returns
    -------
    cloud
        DataArray containing the global mean cloud radiative forcing anomaly.

    Notes
    -----
    The cloud radiative anomaly is computed as:

    .. math::

        CRF = Net_0 - Net

    and then combined with kernel-derived non-cloud contributions:

    .. math::

        dR_{cloud} = -CRF + \\sum (dR_{clr,f} - dR_{cld,f})

    where ``f`` represents the non-cloud feedback components (e.g. Planck,
    lapse-rate, water vapor, albedo, etc.).

    The final result represents the residual cloud contribution needed to close
    the top-of-atmosphere radiative budget.

    Saved Outputs
    -------------
    - dRt_cloud_global.nc
      Global mean of the cloud radiative forcing anomaly.
    """
    dRt_nocloud = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    dRt={}
    
    crf = experiment.ds_anom['Net0'] - experiment.ds_anom['Net']
    crf_glob= ctl.global_mean(crf).groupby('time.year').mean('time')

    dRt = open_dRt(cart_out, names=dRt_nocloud)

    dRt_cloud= -crf_glob + sum([dRt[( 'clr', fbn)] - dRt[('cld', fbn)] for fbn in dRt_nocloud])
    cloud = dRt_cloud.compute()
    cloud.name='cloud'
    cloud.to_netcdf(cart_out + "dRt_cloud_global.nc", format="NETCDF4")
    return cloud

#ALL RAD_ANOM COMPUTATION

def calc_anoms(experiment: Experiment, control: Experiment, kernel: Kernel, cart_out: str, use_strat_mask: bool = True, save_pattern: bool = False, force_recompute: bool = True) -> tuple[dict, dict, dict, dict, dict | xr.DataArray]:
    """
    Compute or load all radiative kernel-based anomaly components.

    This function orchestrates the computation of all major radiative feedback
    components (Planck surface, Planck atmosphere + lapse-rate, albedo, water vapor,
    and cloud) using radiative kernels. It supports caching: previously computed
    results can be read from disk unless recomputation is forced.

    Parameters
    ----------
    experiment
        Instance of the Experiment class containing the model datasets and anomalies.
    control
        Instance of the Experiment class containing the control datasets.
    kernel
        Instance of the Kernel class containing the loaded radiative kernels.
    cart_out
        Output directory where results and intermediate files will be saved.
    use_strat_mask
        If True, masks stratospheric temperature changes when computing atmospheric feedbacks.
    save_pattern
        If True, computes and saves the full spatial feedback patterns for each component.
    force_recompute
        If True, forces the recomputation of all anomalies even if cached files exist.
    
    Returns
    -------
    tuple
        A tuple containing dictionaries of computed anomalies for each feedback component:
        - anom_ps: Planck surface anomalies (clear and cloudy)
        - anom_pal: Planck atmosphere and lapse-rate anomalies (clear and cloudy)
        - anom_a: Albedo anomalies (clear and cloudy)
        - anom_wv: Water vapor anomalies (clear and cloudy)
        - anom_cloud: Cloud anomalies (global mean)

    Notes
    -----
    This function acts as a pipeline wrapper that ensures consistency across all
    radiative feedback components. Each component is computed only if missing or
    if ``force_recompute=True``.
    """
    def load_dict_anomaly(name: str) -> dict[tuple[str, str], xr.DataArray]:
        out = {}
        for tip in ['clr', 'cld']:
            p = os.path.join(cart_out, f"dRt_{name}_global_{tip}.nc")
            if os.path.exists(p):
                out[(tip, name)] = xr.open_dataarray(p, decode_times=time_coder)
        return out
    print('planck surf')
    path = os.path.join(cart_out, "dRt_planck-surf_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_ps = Rad_anomaly_planck_surf(experiment, kernel, cart_out, save_pattern)
    else:
        print(f'Reading already computed anomalies for planck-surf')
        anom_ps = load_dict_anomaly('planck-surf')
    
    print('planck atm')
    path = os.path.join(cart_out, "dRt_planck-atmo_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_pal = Rad_anomaly_planck_atm_lr(experiment, kernel, cart_out, use_strat_mask, save_pattern)
    else:
        print(f'Reading already computed anomalies for planck-atmo')
        anom_pal = load_dict_anomaly('planck-atmo')
        anom_pal.update(load_dict_anomaly('lapse-rate')) 
    
    print('albedo')
    path = os.path.join(cart_out, "dRt_albedo_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_a = Rad_anomaly_albedo(experiment, kernel, cart_out, save_pattern)
    else:
        print(f'Reading already computed anomalies for albedo')
        anom_a = load_dict_anomaly('albedo')
    
    print('w-v')
    path = os.path.join(cart_out, "dRt_water-vapor_global_clr.nc")
    if not os.path.exists(path) or force_recompute:
        anom_wv = Rad_anomaly_wv(experiment, control, kernel, cart_out, use_strat_mask, save_pattern)
    else:
        print(f'Reading already computed anomalies for water-vapor')
        anom_wv = load_dict_anomaly('water-vapor') 

    print('cloud')
    path_check = os.path.join(cart_out, "dRt_cloud_global.nc")
    path = os.path.join(cart_out, "dRt_cloud_global.nc")
    if not os.path.exists(path) or force_recompute:
        anom_cloud = Rad_anomaly_cloud(experiment, control, cart_out)
    else:
        print('Reading already computed anomaly for cloud')
        anom_cloud = xr.open_dataarray(path_check, decode_times=time_coder)

    return anom_ps, anom_pal, anom_a, anom_wv, anom_cloud 

##FEEDBACK COMPUTATION
        
dRt_all=['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo', 'cloud']
dRt_nocloud=['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']

def open_dRt(cart_out: str, names: list[str] = dRt_all) -> dict[tuple[str, str], xr.DataArray]:
    dRt= {}
    for tip in ['clr', 'cld']:
        for i in names:
            if i != 'cloud':
                dRt[(tip, i)]=xr.open_dataarray(cart_out+"dRt_" + i +"_global_"+tip+ ".nc",  decode_times=time_coder)
    if 'cloud' in names:
        dRt[('cld', 'cloud')] = xr.open_dataarray(cart_out+"dRt_cloud_global.nc",  decode_times=time_coder)
    return dRt

def calc_fb(experiment: Experiment, control: Experiment, kernel: Kernel, cart_out: str, use_strat_mask: bool = True, save_pattern: bool = False, num: int = 10) -> dict[str, Any]:
    """
    Compute full radiative feedback decomposition and interannual regression
    against global mean surface temperature.

    Parameters
    ----------
    experiment
        Instance containing the model datasets and anomalies.
    control
        Instance containing the control run datasets.
    kernel
        Instance containing the loaded radiative kernels.
    cart_out
        Output directory where results and intermediate files will be saved.
    use_strat_mask
        If True, masks stratospheric temperature changes when computing atmospheric feedbacks.
    save_pattern
        If True, computes and saves the full spatial feedback patterns.
    num
        Number of years per chunk for temporal averaging (default is 10, decadal).

    Returns
    -------
    dict
        A dictionary containing:
        - "fb_coeffs": A dictionary mapping `(sky_condition, feedback_name)` to SciPy linear regression results.
        - "fb_pattern": A dictionary of spatial feedback patterns (slope and standard error) if `save_pattern` is True.
    """
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
            # mask = np.isfinite(gtas.values) & np.isfinite(feedback.values)
            # res = stats.linregress(gtas.values[mask], feedback.values[mask])

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

    # mask = np.isfinite(gtas.values) & np.isfinite(feedback.values)
    # fb_coef[('cld', 'cloud')] = stats.linregress(gtas.values[mask], feedback.values[mask])
    
    return {
        "fb_coeffs": fb_coef,
        "fb_pattern": fb_pattern if save_pattern else None,
    }

def calc_inter(ds: xr.DataArray, running_years: int) -> xr.DataArray:
    """
    Computes the interannual variability (trend) by subtracting the running mean from the dataset.

    Parameters
    ----------
    ds
        The input dataset or data array.
    running_years
        Number of years used for the running mean window.

    Returns
    -------
    trend
        The interannual anomaly (original data minus its running mean).
    """
    med = ctl.running_mean(ds, running_years)
    trend=ds-med
    return trend

def calc_fb_interannual(experiment: Experiment, control: Experiment, kernel: Kernel, cart_out: str, use_strat_mask: bool = True, save_pattern: bool = False, running_years: int = 25) -> dict[str, Any]:    
    """
    Compute interannual radiative feedback coefficients using kernel-based anomalies
    and global temperature variability.
    If required anomaly components are missing, they are computed automatically.

    Parameters
    ----------
    experiment
        Instance containing the model datasets and anomalies.
    control
        Instance containing the control run datasets.
    kernel
        Instance containing the loaded radiative kernels.
    cart_out
        Output directory where results and intermediate files will be saved.
    use_strat_mask
        If True, masks stratospheric temperature changes.
    save_pattern
        If True, computes and saves the full spatial feedback patterns.
    running_years
        Window size (in years) used to compute the running mean.

    Returns
    -------
    dict
        A dictionary containing:
        - "fb_coeffs": A dictionary mapping `(sky_condition, feedback_name)` to SciPy linear regression results.
        - "fb_pattern": A dictionary of spatial feedback patterns (slope and standard error) if `save_pattern` is True.
    """
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
            # mask = np.isfinite(temp.values) & np.isfinite(inter.values)
            # res = stats.linregress(temp.values[mask], inter.values[mask])

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
    # mask = np.isfinite(temp.values) & np.isfinite(inter.values)
    # res = stats.linregress(temp.values[mask], inter.values[mask])

    fb_coef['cld', 'cloud'] = res
    
    return {
        "fb_coeffs": fb_coef,
        "fb_pattern": fb_pattern if save_pattern else None,
    }    

def single_feedback(name: str, experiment: Experiment, kernel: Kernel, cart_out: str, control: Experiment = None, use_strat_mask: bool = True, save_pattern: bool = False, num: int = 10) -> dict[str, Any]:
    """
    Computes and extracts the linear regression feedback for a single specific variable. 
    This function extracts or computes a radiative feedback component,
    aggregates it into multi-year bins, and estimates its sensitivity to global
    mean surface temperature via linear regression.

    Parameters
    ----------
    name
        The name of the feedback to compute (e.g., 'albedo', 'cloud', 'water-vapor').
    experiment
        Instance containing the model datasets and anomalies.
    kernel
        Instance containing the loaded radiative kernels.
    cart_out
        Output directory where results and intermediate files will be saved.
    control
        Instance containing the control run datasets (required for 'water-vapor' and 'cloud').
    use_strat_mask
        If True, masks stratospheric temperature changes.
    save_pattern
        If True, saves the spatial anomaly patterns during anomaly computation.
    num
        Number of years per chunk for temporal averaging.

    Returns
    -------
    fb
        A dictionary mapping the `(sky_condition, feedback_name)` to SciPy linear regression results.
    
    Notes
    -----
    The function performs the following steps:

    1. Computes or loads the requested radiative anomaly field.
    2. Aggregates both temperature and feedback into multi-year bins.
    3. Performs linear regression:

    .. math::

        \\lambda = \\frac{dR}{dT}

    where:
        - ``dR`` is the radiative anomaly
        - ``dT`` is global mean surface temperature anomaly

    Cloud feedback is treated separately because it does not follow the same
    clear-sky / all-sky decomposition structure as other components.
    """
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
            # mask = np.isfinite(gtas.values) & np.isfinite(feedback.values)
            # res = stats.linregress(gtas.values[mask], feedback.values[mask])

            fb[(tip, name)] = res
    else:
        feedbacks=xr.open_dataarray(cart_out+"dRt_" +name+"_global.nc",  decode_times=time_coder)
        start_year = int(feedbacks.year.min())
        feedback=feedbacks.groupby((feedbacks.year-start_year) // num * num).mean()
        fb = stats.linregress(gtas, feedback)

        # mask = np.isfinite(gtas.values) & np.isfinite(feedback.values)
        # fb[('cld', 'cloud')] = stats.linregress(gtas.values[mask], feedback.values[mask])

    return fb