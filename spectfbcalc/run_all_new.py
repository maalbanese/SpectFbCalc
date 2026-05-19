import spectfbcalc_lib as sfc
from matplotlib import pyplot as plt
import xarray as xr
import yaml
import glob
from climtools import climtools_lib as ctl
import numpy as np

#####################################
# Read config

config_file = 'config_zelinka.yml'
variable_mapping_file = 'configvariable.yaml'
ker= 'HUANG'

allkers=dict()
allkers=sfc.load_kernel_wrapper(ker, config_file)
k = allkers[('cld', 't')]

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

#############################################
## Load data

# Load Picontrol
orig_dir = '/ec/res4/scratch/ccff/tunecs/picontrol_cmip6_r1_ok/'
picont = sfc.Experiment('PI', orig_dir)
# picont.load_raw()
# picont.remap(target_ds = k, save_remapped = True)
picont.load_remapped()
picont.check_vars()
picont.vertical_interp(k)

# Load 4x
exp_dir = '/ec/res4/scratch/ccff/tunecs/abrupt_cmip6_r8/'
abrupt = sfc.Experiment('4x', exp_dir)
# abrupt.load_raw()
# abrupt.remap(target_ds = k, save_remapped = True)
abrupt.load_remapped()
abrupt.check_vars()
abrupt.vertical_interp(k)

# Compute climatology
time_range_clim = config.get("time_range", {})
time_range_clim = time_range_clim if time_range_clim.get("start") and time_range_clim.get("end") else None

picont.compute_clim(time_range=config['time_range'], compute = True)

# Compute anomaly
abrupt.compute_anom_clim(picont)

###########################################################################
# Read kernels, surf_pressure
cart_k = '/perm/ccff/radiative_kernels/Huang/toa/'

k = allkers[('cld', 't')]
vlevs = xr.load_dataset( cart_k + 'dp.nc')  
vlevs=vlevs.rename({'player': 'plev'})

pressure_path = config['file_paths'].get('pressure_data', None)

if pressure_path:  # If pressure data is specified, load it
    print("Loading surface pressure data...")
    ps_files = sorted(glob.glob(pressure_path))  # Support for patterns like "*.nc"
    if not ps_files:
        raise FileNotFoundError(f"No matching pressure files found for pattern: {pressure_path}")
    
    surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')

psclim = surf_pressure.groupby('time.month').mean(dim='time')
psye = psclim['ps'].mean('month')

psye_rg = ctl.regrid_dataset(psye, k.lat, k.lon).compute()

# Surface mask
wid_mask = np.empty([len(vlevs.plev)] + list(psye_rg.shape))

for ila in range(len(psye_rg.lat)):
    for ilo in range(len(psye_rg.lon)):
        ind = np.where((psye_rg[ila, ilo].values - vlevs.plev.values) > 0)[0][0]
        wid_mask[:ind, ila, ilo] = np.nan
        wid_mask[ind, ila, ilo] = psye_rg[ila, ilo].values - vlevs.plev.values[ind]
        wid_mask[ind+1:, ila, ilo] = vlevs.dp.values[ind+1:]
    

wid_mask = xr.DataArray(wid_mask, dims = k.dims[1:], coords = k.drop('month').coords)

# Strato mask
mask = sfc.mask_strato(abrupt.ds.ta)

########
tip = 'cld'
kernel_lw = allkers[(tip, 'wv_lw')]
kernel_sw = allkers[(tip, 'wv_sw')]
kernel = kernel_lw + kernel_sw

coso3 = (abrupt.ds_anom.hus_log * mask).groupby('time.month') * sfc.dlnws(picont.ds_clim.ta).sel(plev = mask.plev) * kernel * wid_mask/100

fin = coso3.sum('plev').groupby('time.year').mean('time')

dRt_glob = ctl.global_mean(fin)
wv = dRt_glob.compute()

print(wv.mean())
print('done!')