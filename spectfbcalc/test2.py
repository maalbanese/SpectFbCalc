import spectfbcalc_lib as sfc
from climtools import climtools_lib as ctl

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

config_file='/home/rotoli/codici/SpectFbCalc/spectfbcalc/config_deb2.yaml'
#allkers=dict()
ker= 'HUANG'
standard_dict = sfc.standard_names

# allkers=sfc.load_kernel_wrapper(ker, config_file)

#ds=sfc.read_data(config_file, standard_dict)

anom=dict()
ps_files="/data-hobbes/fabiano/radiative_kernels/ps_ece3/ps_Amon_EC-Earth3_stabilization-hist-1990_r1i1p1f1_gr_199?01-199?12.nc"
surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')
cart_out="/home/rotoli/codici/test_nuovamaskpress/"
# cart_k="/data-hobbes/fabiano/radiative_kernels/Huang/toa/"
# finam= "RRTMG_{}_toa_{}_highR.nc"
#piok=pickle.load(open('/home/rotoli/codici/test_provapress/piok_huang.p', 'rb'))
anom=sfc.Rad_anomaly_wv_wrapper(config_file, ker, standard_dict)

# fb_coef=dict()
# fb_coef, fb_cloud, fb_cloud_err =sfc.calc_fb(ds, piok, ker, allkers, cart_out, surf_pressure, True, None, False, True)

# ac = (fb_coef[('cld', 'planck-surf')].slope + fb_coef[('cld', 'planck-atmo')].slope)
# data = [ac, fb_coef[('cld','lapse-rate')].slope, fb_coef[('cld', 'water-vapor')].slope, fb_coef[('cld', 'albedo')].slope, fb_cloud]
# data1 =[-3.24, -0.22,  1.72,  0.58, 0.29]
# err = [(fb_coef[('cld', 'planck-surf')].stderr + fb_coef[('cld', 'planck-atmo')].stderr), fb_coef[('cld','lapse-rate')].stderr, fb_coef[('cld', 'water-vapor')].stderr, fb_coef[('cld', 'albedo')].stderr, fb_cloud_err]
# err1 = [0.05, 0.18, 0.14, 0.09, 0.36]
# fbnams1 = ['planck', 'lapse-rate', 'water-vapor', 'albedo', 'cloud']

# fig = plt.figure(figsize=(7,5))
# offset=0.05
# plt.errorbar(range(len(fbnams1)), data, yerr=err, marker=".", linestyle= 'None', label='Huang kernels (using climatology)', color='royalblue')
# plt.errorbar([x+offset for x in range(len(fbnams1))], data1, yerr=err1, marker=".", linestyle= 'None', label='by Zelinka et al. 2020', color='navy')
# plt.xticks(range(len(fbnams1)), fbnams1)
# plt.legend(loc='lower right')
# plt.ylabel('[W $m^{-2}$ $K^{-1}$]')
# plt.savefig(cart_out +'feedback_HUANG(climatology_nuovamaskpress).pdf')