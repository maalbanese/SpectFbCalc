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

# cart_out = '/home/rotoli/codici/test3/'
# ctl.mkdir(cart_out)

# cart_in = '/nas/archive_CMIP6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/'
# filin_pi = cart_in + 'piControl/atmos/Amon/r1i1p1f1/{}/*nc'
# filin_4c = cart_in + 'abrupt-4xCO2/atmos/Amon/r8i1p1f1_r25/{}/*nc'
# filin_4c1 = cart_in + 'abrupt-4xCO2/atmos/Amon/r8i1p1f1/{}/*nc'
# cart_k = '/data-hobbes/fabiano/radiative_kernels/Huang/toa/'
# use_climatology=True
# fb_coef=dict()

# dir='/data-hobbes/fabiano/radiative_kernels/ps_ece3/ps_Amon_EC-Earth3_stabilization-hist-1990_r1i1p1f1_gr_199?01-199?12.nc'
# allkers=dict()
# ker= 'HUANG'
# allkers=sfc.load_kernel(ker, cart_k, cart_out)
# ds=dict()
# k=allkers[('cld', 't')]

# time_range=['2540-01-01', '2689-12-31']
# piok=dict()
# allvars= ' hus ta'.split()
# for vnams in allvars:  
#         piok[vnams]=sfc.climatology(filin_pi, allkers, vnams, time_range, False)
# print('fatto piok')

# nomi='ta tas hus'.split()
# for nom in nomi:
#  filist = glob.glob(filin_4c.format(nom))
#  filist.sort()
#  ds[nom]= xr.open_mfdataset(filist, chunks = {'time': 12}, use_cftime=True)

# print('caricari ds')
# pressure= xr.open_mfdataset(dir)

# time_range=['1850-01-01', '1999-12-31']
# sfc.Rad_anomaly_wv(ds, piok, cart_out, 'HUANG', allkers, pressure, time_range, False)

config_file='/home/rotoli/codici/SpectFbCalc/spectfbcalc/config_deb.yaml'
allkers=dict()
ker= 'HUANG'
standard_dict = sfc.standard_names

allkers=sfc.load_kernel_wrapper(ker, config_file)

ds=sfc.read_data(config_file, standard_dict)

anom=dict()
ps_files="/data-hobbes/fabiano/radiative_kernels/ps_ece3/ps_Amon_EC-Earth3_stabilization-hist-1990_r1i1p1f1_gr_199?01-199?12.nc"
surf_pressure = xr.open_mfdataset(ps_files, combine='by_coords')
cart_out="/home/rotoli/codici/test_provapress/"
cart_k="/data-hobbes/fabiano/radiative_kernels/Huang/toa/"
finam= "RRTMG_{}_toa_{}_highR.nc"
piok=pickle.load(open('/home/rotoli/codici/test_provapress/piok_huang.p', 'rb'))

#rad=dict()
#rad=sfc.calc_anoms_wrapper(config_file, ker, standard_dict)
fb_coef=dict()
fb_coef, fb_cloud, fb_cloud_err =sfc.calc_fb(ds, piok,ker, allkers, cart_out, surf_pressure, True, None, False, config_file, True)

from matplotlib import pyplot as plt
cart_out = '/home/rotoli/codici/test_provapress/'
#ac2 = (fb_coef2[('cld', 'planck-surf')].slope + fb_coef2[('cld', 'planck-atmo')].slope)
#data2 = [ac2, fb_coef2[('cld','lapse-rate')].slope, fb_coef2[('cld', 'water-vapor')].slope, fb_coef2[('cld', 'albedo')].slope, fb_cloud2]
ac = (fb_coef[('cld', 'planck-surf')].slope + fb_coef[('cld', 'planck-atmo')].slope)
data = [ac, fb_coef[('cld','lapse-rate')].slope, fb_coef[('cld', 'water-vapor')].slope, fb_coef[('cld', 'albedo')].slope, fb_cloud]
#err2 = [(fb_coef2[('cld', 'planck-surf')].stderr + fb_coef2[('cld', 'planck-atmo')].stderr), fb_coef2[('cld','lapse-rate')].stderr, fb_coef2[('cld', 'water-vapor')].stderr, fb_coef2[('cld', 'albedo')].stderr, fb_cloud_err2]
data1 =[-3.24, -0.22,  1.72,  0.58, 0.29]
err = [(fb_coef[('cld', 'planck-surf')].stderr + fb_coef[('cld', 'planck-atmo')].stderr), fb_coef[('cld','lapse-rate')].stderr, fb_coef[('cld', 'water-vapor')].stderr, fb_coef[('cld', 'albedo')].stderr, fb_cloud_err]
err1 = [0.05, 0.18, 0.14, 0.09, 0.36]
fbnams1 = ['planck', 'lapse-rate', 'water-vapor', 'albedo', 'cloud']

fig = plt.figure(figsize=(7,5))
offset=0.05
plt.errorbar(range(len(fbnams1)), data, yerr=err, marker=".", linestyle= 'None', label='Huang kernels (using climatology)', color='royalblue')
plt.errorbar([x+offset for x in range(len(fbnams1))], data1, yerr=err1, marker=".", linestyle= 'None', label='by Zelinka et al. 2020', color='navy')
#plt.errorbar([x-offset for x in range(len(fbnams1))], data2, yerr=err2, marker=".", linestyle= 'None', label='Huang kernels (using 21year mean)', color='b')
plt.xticks(range(len(fbnams1)), fbnams1)
plt.legend(loc='lower right')
plt.ylabel('[W $m^{-2}$ $K^{-1}$]')

plt.savefig(cart_out +'feedback_HUANG(climatology_provapresssup).pdf')