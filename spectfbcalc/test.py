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

cart_out = '/home/rotoli/codici/test3/'
ctl.mkdir(cart_out)

cart_in = '/nas/archive_CMIP6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/'
filin_pi = cart_in + 'piControl/atmos/Amon/r1i1p1f1/{}/*nc'
filin_4c = cart_in + 'abrupt-4xCO2/atmos/Amon/r8i1p1f1_r25/{}/*nc'
filin_4c1 = cart_in + 'abrupt-4xCO2/atmos/Amon/r8i1p1f1/{}/*nc'
cart_k = '/data-hobbes/fabiano/radiative_kernels/Huang/toa/'
use_climatology=True
fb_coef=dict()

dir='/data-hobbes/fabiano/radiative_kernels/ps_ece3/ps_Amon_EC-Earth3_stabilization-hist-1990_r1i1p1f1_gr_199?01-199?12.nc'
allkers=dict()
ker= 'HUANG'
allkers=sfc.load_kernel(ker, cart_k, cart_out)
ds=dict()
k=allkers[('cld', 't')]

time_range=['2540-01-01', '2639-12-31']
piok=dict()
allvars= ' hus ta'.split()
for vnams in allvars:  
        piok[vnams]=sfc.climatology(filin_pi, allkers, vnams, time_range, False)
print('fatto piok')

nomi='ta tas hus'.split()
for nom in nomi:
 filist = glob.glob(filin_4c.format(nom))
 filist.sort()
 ds[nom]= xr.open_mfdataset(filist, chunks = {'time': 12}, use_cftime=True)

print('caricari ds')
pressure= xr.open_mfdataset(dir)

time_range=['1850-01-01', '1949-12-31']
sfc.Rad_anomaly_wv(ds, piok, cart_out, 'HUANG', allkers, pressure, time_range, False)