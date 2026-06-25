# %%
import spectfbcalc_lib as sfc
from matplotlib import pyplot as plt
import xarray as xr
import yaml
import glob
from climtools import climtools_lib as ctl
import numpy as np

import warnings
warnings.filterwarnings('once')

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# Dask will automatically submit SLURM jobs for you
cluster = SLURMCluster(
    cores=4,
    memory="128GB",
    processes=4,
    walltime="02:00:00",
    #qos="np",
    #account='spitfabi',
    #interface='ib0'  # or 'eth0', depends on your HPC
    job_extra_directives=[
        "--account=spitfabi",
        "--qos=np"
        # "--constraint=haswell",
        # "--exclusive",
        # "--mail-type=END,FAIL",
        # "--mail-user=your.email@domain.com"
    ]
)

# Scale to desired number of workers
cluster.scale(jobs=1)  # This submits 4 SLURM jobs

# Connect client
client = Client(cluster)

config_file = 'config_zelinka.yml'
control, experiment, kernel = sfc.preprocess_data(config_file, ker = 'SPECTRAL')

cart_out = '/perm/ccff/lavori/tunecs/fbcalc/ece3_abrupt/fb_SPECTRAL/'
anoms = sfc.calc_anoms(experiment, control, kernel, cart_out, use_strat_mask=False, save_pattern=False, force_recompute=True)
