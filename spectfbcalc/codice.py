import spectfbcalc_lib as sfc
from climtools import climtools_lib as ctl

cart_out = '/home/rotoli/codici/test/'
ctl.mkdir(cart_out)

cart_in = '/nas/archive_CMIP6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/'
filin_pi = cart_in + 'piControl/atmos/Amon/r1i1p1f1/{}/*nc'
filin_4c = cart_in + 'abrupt-4xCO2/atmos/Amon/r8i1p1f1_r25/{}/*nc'
filin_4c1 = cart_in + 'abrupt-4xCO2/atmos/Amon/r8i1p1f1/{}/*nc'

fb_coef=dict()
fb_coef2=dict()

dir='/data-hobbes/fabiano/radiative_kernels/ps_ece3/ps_Amon_EC-Earth3_stabilization-hist-1990_r1i1p1f1_gr_199?01-199?12.nc'

ker= 'ERA5'
cart_k="/data-hobbes/fabiano/radiative_kernels/Huang_ERA5/"

fb_coef=sfc.calc_fb(filin_4c, filin_pi, cart_out, ker, cart_k, dir, True, 12)

fb_coef2=sfc.calc_fb(filin_4c, filin_pi, cart_out, ker, cart_k, dir, False, 12)

from matplotlib import pyplot as plt

ac2 = (fb_coef2[('cld', 'planck-surf')].slope + fb_coef2[('cld', 'planck-atmo')].slope)
data2 = [ac2, fb_coef2[('cld','lapse-rate')].slope, fb_coef2[('cld', 'water-vapor')].slope, fb_coef2[('cld', 'albedo')].slope]
ac = (fb_coef[('cld', 'planck-surf')].slope + fb_coef[('cld', 'planck-atmo')].slope)
data = [ac, fb_coef[('cld','lapse-rate')].slope, fb_coef[('cld', 'water-vapor')].slope, fb_coef[('cld', 'albedo')].slope]
err2 = [(fb_coef2[('cld', 'planck-surf')].stderr + fb_coef2[('cld', 'planck-atmo')].stderr), fb_coef2[('cld','lapse-rate')].stderr, fb_coef2[('cld', 'water-vapor')].stderr, fb_coef2[('cld', 'albedo')].stderr]
data1 =[-3.24, -0.22,  1.72,  0.58]
err = [(fb_coef[('cld', 'planck-surf')].stderr + fb_coef[('cld', 'planck-atmo')].stderr), fb_coef[('cld','lapse-rate')].stderr, fb_coef[('cld', 'water-vapor')].stderr, fb_coef[('cld', 'albedo')].stderr]
err1 = [0.05, 0.18, 0.14, 0.09]
fbnams1 = ['planck', 'lapse-rate', 'water-vapor', 'albedo']

fig = plt.figure(figsize=(7,5))
offset=0.05
plt.errorbar(range(len(fbnams1)), data, yerr=err, marker=".", linestyle= 'None', label='climatology', color='royalblue')
plt.errorbar([x+offset for x in range(len(fbnams1))], data1, yerr=err1, marker=".", linestyle= 'None', label='by Zelinka et al. 2020', color='navy')
plt.errorbar([x-offset for x in range(len(fbnams1))], data2, yerr=err2, marker=".", linestyle= 'None', label='21yearmean', color='b')
plt.xticks(range(len(fbnams1)), fbnams1)
plt.legend(loc='upper left')
plt.ylabel('[W $m^{-2}$ $K^{-1}$]')
#plt.title('Climate Feedback')

plt.savefig(cart_out +'feedback_ERA5(climatologyvs21year).pdf')
