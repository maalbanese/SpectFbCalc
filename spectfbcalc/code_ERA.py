import spectfbcalc_lib as sfc

cart_out="/perm/ecme5989/wv_ERA/70-84/"
config='/perm/ecme5989/SpectFbCalc/spectfbcalc/config_r8.yaml'
variable_map="/perm/ecme5989/SpectFbCalc/spectfbcalc/configvariable.yaml"
ker='ERA5'
print('vado')
print(cart_out)
#rad_a=sfc.Rad_anomaly_planck_atm_lr_wrapper(config, ker, variable_map)
rad=sfc.Rad_anomaly_wv_wrapper(config, ker, variable_map)
