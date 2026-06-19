import spectfbcalc_lib as sfc

config_file = '/perm/ecme5989/SpectFbCalc/spectfbcalc/config_r8.yaml'
control, experiment, kernel = sfc.preprocess_data(config_file)
cart_out='/perm/ecme5989/new_func_prove/model_1/remapped_HUANG/'

rad=sfc.Rad_anomaly_planck_surf(experiment, kernel, cart_out )