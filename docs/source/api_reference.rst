API Reference
=============

This section provides a detailed reference for SpectFbCalc's Application Programming Interface (API). 
It describes the core classes and functions intended for user interaction.

Core Classes
------------
These classes manage the data loading, remapping, and preprocessing workflows.

.. autoclass:: spectfbcalc_lib.Experiment
    :members: load_raw, remap, load_remapped, compute_clim, compute_runmean, compute_anom_clim
    :undoc-members:
    :show-inheritance:

.. autoclass:: spectfbcalc_lib.Kernel
    :members:
    :undoc-members:
    :show-inheritance:

Configuration & Preprocessing
-----------------------------
Functions to orchestrate the pipeline before the actual feedback calculations.

.. autofunction:: spectfbcalc_lib.load_config
.. autofunction:: spectfbcalc_lib.load_kernel
.. autofunction:: spectfbcalc_lib.preprocess_data

Feedback & Anomaly Calculation
------------------------------
Core functions to calculate radiative anomalies and climate feedbacks.

.. autofunction:: spectfbcalc_lib.calc_anoms
.. autofunction:: spectfbcalc_lib.calc_fb
.. autofunction:: spectfbcalc_lib.calc_fb_interannual
.. autofunction:: spectfbcalc_lib.single_feedback

Radiative Anomalies (Individual)
--------------------------------
Functions to compute specific radiative anomalies.

.. autofunction:: spectfbcalc_lib.Rad_anomaly_planck_surf
.. autofunction:: spectfbcalc_lib.Rad_anomaly_planck_atm_lr
.. autofunction:: spectfbcalc_lib.Rad_anomaly_albedo
.. autofunction:: spectfbcalc_lib.Rad_anomaly_wv
.. autofunction:: spectfbcalc_lib.Rad_anomaly_cloud