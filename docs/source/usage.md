# Usage
## Running the Notebook
This tool can be used in Jupyter Notebook or directly in VS Code.
To execute the code interactively:
1. Activate the environment:
```bash
conda activate spectfbcalc
```
2. Open Jupyter Notebook and run:
```bash
jupyter notebook
```
3. Navigate to **`test_spectfbcalc.ipynb`** and execute the cells to see an example usage.

## Initial Setup in the Notebook
Before running the core functions for calculating anomalies and feedbacks, create a **`config.yaml/`** file in the **`spectfbcalc/`** folder by copying **`config_example.yaml/`** and modifying paths and configurations based on your purpose: 
```bash
cp spectfbcalc/config_example.yaml spectfbcalc/config.yaml
```
Then the data needs to be processed trough the **`sfc.preprocess_data`** function. The user must define:
```bash
config = "path/to/config/file/config.yaml"
control, experiment, kernel = sfc.preprocess_data(config_file, 'KERNEL')
```
It may also be necessary to define the list of **`raw_variables`** if it differs from the default: **`STD_VARS_NOALB`**. The list of currently available lists is shown below: 
```bash
STD_VARS = {"hus", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_LOGQ = {"hus_log", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_NOALB = {"hus", "rlut", "rsdt", "rlutcs", "rsut", "rsutcs", "ta", "tas", "ts", "rsds", "rsus"}
STD_VARS_ECE4 = {"hus", "rlut", "rsdt", "rlntcs", "rsut", "rsntcs", "alb", "ta", "tas", "ts"}
```
Users can define a new one themselves to replace the default.
Once the pre-processing is complete, everything is ready to carry out the necessary analyses

## The core functions
The **`spectfbcalc_lib`** contains several functions for radiative feedback calculations: 
- Core functions for radiance anomaly calculation: these functions compute radiance anomalies under both clear-sky and all-sky conditions. 
- Feedback calculations: includes functions for computing the following radiative feedbacks either individually or all at once: Planck, Albedo, Lapse-rate, Water-vapor and Cloud. It is also possible to calculate inter-annual feedback.
- Feedback pattern calculation: it is possible to obtain the feedback pattern for each lat/lon point using a flag that can be enabled or disabled in the **`config.yaml`**

## What the user can do 
The user can choose what to compute according to its need:
- individual radiance variations.
- all radiance variations at once (using the **`calc_anoms`** function).
- all feedback at once (using the **`calc_fb`** or **`calc_fb_interannual`** function) or one feedback at a time using **`single_feedback`**