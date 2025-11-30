import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import re
import glob
import fnmatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.backends.backend_pdf import PdfPages

# ---------- Saving ouput -------------
def save_feedback_output(output, out_path_txt, out_path_nc=None):
    """
    Save feedback regression results to text and NetCDF files.

    The function writes global feedback coefficients (with errors) to a .txt file,
    and optionally saves spatial patterns of feedback slopes and errors to a .nc file.

    Parameters
    ----------
    output : tuple
        Tuple containing:
        (fb_coeffs, fb_cloud, fb_cloud_err, fb_pattern, fb_cloud_pattern).
    out_path_txt : str
        Path to the output .txt file (always required).
    out_path_nc : str, optional
        Path to the output .nc file. Required if feedback patterns are included.
    """
    if not isinstance(output, tuple) or len(output) < 3:
        raise ValueError("Expected output to be a tuple with at least 3 elements.")

    fb_coeffs = output[0]
    fb_cloud = output[1]
    fb_cloud_err = output[2]
    fb_pattern = output[3] if len(output) > 3 else None
    fb_cloud_pattern = output[4] if len(output) > 4 else None

    # ---------- TXT output ----------
    if out_path_txt:
        with open(out_path_txt, "w") as f:
            def write_block(name, results_dict):
                f.write(f"{name} feedback:\n")
                for key in ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']:
                    result = results_dict.get((name, key))
                    if result is not None:
                        f.write(f"{key.replace('_', '-') + ' feedback'}: {result.slope:.4f}\n")
                        f.write(f"{key.replace('_', '-') + ' feedback error'}: {result.stderr:.4f}\n")

            write_block("cld", fb_coeffs)
            write_block("clr", fb_coeffs)
            f.write(f"cloud feedback: {fb_cloud:.4f}\n")
            f.write(f"cloud feedback error: {fb_cloud_err:.4f}\n")

    print(f"Saved feedback coefficients to {out_path_txt}")

    # ---------- NetCDF output ----------
    if out_path_nc and (fb_pattern or fb_cloud_pattern):
        # Dummy lat/lon if not available in pattern
        lat = np.linspace(-90, 90, 73)
        lon = np.linspace(0, 360, 144, endpoint=False)

        data_vars = {}

        # Add standard component patterns
        if fb_pattern:
            for (cloud_type, component), (slope, stderr) in fb_pattern.items():
                key_slope = f"{cloud_type}_{component}_slope"
                key_stderr = f"{cloud_type}_{component}_stderr"
                safe_key_slope = key_slope.replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace(" ", "_")
                data_vars[safe_key_slope] = (["lat", "lon"], slope.data if hasattr(slope, "data") else slope)
                safe_key_stderr = key_stderr.replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace(" ", "_")
                data_vars[safe_key_stderr] = (["lat", "lon"], stderr.data if hasattr(stderr, "data") else stderr)

        # Add cloud feedback spatial pattern
        if fb_cloud_pattern:
            for key, (slope, stderr) in fb_cloud_pattern.items():
                # If key is a tuple, process normally
                if isinstance(key, tuple):
                    cloud_type, comp = key
                    if comp == "cloud":
                        new_key = f"{cloud_type}_cloud"
                    else:
                        new_key = f"{cloud_type}_{comp}"
                else:
                    new_key = key
                key_slope = f"{new_key}_slope"
                key_stderr = f"{new_key}_stderr"
                safe_key_slope = key_slope.replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace(" ", "_")
                data_vars[safe_key_slope] = (["lat", "lon"], slope.data if hasattr(slope, "data") else slope)
                safe_key_stderr = key_stderr.replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace(" ", "_")
                data_vars[safe_key_stderr] = (["lat", "lon"], stderr.data if hasattr(stderr, "data") else stderr)

        ds = xr.Dataset(data_vars=data_vars, coords={"lat": lat, "lon": lon})
        ds.to_netcdf(out_path_nc)
        print(f"Saved feedback spatial patterns to {out_path_nc}")

# -------- Spatial pattern plot -----------
def plot_fb_pattern(slope, stderr, title, output_folder, filename_prefix="fb_pattern", pdf=None):
    """
    Plot spatial patterns of feedback slopes and standard errors.

    Creates two global maps (one for slope, one for stderr) using cartopy with coastlines, 
    borders, and gridlines. Each map is saved as a PNG, and optionally appended to a 
    combined multi-page PDF.

    Parameters
    ----------
    slope : xarray.DataArray
        2D field of feedback slopes (W/m²/K).
    stderr : xarray.DataArray
        2D field of feedback standard errors (W/m²/K).
    title : str
        Plot title applied to both maps.
    output_folder : str
        Directory where PNG files will be saved (created if missing).
    filename_prefix : str, optional
        Prefix for the output PNG filenames (default: "fb_pattern").
    pdf : PdfPages, optional
        If provided, figures are also added as pages to the given PDF object.
    """
    os.makedirs(output_folder, exist_ok=True)

    def plot_field(field, cmap, label, fname):
        fig, ax = plt.subplots(figsize=(10, 4.5),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        # just plot normally, without add_label
        im = field.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            robust=True,
            cbar_kwargs={"label": label}
        )

        # Continents & borders
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=-1)

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        # Set title manually
        ax.set_title(title, fontsize=16, weight="bold")

        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, fname), dpi=300)
        if pdf:
            pdf.savefig(fig)
        plt.close(fig)

    plot_field(slope, "RdBu_r", "W/m²/K", f"{filename_prefix}_slope.png")
    plot_field(stderr, "viridis", "W/m²/K", f"{filename_prefix}_stderr.png")

def save_all_fb_patterns_to_pdf(ds: xr.Dataset, output_folder: str, pdf_name: str = "all_fb_patterns.pdf", components: list = None, skies: list = None, plot_function=None, run_label: str = "exp"):
    """
    Generate and save feedback pattern maps for multiple components and sky conditions.

    For each (component, sky) pair in the dataset, this function creates slope and stderr
    maps using the provided plotting function. Each map is saved as an individual PNG and 
    appended to a combined multi-page PDF.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing feedback fields, with variables named like "<sky>_<component>_slope"
        and "<sky>_<component>_stderr".
    output_folder : str
        Directory where PNG files and the combined PDF will be saved.
    pdf_name : str, optional
        Name of the combined PDF file (default: "all_fb_patterns.pdf").
    components : list, optional
        List of feedback components to plot (default: 
        ["albedo", "water-vapor", "lapse-rate", "planck-atmo", "planck-surf", "cloud"]).
    skies : list, optional
        List of sky conditions to plot (default: ["clr", "cld"]).
    plot_function : callable
        Function used to create each map. Must accept slope, stderr, title, output_folder,
        filename_prefix, and pdf as arguments.
    run_label : str, optional
        Label for the experiment, added to plot titles (e.g., "cold", "warm").

    Output
    ------
    - Saves one PNG per (component, sky) pair in `output_folder`.
    - Saves a multi-page PDF combining all maps into `output_folder/pdf_name`.
    """
    if components is None:
        components = ["albedo", "water-vapor", "lapse-rate", "planck-atmo", "planck-surf", "cloud"]
    if skies is None:
        skies = ["clr", "cld"]

    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, pdf_name)

    with PdfPages(pdf_path) as pdf:
        for comp in components:
            for sky in skies:
                var_slope = f"{sky}_{comp}_slope"
                var_stderr = f"{sky}_{comp}_stderr"

                if var_slope not in ds or var_stderr not in ds:
                    print(f"⚠️  {var_slope} or {var_stderr} not found, skipping.")
                    continue

                slope = ds[var_slope]
                stderr = ds[var_stderr]

                # ✅ build title with your logic
                feedback_name = comp.capitalize()
                cloud_label = sky.upper() if sky == "cld" else sky
                title = f"{feedback_name} ({cloud_label}) - {run_label}"

                filename_prefix = f"fb_pattern_{sky}_{comp}"

                try:
                    plot_function(
                        slope=slope,
                        stderr=stderr,
                        title=title,
                        output_folder=output_folder,
                        filename_prefix=filename_prefix,
                        pdf=pdf
                    )
                except Exception as e:
                    print(f"Error for {comp} - {sky}: {e}")

    print(f"Combined feedback PDF saved to: {pdf_path}")

# -------- Gregory plot -----------
def plot_single_feedback_file(feedback_file, sim_label="exp1", save_path="feedback_summary.png"):
    """
    Parse a feedback summary file and plot slopes with 95% CI.

    Reads slopes and errors for cloud ("cld") and clear-sky ("clr") feedbacks 
    from a text file, builds a DataFrame, and plots each component with error bars. 
    Blue = cld, Orange = clr. Saves the plot as PNG.

    Parameters
    ----------
    feedback_file : str
        Path to the feedback summary text file.
    sim_label : str, optional
        Label for the simulation (used in the plot title).
    save_path : str, optional
        Output filename for the saved figure.
    """
    feedback_data = {}
    section = ""
    
    with open(feedback_file, "r") as f:
        for line in f:
            line = line.strip()

            if "cld feedback" in line.lower():
                section = "cld"
            elif "clr feedback" in line.lower():
                section = "clr"
            else:
                # Match value
                match = re.match(r"(.+?) feedback: ([-\d\.Ee]+)", line)
                if match:
                    name, val = match.groups()
                    key = f"{section}_{name.strip()}"
                    feedback_data[key] = float(val)
                    continue

                # Match error
                match_err = re.match(r"(.+?) feedback error: ([-\d\.Ee]+)", line)
                if match_err:
                    name, val = match_err.groups()
                    key = f"{section}_{name.strip()}_error"
                    feedback_data[key] = float(val)

    # Prepare DataFrame for plotting
    records = []

    for key in feedback_data:
        if "_error" not in key:
            base = key  # e.g., "cld_planck-surf"
            name_only = base.replace("cld_", "").replace("clr_", "").replace("-", "_")
            err_key = base + "_error"

            records.append({
                "feedback": name_only.capitalize(),
                "cs": "clr_" in base,
                "slope": feedback_data[base],
                "std_err": feedback_data.get(err_key, np.nan),
            })

    df = pd.DataFrame(records)
    feedbacks = df["feedback"].unique()
    x_base = np.arange(len(feedbacks))

    # Plot
    plt.figure(figsize=(10, 5))
    width = 0.25

    for i, fb in enumerate(feedbacks):
        for cs in [False, True]:
            row = df[(df["feedback"] == fb) & (df["cs"] == cs)]
            if not row.empty:
                y = row["slope"].values[0]
                err = row["std_err"].values[0]
                if pd.isna(y) or pd.isna(err):
                    continue
                err = 1.96 * err
                x = i + (-width if not cs else width)

                # Choose color
                color = "#1f77b4" if not cs else "#ff7f0e"  # blue = cld, orange = clr
                label = "cld" if not cs else "clr"
                plt.errorbar(
                    x, y, yerr=err, fmt='o', color=color, capsize=5, ecolor='gray',
                    label=label if i == 0 else None  # only add legend once
                )

    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(x_base, feedbacks, rotation=45, ha='right')
    plt.ylabel("Slope [W/m²/K]")
    plt.title(f"Gregory Feedback Slopes ± 95% CI ({sim_label})", fontsize=20)
    plt.grid(True)
    plt.legend(title="Sky condition")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# --------- Plot dRt for each feedback and for different parameters change variation ------------
# Add plot of each parameter with all change variations value?
def plot_dRt_fb_all_params(base_folder, xlabel, ylabel, title, feedback_file, output_file=None, subfolder_pattern="*", param_map=None, param_order=None, show_zero_line=True, mapping_file="param_mappings.yml", model_type=None):
    """
    Plot dRt values for different parameters. Supports flexible mappings via dictionary or YAML.

    Parameters
    ----------
    base_folder : str
        The base folder containing the subfolders with the data files.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title for the plot.
    feedback_file : str
        The specific feedback file to use (e.g., "dRt_planck-surf_global_clr_climatology-HUANGkernels.nc").
    output_file : str, optional
        File path to save the plot (default: None, plot shown interactively).
    subfolder_pattern : str, optional
        Subfolder pattern to match files (default: "*").
    param_map : dict, optional
        Dictionary mapping subfolder name patterns to pretty parameter names.
        Example: {"pi*a": "ENTRORG", "pi*b": "RPRCON"}.
    param_order : list, optional
        Order of parameters to plot (default: alphabetical).
    show_zero_line : bool, optional
        Whether to draw a horizontal line at 0 (default: True).
    mapping_file : str, optional
        YAML file containing mappings for multiple model types.
    model_type : str, optional
        If provided, will select mapping from YAML by key (e.g., "ece3", "ece4").
    """

    # --- Load mapping if not provided ---
    if param_map is None and os.path.exists(mapping_file):
        with open(mapping_file) as f:
            mappings = yaml.safe_load(f)
        if model_type and model_type in mappings:
            param_map = mappings[model_type]
            print(f"✅ Loaded parameter mapping for {model_type} from {mapping_file}")
        else:
            print(f"⚠️ No model_type provided or not found in {mapping_file}. Will use raw subfolder names.")

    param_names = []
    dRt_values = []

    subfolders = glob.glob(os.path.join(base_folder, subfolder_pattern))

    for subfolder_path in subfolders:
        if os.path.isdir(subfolder_path):
            subfolder_name = os.path.basename(subfolder_path)

            # Match subfolder name against param_map
            matched_param = subfolder_name
            if param_map:
                for pattern, real_name in param_map.items():
                    if fnmatch.fnmatch(subfolder_name, pattern):
                        matched_param = real_name
                        break

            file_path = os.path.join(subfolder_path, feedback_file)
            if not os.path.exists(file_path):
                continue

            try:
                ds = xr.open_dataset(file_path)
                if "__xarray_dataarray_variable__" in ds.data_vars:
                    dRt_mean = float(ds["__xarray_dataarray_variable__"].values)
                    param_names.append(matched_param)
                    dRt_values.append(dRt_mean)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

    df = pd.DataFrame({"param": param_names, "dRt": dRt_values})

    # Sorting if requested
    if param_order:
        df["param"] = pd.Categorical(df["param"], categories=param_order, ordered=True)
        df = df.sort_values("param")
    else:
        df = df.sort_values("param")

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.arange(len(df)))

    # scatter points
    for i, (param, dRt) in enumerate(zip(df["param"], df["dRt"])):
        plt.scatter(i, dRt, color=colors[i % len(colors)], s=100)

    # xticks with custom colors
    xticks = range(len(df))
    plt.xticks(xticks, df["param"], rotation=45, ha="right")

    ax = plt.gca()
    for ticklabel, color in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold", fontsize=12)
    if show_zero_line:
        plt.axhline(0, color="black", linestyle="--", linewidth=1)

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()

    return df


# --------- Time series ------------
def plot_toa_anomaly(base_folder,param,dRt_folder,title,model_type="ece3",sky="clr",output_file=None,):
    """
    Plot Net TOA anomaly (simulation - control climatology) alongside dRt component time series.

    Parameters
    ----------
    base_folder : str
        Base path containing the simulation and control datasets.
    param : str
        Parameter identifier (e.g., 'pilf' for ECE3 or 's001' for ECE4).
    dRt_folder : str
        Folder containing dRt component files (per-year time series).
    title : str
        Title for the plot (function appends '(CLR sky)' or '(CLD sky)').
    model_type : {'ece3','ece4'}
        Model family to select IO conventions and variables.
    sky : {'clr','cld'}
        Sky condition for dRt files.
    output_file : str, optional
        If provided, save figure to this path, else show it.
    """

    # ---------- helpers ----------
    def _guess_names(da):
        # best-effort name guessing for coordinates/dims
        lat = next((n for n in ["lat", "latitude", "nav_lat", "y"] if n in da.coords or n in da.dims), None)
        lon = next((n for n in ["lon", "longitude", "nav_lon", "x"] if n in da.coords or n in da.dims), None)
        time = next((n for n in ["year", "time", "time_counter"] if n in da.coords or n in da.dims), None)
        return lat, lon, time

    def _global_mean(da):
        """Area-weighted global mean over spatial dims (cos(lat) weights)."""
        lat, lon, time = _guess_names(da)
        # If we have lat/lon, do weighted mean
        if lat is not None and lon is not None:
            # weights need to be subset of da dims; 1D weights over lat are fine
            weights = np.cos(np.deg2rad(xr.where(np.isfinite(da[lat]), da[lat], 0.0)))
            gm = da.weighted(weights).mean(dim=[lat, lon])
        else:
            # Fallback: mean over all non-time dims
            spatial_dims = [d for d in da.dims if d not in {"time", "time_counter", "year"}]
            gm = da.mean(dim=spatial_dims)
        return gm

    def _get_time(da):
        lat, lon, time = _guess_names(da)
        if time is None:
            # fallback: take first dim as "time-like"
            time = da.dims[0]
        return time

    # ---------- load sim/control & compute anomaly ----------
    if model_type.lower() == "ece3":
        # tsrc + ttrc
        sim_path_tsrc  = os.path.join(base_folder, "pi", "t_sim", param, f"{param}_*_tsrc.nc")
        sim_path_ttrc  = os.path.join(base_folder, "pi", "t_sim", param, f"{param}_*_ttrc.nc")
        ctrl_path_tsrc = os.path.join(base_folder, "pi", "std_sim", "tpa1", "tpa1_*_tsrc.nc")
        ctrl_path_ttrc = os.path.join(base_folder, "pi", "std_sim", "tpa1", "tpa1_*_ttrc.nc")

        ds_sim_tsrc  = xr.open_mfdataset(sim_path_tsrc, combine="by_coords")
        ds_sim_ttrc  = xr.open_mfdataset(sim_path_ttrc, combine="by_coords")
        ds_ctrl_tsrc = xr.open_mfdataset(ctrl_path_tsrc, combine="by_coords")
        ds_ctrl_ttrc = xr.open_mfdataset(ctrl_path_ttrc, combine="by_coords")

        tnrc_sim  = ds_sim_tsrc["tsrc"] + ds_sim_ttrc["ttrc"]
        tnrc_ctrl = ds_ctrl_tsrc["tsrc"] + ds_ctrl_ttrc["ttrc"]

        time_name = _get_time(tnrc_sim)  # should be 'time'
        tnr_sim_annual  = tnrc_sim.groupby(f"{time_name}.year").mean(dim=time_name)
        tnr_ctrl_annual = tnrc_ctrl.groupby(f"{time_name}.year").mean(dim=time_name)
        # Force integer year coordinates
        tnr_sim_annual = tnr_sim_annual.assign_coords(year=tnr_sim_annual["year"].astype(int))
        tnr_ctrl_annual = tnr_ctrl_annual.assign_coords(year=tnr_ctrl_annual["year"].astype(int))

    elif model_type.lower() == "ece4":
        # rsntcs + rlntcs
        sim_path  = os.path.join(base_folder, "t_sim", param, "oifs", "regridded", f"{param}_*_1m_*.nc")
        ctrl_path = os.path.join(base_folder, "std_sim", "oifs", "s000_*_1m_*.nc")

        ds_sim  = xr.open_mfdataset(sim_path, combine="by_coords")
        ds_ctrl = xr.open_mfdataset(ctrl_path, combine="by_coords")

        tnrc_sim  = ds_sim["rsntcs"] + ds_sim["rlntcs"]
        tnrc_ctrl = ds_ctrl["rsntcs"] + ds_ctrl["rlntcs"]

        time_name = _get_time(tnrc_sim)  # should be 'time_counter'
        tnr_sim_annual  = tnrc_sim.groupby(f"{time_name}.year").mean(dim=time_name)
        tnr_ctrl_annual = tnrc_ctrl.groupby(f"{time_name}.year").mean(dim=time_name)
        tnr_sim_annual  = tnrc_sim.groupby(f"{time_name}.year").mean(dim=time_name)
        tnr_ctrl_annual = tnrc_ctrl.groupby(f"{time_name}.year").mean(dim=time_name)

    else:
        raise ValueError("model_type must be 'ece3' or 'ece4'.")

    # now (year, lat, lon) -> global mean -> (year,)
    climatology     = _global_mean(tnr_ctrl_annual).mean(dim="year")
    tnr_anomaly_gm  = _global_mean(tnr_sim_annual) - climatology  # (year,)
    years_anom      = tnr_anomaly_gm["year"].values
    vals_anom       = tnr_anomaly_gm.values

    # ---------- dRt components (per sky) ----------
    if model_type.lower() == "ece3":
        # albedo exists for clr; may not exist for cld depending on your pipeline → we’ll skip missing files
        default_files = [
            f"dRt_albedo_global_{sky}_climatology-HUANGkernels.nc",
            f"dRt_lapse-rate_global_{sky}_climatology-HUANGkernels.nc",
            f"dRt_planck-atmo_global_{sky}_climatology-HUANGkernels.nc",
            f"dRt_planck-surf_global_{sky}_climatology-HUANGkernels.nc",
            f"dRt_water-vapor_global_{sky}_climatology-HUANGkernels.nc",
        ]
        default_labels = ["Albedo", "Lapse Rate", "Planck Atmos", "Planck Surface", "Water Vapor"]
    else:  # ece4
        default_files = [
            f"dRt_lapse-rate_global_{sky}_climatology-HUANGkernels.nc",
            f"dRt_planck-atmo_global_{sky}_climatology-HUANGkernels.nc",
            f"dRt_planck-surf_global_{sky}_climatology-HUANGkernels.nc",
            f"dRt_water-vapor_global_{sky}_climatology-HUANGkernels.nc",
        ]
        default_labels = ["Lapse Rate", "Planck Atmos", "Planck Surface", "Water Vapor"]

    dRt_components = []
    comp_labels    = []
    comp_times     = None

    for fname, lab in zip(default_files, default_labels):
        path = os.path.join(dRt_folder, fname)
        if not os.path.exists(path):
            # quiet skip if missing
            # print(f"Missing {path}, skipping.")
            continue
        ds = xr.open_dataset(path)
        da = ds["__xarray_dataarray_variable__"]
        # ensure we have a time-like coord (prefer 'year')
        tname = "year" if "year" in da.coords or "year" in da.dims else _get_time(da)
        da = da.assign_coords({tname: np.round(da[tname].values).astype(int)})
        if comp_times is None:
            comp_times = da[tname].values
        dRt_components.append(da)
        comp_labels.append(lab)

    if len(dRt_components) == 0:
        raise FileNotFoundError(f"No dRt component files found in {dRt_folder} for sky='{sky}'.")

    # align and sum
    aligned = xr.align(*dRt_components, join="inner")
    dRt_sum = sum(aligned)
    comp_times = aligned[0]["year"].values.astype(int) if "year" in aligned[0].coords else aligned[0][_get_time(aligned[0])].values.astype(int)

    # ---------- plotting ----------
    plt.figure(figsize=(12,5))
    plt.plot(years_anom.astype(int), vals_anom, color="black", marker="o", linestyle="-", label="Net TOA Anomaly")

    color_cycle = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan", "tab:pink", "tab:brown"]
    for i, (comp, lab) in enumerate(zip(aligned, comp_labels)):
        tname = "year" if "year" in comp.coords or "year" in comp.dims else _get_time(comp)
        plt.plot(np.round(comp[tname].values).astype(int), comp.values, marker="s", linestyle="--",
                 label=lab, color=color_cycle[i % len(color_cycle)])

    tname_sum = "year" if "year" in dRt_sum.coords or "year" in dRt_sum.dims else _get_time(dRt_sum)
    plt.plot(np.round(dRt_sum[tname_sum].values).astype(int), dRt_sum.values, color="red", linestyle="-", linewidth=2,
             label="Sum of dRt Components")

    plt.xlabel("Year")
    plt.ylabel("W/m²")
    plt.title(f"{title} ({sky.upper()} sky)", fontsize=14)
    plt.xticks(years_anom, [str(y) for y in years_anom])
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        plt.show()