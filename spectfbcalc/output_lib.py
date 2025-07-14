import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import re

from matplotlib.backends.backend_pdf import PdfPages

def save_feedback_output(output, out_path_txt, out_path_nc=None):
    """
    Save feedback results to .txt and .nc files.

    Parameters:
    - output: a tuple containing (fb_coeffs, fb_cloud, fb_cloud_err, fb_pattern, fb_cloud_pattern)
    - out_path_txt: full path to output .txt file
    - out_path_nc: full path to output .nc file (optional; required if patterns are present)
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

def plot_fb_pattern(slope, stderr, title, output_folder, filename_prefix="fb_pattern", pdf=None):
    """
    Plot slope and stderr separately, save as PNG, and optionally add to a combined PDF.

    Parameters:
    - slope (xarray.DataArray): Feedback pattern slope.
    - stderr (xarray.DataArray): Standard error of the feedback pattern.
    - title (str): Title of the plots.
    - output_folder (str): Path to the folder where plots will be saved.
    - filename_prefix (str): Prefix for output filenames.
    - pdf (PdfPages, optional): If provided, adds plots to this PDF.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Plot slope
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    slope.plot(
        ax=ax1,
        cmap="RdBu_r",
        robust=True,
        cbar_kwargs={"label": "W/m²/K"}
    )
    ax1.grid(False)
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)
    ax1.set_title(title, fontsize=16, weight="bold")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_folder, f"{filename_prefix}_slope.png"), dpi=300)
    if pdf:
        pdf.savefig(fig1)
    plt.close(fig1)

    # Plot stderr
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    stderr.plot(
        ax=ax2,
        cmap="viridis",
        robust=True,
        cbar_kwargs={"label": "W/m²/K"}
    )
    ax2.grid(False)
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)
    ax2.set_title(title, fontsize=16, weight="bold")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_folder, f"{filename_prefix}_stderr.png"), dpi=300)
    if pdf:
        pdf.savefig(fig2)
    plt.close(fig2)


def save_all_fb_patterns_to_pdf(ds: xr.Dataset, output_folder: str, pdf_name: str = "all_fb_patterns.pdf", components: list = None, skies: list = None, plot_function=None):
    """
    Save all feedback pattern maps to individual PNGs and a combined multipage PDF.

    Parameters:
    - ds (xr.Dataset): Dataset containing *_slope and *_stderr variables.
    - output_folder (str): Folder where PNGs and PDF will be saved.
    - pdf_name (str): Name of the combined PDF file.
    - components (list): List of components (e.g. ["albedo", "water-vapor", ...]).
    - skies (list): List of sky types (e.g. ["clr", "cld"]).
    - plot_function (callable): Function to plot the feedback pattern. Must accept 
                                slope, stderr, title, output_folder, filename_prefix, pdf.
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

                title = f"Feedback Pattern: {sky.upper()} - {comp}"
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


def plot_single_feedback_file(feedback_file, sim_label="exp1", save_path="feedback_summary.png"):
    # Read and parse the single feedback file
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
    plt.title(f"Gregory Feedback Slopes ± 95% CI ({sim_label})")
    plt.grid(True)
    plt.legend(title="Sky condition")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()