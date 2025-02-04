import xarray as xr
import os
import matplotlib.pyplot as plt

def plot_dRt(folder_path, files, variations, xlabel, ylabel, title, labels, output_file=None):
    """
    Plot dRt values against parameter variations, with different colors for each feedback.

    Parameters:
    -----------
    folder_path : str
        The folder containing the NetCDF files.
    files : list
        List of file names to process.
    variations : list
        List of parameter variation values (e.g., [-30, -20, 20, 30]).
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title for the plot.
    labels : list
        List of labels for each file/feedback.
    output_file : str, optional
        File path to save the plot (default: None, plot shown interactively).
    """
    dRt_means = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # List of colors for points
    
    for file_name in files:
        file_path = os.path.expanduser(os.path.join(folder_path, file_name))
        try:
            ds = xr.open_dataset(file_path)
            
            # Check for variable existence and extract mean
            if "__xarray_dataarray_variable__" in ds.data_vars:
                dRt_mean = float(ds["__xarray_dataarray_variable__"].mean().values)
                dRt_means.append(dRt_mean)
                print(f"Processed: {file_name}, dRt = {dRt_mean:.6f}")
            else:
                print(f"File {file_name} is missing '__xarray_dataarray_variable__'. Skipping...")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Plot if data exists
    if dRt_means:
        plt.figure(figsize=(8, 6))
        
        for i, (variation, dRt, label) in enumerate(zip(variations, dRt_means, labels)):
            color = colors[i % len(colors)]  # Cycle through colors if more files than colors
            plt.scatter(variation, dRt, color=color, label=label, s=100)  # s=100 for larger points
        
        plt.xticks([-30, -20, -10, 0, 10, 20, 30])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        
        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
    else:
        print("No valid data to plot.")

