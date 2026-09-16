import yaml
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import glob
from collections import defaultdict
import re
import plot_DEFINITIONS

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42



# --- Plotting Configuration & Definitions ---

# Set global plot styling
plt.rcParams.update({
    'font.size': 32,           # Default font size
    'axes.labelsize': 32,      # Axes labels font size
    'xtick.labelsize': 28,     # X-tick label size
    'ytick.labelsize': 28,     # Y-tick label size
    'legend.fontsize': 28,     # Legend font size
    'axes.titlesize': 32,      # Axes titles font size
    'axes.titleweight': 'bold',  # Axes titles font weight
})

# Define the metrics to be plotted from the yaml files
METRICS_TO_PLOT = [
    "error_vel_xy",
    "error_vel_yaw",
    "mean_mechanical_cot",
    "agent_expert_distances",
]

# Define user-friendly names for plot titles
METRIC_PLOT_TITLES = {
    "error_vel_yaw": "Tracking Error\nYaw [rad]",
    "error_vel_xy": "Tracking Error\nVel. [m/s]",
    "mean_mechanical_cot": "Cost of\nTransport [1]",
    "agent_expert_distances": "Imitation\nScore [1]",
}

# Hack to extract same colors as used in plot_selected_metrics.py
all_run_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] 
cmap = plt.cm.get_cmap('tab10', len(all_run_names))
COLORS_TO_PLOT = {
    "MoCap": cmap.colors[1],
    "MoCap (AMP)": cmap.colors[1],
    "Video w. Depth Camera (extended) (AMP)": cmap.colors[3],
}

# --- Data Loading and Processing Functions ---


def load_yaml_file(file_path):
    """Safely loads a YAML file."""
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except FileNotFoundError:
        print(f"Warning: YAML file not found at {file_path}. Skipping.")
        return {}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def get_weight_from_path(path):
    """Extracts the trackAnVelRewWeight value from a directory path."""
    match = re.search(r"trackAnVelRewWeight_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def collect_experiment_data(experiment_pattern):
    """
    Collects and groups experiment paths by trackAnVelRewWeight.

    Args:
        experiment_pattern (str): A glob pattern to find experiment directories.

    Returns:
        defaultdict: A dictionary mapping weight values to lists of seed paths.
    """
    grouped_by_weight = defaultdict(list)
    # Use glob to find all matching directories for all seeds
    full_pattern = os.path.join(
        "logs/rsl_rl/unitree_go2_AMPflat/", f"{experiment_pattern}_SEED_*"
    )

    for path in glob.glob(full_pattern):
        weight = get_weight_from_path(path)
        if weight is not None:
            grouped_by_weight[weight].append(path)

    return grouped_by_weight


def process_data(grouped_paths):
    """
    Processes grouped paths to extract and aggregate metrics.

    Args:
        grouped_paths (defaultdict): A dictionary mapping weights to paths.

    Returns:
        dict: A dictionary containing sorted weights and metric statistics.
    """
    processed_metrics = {metric: [] for metric in METRICS_TO_PLOT}
    weights = sorted(grouped_paths.keys())

    for weight in weights:
        paths = grouped_paths[weight]

        # Temp storage for metrics from each seed at this weight
        metrics_per_seed = {metric: [] for metric in METRICS_TO_PLOT}

        for path in paths:
            yaml_path = os.path.join(path, "None_metrics.yaml")
            data = load_yaml_file(yaml_path)

            for metric in METRICS_TO_PLOT:
                if metric in data:
                    metrics_per_seed[metric].append(data[metric])

        # Calculate mean, min, max for each metric across seeds
        for metric, values in metrics_per_seed.items():
            if values:
                arr = np.array(values)
                processed_metrics[metric].append(
                    {
                        "mean": np.mean(arr),
                        "min": np.min(arr),
                        "max": np.max(arr),
                        "range": np.max(arr) - np.min(arr),
                    }
                )

    return {"weights": weights, **processed_metrics}


# --- Plotting Function ---


def plot_comparison(all_experiments_data, metrics_to_plot):
    """
    Generates and saves line plots comparing different experiments.
    """
    if not os.path.exists("plots"):
        os.makedirs("plots")
        print("Created directory 'plots/' for saving figures.")

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Collect all unique weights from all experiments for setting x-ticks
        all_weights = []
        for exp_name, data in all_experiments_data.items():
            if "weights" in data and data["weights"]:
                all_weights.extend(data["weights"])
        unique_weights = sorted(list(set(all_weights)))

        for exp_name, data in all_experiments_data.items():
            weights = data.get("weights", [])
            metric_stats = data.get(metric, [])

            if not weights or not metric_stats:
                print(
                    f"Skipping plot for '{exp_name}' in metric '{metric}' due to missing data."
                )
                continue

            means = [s["mean"] for s in metric_stats]

            # Plot the mean line and capture its color
            (line,) = ax.plot(
                weights,
                means,
                marker="o",
                linestyle="-",
                label=exp_name,
                linewidth=4,
                color=plot_DEFINITIONS.get_color_for_experiment_name(exp_name),
            )
            line_color = line.get_color()  # Get the color assigned by matplotlib

            # Calculate the error (difference from mean) for error bars
            if all(st.get("min") is not None and st.get("max") is not None for st in metric_stats):
                mins = [s["min"] for s in metric_stats]
                maxs = [s["max"] for s in metric_stats]
                lower_errors = [mean - m_min for mean, m_min in zip(means, mins)]
                upper_errors = [m_max - mean for mean, m_max in zip(means, maxs)]
                yerr = [
                    lower_errors,
                    upper_errors,
            ]  # This format is for asymmetric error bars

                # Plot error bars using the same color as the line
                ax.errorbar(weights, means, yerr=yerr, fmt="o", capsize=4, color=line_color)

        ax.set_xlabel("Yaw Tracking Reward Weight")
        ax.set_ylabel(METRIC_PLOT_TITLES.get(metric, metric))
        if metric == "error_vel_yaw":
            # ax.legend()
            pass
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Set x-ticks and tick labels only to the datapoints
        if unique_weights:
            ax.set_xticks(unique_weights)
            ax.set_xticklabels([str(w) for w in unique_weights])

        fig.tight_layout()

        # Save the figure
        filename = f"plots/{metric}_vs_trackYawRewWeight.pdf"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        plt.close(fig)
        
def plot_legend_only(all_experiments_data, filename="plots/legend_only.pdf"):
    """
    Generates and saves a standalone legend figure.
    
    Args:
        all_experiments_data (dict): Dictionary containing experiment data
        filename (str): Output filename for the legend plot
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')  # Hide the axes
    
    # Create dummy plots to generate legend entries
    handles = []
    labels = []
    for exp_name in all_experiments_data.keys():
        line, = ax.plot([], [], marker="o", linestyle="-", linewidth=4,
                       color=plot_DEFINITIONS.get_color_for_experiment_name(exp_name),
                       label=exp_name)
        handles.append(line)
        labels.append(exp_name)
    
    # Create legend in the center of the figure
    legend = ax.legend(handles, labels, loc='center', frameon=False, ncol=2)
    
    fig.tight_layout()
    
    # Save the figure
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)  # Minimal padding
    print(f"Saved legend figure: {filename}")
    plt.close(fig)



# --- Main Execution ---


def main():
    """
    Main function to define experiments, collect data, and generate plots.
    """
    # Define the experiment patterns and their display names for the legend
    # You can add or modify entries here to change the plots.
    EXPERIMENTS = {
        "Video w. Depth Camera (extended) (AMP)": "2025-07-11_21-14-15_fromVision_motions_DepthCam_extendedWithoutReverse_*trackAnVelRewWeight*",
        "MoCap (AMP)": "2025-07-13_13-22-15_mocap_AMP_for_hardware_*trackAnVelRewWeight*",
    }

    all_data = {}
    print("Starting data collection...")
    for name, pattern in EXPERIMENTS.items():
        print(f"Processing experiment: '{name}'")
        grouped_paths = collect_experiment_data(pattern)
        if not grouped_paths:
            print(f"Warning: No data found for pattern: {pattern}")
            continue
        all_data[name] = process_data(grouped_paths)

    # prepend datapoint (yaw)
    all_data["MoCap (AMP)"]["weights"] = [20] + all_data["MoCap (AMP)"]["weights"]
    all_data["MoCap (AMP)"]["error_vel_yaw"]  = [
        {"mean": 0.64500141143, "min": 0.5899317264556885, "max": 0.6824547648429871},
    ] + all_data["MoCap (AMP)"]["error_vel_yaw"]
    
    all_data["MoCap (AMP)"]["error_vel_xy"]  = [
        {"mean": 0.06239856034, "min": 0.06087314337491989, "max": 0.06330526620149612},
    ] + all_data["MoCap (AMP)"]["error_vel_xy"]
    
    all_data["MoCap (AMP)"]["mean_mechanical_cot"] = [
        {"mean": 1.5135, "min": 1.4792, "max": 1.5474},   # three paper seeds, metrics.yaml (logs.zip)
    ] + all_data["MoCap (AMP)"]["mean_mechanical_cot"]
    all_data["MoCap (AMP)"]["agent_expert_distances"] = [
        {"mean": 2.1453, "min": 2.1090, "max": 2.1917},   # three paper seeds, metrics.yaml (logs.zip)
    ] + all_data["MoCap (AMP)"]["agent_expert_distances"]

    # prepend datapoint (xy vel)
    all_data["Video w. Depth Camera (extended) (AMP)"]["weights"] = [20] + all_data["Video w. Depth Camera (extended) (AMP)"]["weights"]
    all_data["Video w. Depth Camera (extended) (AMP)"]["error_vel_yaw"]  = [
        {"mean": 0.12919001529, "min": 0.1237926259636879, "max": 0.13561595976352692},
    ] + all_data["Video w. Depth Camera (extended) (AMP)"]["error_vel_yaw"]
    
    all_data["Video w. Depth Camera (extended) (AMP)"]["error_vel_xy"]  = [
        {"mean": 0.04806786527, "min": 0.04705556482076645, "max": 0.04966380074620247},
    ] + all_data["Video w. Depth Camera (extended) (AMP)"]["error_vel_xy"]

    all_data["Video w. Depth Camera (extended) (AMP)"]["mean_mechanical_cot"] = [
        {"mean": 1.3112, "min": 1.2464, "max": 1.4295},   # three paper seeds, metrics.yaml (logs.zip)
    ] + all_data["Video w. Depth Camera (extended) (AMP)"]["mean_mechanical_cot"]
    all_data["Video w. Depth Camera (extended) (AMP)"]["agent_expert_distances"] = [
        {"mean": 2.1754, "min": 2.1365, "max": 2.2072},   # three paper seeds, metrics.yaml (logs.zip)
    ] + all_data["Video w. Depth Camera (extended) (AMP)"]["agent_expert_distances"]

    if all_data:
        print("\nData collection complete. Generating plots...")
        plot_comparison(all_data, METRICS_TO_PLOT)
        print("\nScript finished.")
    else:
        print("\nNo data was collected. Please check your paths and patterns.")
        
    plot_legend_only(all_data)
        
        


if __name__ == "__main__":
    main()
