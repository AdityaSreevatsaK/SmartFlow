import warnings

# Ignore informational warnings from pandas and seaborn
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress C++ logs from XLA / glog
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disabling the tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pickle
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from tensorboard.backend.event_processing import event_accumulator

from .routing import init_worker, find_path_worker

seeds = [0, 11, 28]


def plot_all_learning_curves(log_directories: list):
    """
    Plots the learning curves (cumulative reward over time) for multiple training runs.

    This function generates two types of plots:
    1. Individual learning curves for each seed/run, showing the moving average
       of rewards over episodes.
    2. An aggregated learning curve, displaying the mean and standard deviation
       of rewards across all runs, providing an overall view of the agent's
       learning progress.

    The rewards are read from 'env_*.csv' files within the specified log directories.
    All reward series are trimmed to the length of the shortest run to ensure
    consistent averaging. A moving average is applied to smooth the curves.

    Args:
        log_directories (list): A list of paths to directories, where each directory
                                contains log files (e.g., 'env_*.csv') from a
                                single training run.

    Returns:
        None. Displays matplotlib plots directly.

    Prints:
        - Status messages about log processing.
        - An error message if no valid log files are found.
    """
    print("--- Metric 1: Cumulative Reward Over Time ---")
    all_rewards = []
    min_episodes = float('inf')

    for log_dir in log_directories:
        try:
            log_files = sorted(glob.glob(os.path.join(log_dir, "env_*.csv")))
            if not log_files:
                continue

            all_logs = pd.concat([pd.read_csv(f, comment="#") for f in log_files], ignore_index=True)
            rewards = all_logs["r"].values
            all_rewards.append(rewards)
            if len(rewards) > 0:
                min_episodes = min(min_episodes, len(rewards))

        except Exception as e:
            print(f"⚠️ Could not process logs in {log_dir}: {e}")

    if not all_rewards or min_episodes == float('inf'):
        print("❌ No valid log files found to plot learning curves.")
        return

    # Trim all reward lists to the length of the shortest run for averaging
    trimmed_rewards = [rewards[:min_episodes] for rewards in all_rewards]

    # --- Individual Plots ---
    num_seeds = len(trimmed_rewards)
    fig, axes = plt.subplots(1, num_seeds, figsize=(6 * num_seeds, 5), sharey=True)
    if num_seeds == 1: axes = [axes]  # Ensure axes is always iterable
    fig.suptitle('Individual Learning Curves per Seed', fontsize=16)
    for i, rewards in enumerate(trimmed_rewards):
        ax = axes[i]
        window = min(200, len(rewards) // 5) if len(rewards) // 5 > 0 else 1
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        episodes = np.arange(1, len(ma) + 1)  # Simplified x-axis calculation
        ax.plot(episodes, ma, label=f"Seed {seeds[i]} Run Reward MA")
        ax.set_title(f"Seed {seeds[i]} Run")
        ax.set_xlabel("Episode")
        if i == 0: ax.set_ylabel("Moving Average Reward")
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Aggregated Plot ---
    window = min(200, min_episodes // 5) if min_episodes // 5 > 0 else 1

    all_ma = [np.convolve(rewards, np.ones(window) / window, mode="valid") for rewards in trimmed_rewards]

    mean_ma = np.mean(all_ma, axis=0)
    std_ma = np.std(all_ma, axis=0)

    ma_episodes = np.arange(1, len(mean_ma) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(ma_episodes, mean_ma, color='#E43636', label='Mean Reward (across all seeds)')
    plt.fill_between(ma_episodes, mean_ma - std_ma, mean_ma + std_ma, color='#FFE100', alpha=0.2,
                     label='Standard Deviation')
    plt.title('Aggregated Agent Learning Curve (Mean ± Std Dev)')
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_all_training_stats(log_directories: list):
    """
    This function processes log files and TensorBoard event files from various
    training runs to extract and present key training metrics such as:
    - Total training time
    - Frames per second (FPS)
    - Final policy loss

    It includes logic to robustly find the correct TensorBoard loss tag,
    handling common variations.

    Args:
        log_directories (list): A list of paths to directories containing
                                training log files (e.g., env_*.csv) and
                                TensorBoard event files (e.g., tb_seed_*).
    Includes detective logic to find the correct TensorBoard loss tag.
    """
    print("\n--- Metric 2: Aggregated Training Statistics ---")
    times, fps, losses = [], [], []

    base_path = os.path.dirname(log_directories[0]) if log_directories else "."
    all_tb_dirs = sorted(glob.glob(os.path.join(base_path, "tb_seed_*")))

    for i, log_dir in enumerate(log_directories):
        try:
            # Time and FPS
            log_files = sorted(glob.glob(os.path.join(log_dir, "env_*.csv")))
            if log_files:
                all_logs = pd.concat([pd.read_csv(f, comment="#") for f in log_files], ignore_index=True)
                times.append(all_logs["t"].iloc[-1] / 60)
                fps.append(all_logs["l"].sum() / all_logs["t"].iloc[-1])

            # Final Loss
            if i < len(all_tb_dirs):
                tb_log_search_path = os.path.join(all_tb_dirs[i], "DQN_*")
                tb_log_dir_list = glob.glob(tb_log_search_path)

                if tb_log_dir_list:
                    tb_log_dir = sorted(tb_log_dir_list)[-1]
                    ea = event_accumulator.EventAccumulator(tb_log_dir, size_guidance={'scalars': 0})
                    ea.Reload()

                    available_tags = ea.Tags()['scalars']
                    loss_tag_primary = 'train/loss'
                    loss_tag_fallback = 'rollout/loss'

                    if loss_tag_primary in available_tags:
                        losses.append(ea.Scalars(loss_tag_primary)[-1].value)
                    elif loss_tag_fallback in available_tags:
                        print(f"⚠️ '{loss_tag_primary}' not found. Using fallback '{loss_tag_fallback}'.")
                        losses.append(ea.Scalars(loss_tag_fallback)[-1].value)
                    else:
                        # If neither is found, print all available tags for debugging
                        print(f"⚠️ Could not find '{loss_tag_primary}' or '{loss_tag_fallback}' in {tb_log_dir}.")
                        print(f"   --> Available scalar tags are: {available_tags}")
                else:
                    print(f"⚠️ Could not find a DQN sub-directory in {all_tb_dirs[i]}.")
            else:
                print(f"⚠️ Could not find a matching TensorBoard directory for {os.path.basename(log_dir)}.")

        except Exception as e:
            print(f"⚠️ Could not calculate stats for {log_dir}: {e}")

    print(f"\n[Aggregated Results across {len(times)} runs]")

    mean_time = f"{np.mean(times):.2f} ± {np.std(times):.2f}" if len(
        times) > 1 else f"{times[0]:.2f}" if times else "N/A"
    mean_fps = f"{np.mean(fps):.2f} ± {np.std(fps):.2f}" if len(fps) > 1 else f"{fps[0]:.2f}" if fps else "N/A"
    if losses:
        mean_loss = f"{np.mean(losses):.4f} ± {np.std(losses):.4f}" if len(losses) > 1 else f"{losses[0]:.4f}"
        print(f"   - Final Policy Loss: {mean_loss}")
    else:
        print("   - Final Policy Loss: Not available (could not load from logs).")


def calculate_all_operational_metrics(all_results: list):
    """
    Calculates and prints aggregated operational and system performance metrics
    from multiple simulation runs.

    Metrics include:
    - Final network imbalance score (mean and standard deviation).
    - Imbalance reduction percentage (mean and standard deviation).
    - Total fleet distance traveled (mean and standard deviation).
    - Truck utilization rate for multi-leg journeys (mean and standard deviation).

    Args:
        all_results (list): A list of dictionaries, where each dictionary
                            contains the results from a single simulation run.
                            Expected keys in each dictionary include:
                            "initial_counts", "live_counts", "thresh",
                            "optimized_journeys", "graph", "station_to_node_map".
    """
    print("\n--- Metric 3: Aggregated Operational & System Performance ---")
    imbalances, reductions, distances, utilisations = [], [], [], []

    for results in all_results:
        initial_counts = results["initial_counts"]
        live_counts = results["live_counts"]
        thresh = results["thresh"]

        initial_imbalance = sum(abs(c - thresh.get(s, 0)) for s, c in initial_counts.items())
        final_imbalance = sum(abs(c - thresh.get(s, 0)) for s, c in live_counts.items())

        imbalances.append(final_imbalance)
        reductions.append(
            ((initial_imbalance - final_imbalance) / initial_imbalance) * 100 if initial_imbalance > 0 else 0)

        total_distance_km = 0
        multi_leg_trucks = 0
        optimized_journeys = results["optimized_journeys"]
        if optimized_journeys:
            G = results["graph"]
            station_to_node = results["station_to_node_map"]
            pathfinding_tasks = [(leg["src"], leg["tgt"]) for j in optimized_journeys for leg in j["legs"]]
            with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker,
                                      initargs=(G, station_to_node)) as pool:
                path_results = pool.map(find_path_worker, pathfinding_tasks)
            path_dict = {(src, tgt): nodes for src, tgt, nodes in path_results if nodes}
            for journey in optimized_journeys:
                if len(journey['legs']) > 1: multi_leg_trucks += 1
                for leg in journey['legs']:
                    path_nodes = path_dict.get((leg['src'], leg['tgt']))
                    if path_nodes:
                        path_len = sum(G.edges[u, v, 0]['length'] for u, v in zip(path_nodes[:-1], path_nodes[1:]))
                        total_distance_km += path_len / 1000
        distances.append(total_distance_km)
        utilisations.append((multi_leg_trucks / len(optimized_journeys)) * 100 if optimized_journeys else 0)

    print("\n[System-Level Performance]")
    print(f"   - Final Network Imbalance Score: {np.mean(imbalances):.2f} ± {np.std(imbalances):.2f} bikes")
    print(f"   - Imbalance Reduction: {np.mean(reductions):.2f} ± {np.std(reductions):.2f} %")
    print("\n[Operational Efficiency]")
    print(f"   - Total Fleet Distance Travelled: {np.mean(distances):.2f} ± {np.std(distances):.2f} km")
    print(
        f"   - Truck Utilisation Rate (multi-leg journeys): {np.mean(utilisations):.2f} ± {np.std(utilisations):.2f} %")


def plot_all_task_prioritizations(all_results: list):
    """
    Plots individual and aggregated histograms of task prioritization based on the
    'first_hour' of transfer events. This function visualizes the distribution
    of when tasks (e.g., rebalancing transfers) are initiated throughout the day.

    It generates two types of plots:
    1. Individual plots for each simulation run (seed), showing the density
       of tasks per hour, along with a Kernel Density Estimate (KDE) for trend.
    2. An aggregated plot combining data from all runs, providing an overall
       view of task prioritization patterns.

    Args:
        all_results (list): A list of dictionaries, where each dictionary
                            contains the results from a single simulation run,
                            including a 'transfers' key if available."""
    print("\n--- Metric 4: Task Prioritisation Analysis ---")
    all_hours = []

    for res in all_results:
        transfers = res.get("transfers")
        if transfers:
            all_hours.append([data['first_hour'] for _, data in transfers.items()])

    if not all_hours:
        print("❌ No transfer data available to plot task prioritization.")
        return

    # --- Individual Plots ---
    num_seeds = len(all_hours)
    fig, axes = plt.subplots(1, num_seeds, figsize=(6 * num_seeds, 5), sharey=True)
    fig.suptitle('Individual Task Prioritisation per Seed', fontsize=16)
    for i, hours in enumerate(all_hours):
        ax = axes[i] if num_seeds > 1 else axes
        sns.histplot(hours, bins=24, binrange=(0, 24), color='#08CB00', stat='density', ax=ax, label='Task Density')
        sns.kdeplot(hours, linewidth=2, ax=ax, color='#253900', label='Demand Trend (KDE)')
        ax.set_title(f"Seed {seeds[i]} Run")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Density of Tasks")
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Aggregated Plot ---
    flat_hours = [hour for seed_hours in all_hours for hour in seed_hours]
    plt.figure(figsize=(12, 6))
    sns.histplot(flat_hours, bins=24, binrange=(0, 24), stat='density', color='#FF8040', edgecolor='black',
                 label='Task Density (All Runs)')
    sns.kdeplot(flat_hours, color='#0046FF', linewidth=2, label='Overall Demand Trend (KDE)')
    plt.title('Aggregated Agent Task Prioritisation')
    plt.xlabel("Hour of Day (24h format)")
    plt.ylabel("Density of Tasks")
    plt.xticks(range(0, 25, 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def analyse_all_runs(base_path: str = "../../results/metrics/"):
    """
    This function orchestrates the analysis of simulation results and training logs.
    It searches for pickled simulation result files and log directories based on a
    specified base path, loads the data, and then calls various plotting and
    calculation functions to generate aggregated metrics and visualizations.

    Args:
        base_path (str): The base directory where simulation results and log directories are stored.
    """
    try:
        results_files = sorted(glob.glob(os.path.join(base_path, "simulation_results_seed_*.pkl")))
        log_dirs = sorted(glob.glob(os.path.join(base_path, "logs_seed_*")))

        if not results_files:
            raise FileNotFoundError("No simulation result files (simulation_results_seed_*.pkl) found.")

        print(f"Found {len(results_files)} result files and {len(log_dirs)} log directories. Analyzing...")

        all_results_data = []
        for f_path in results_files:
            with open(f_path, "rb") as f:
                all_results_data.append(pickle.load(f))

        # Generate each aggregated metric and plot
        plot_all_learning_curves(log_dirs)
        calculate_all_training_stats(log_dirs)
        calculate_all_operational_metrics(all_results_data)
        plot_all_task_prioritizations(all_results_data)

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("   Please run the main simulation script first to generate the necessary files.")
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")


if __name__ == "__main__":
    analyse_all_runs()
