import glob
import multiprocessing
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

from .routing import find_path_worker, init_worker


# ==============================================================================
# Helper Functions for Metric Calculation
# ==============================================================================

def plot_learning_curve(log_directory: str):
    """
    Plots the agent's cumulative reward curve over training episodes using log files.

    Args:
        log_directory (str): Path to the directory containing training log CSV files.

    The function:
        - Loads all CSV log files matching 'env_*.csv' in the specified directory.
        - Extracts episode rewards and computes a moving average.
        - Fits a linear trend to the moving average to visualize learning progress.
        - Plots the moving average and trend line using matplotlib.
        - Prints the total number of episodes found.
        - Warns if there are not enough episodes to plot a meaningful curve.
        - Handles and reports errors if log files are missing or unreadable.
    """
    print("--- Metric 1: Cumulative Reward Over Time ---")
    try:
        log_files = sorted(glob.glob(os.path.join(log_directory, "env_*.csv")))
        if not log_files:
            raise FileNotFoundError(f"No Monitor log files found in {log_directory}")

        all_logs = pd.concat([pd.read_csv(f, comment="#") for f in log_files], ignore_index=True)
        rewards = all_logs["r"].values
        print(f"Total episodes found in logs: {len(rewards)}")

        if len(rewards) > 20:
            window = min(500, len(rewards) // 5)
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            episodes = np.arange(window, window + len(ma))
            slope, intercept = np.polyfit(episodes, ma, deg=1)
            trend = slope * episodes + intercept

            plt.figure(figsize=(12, 6))
            plt.plot(episodes, ma, label=f"{window}-episode Moving Average Reward", color="#F97A00")
            plt.plot(episodes, trend, linestyle="--", label=f"Learning Trend (Slope: {slope:.4f})", color="#386641")
            plt.xlabel("Episode")
            plt.ylabel("Moving Average Reward")
            plt.title("Agent Learning Curve: Reward Over Time")
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='both')
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Not enough training episodes to generate a meaningful reward plot.")
    except Exception as e:
        print(f"❌ Could not generate reward plot: {e}")


def calculate_training_stats(log_directory: str):
    """
    Calculates and prints training statistics from log files.

    Args:
        log_directory (str): Path to the directory containing training log files.

    Functionality:
        - Loads Monitor log CSV files matching 'env_*.csv' in the specified directory.
        - Calculates total training time (in minutes) and average training speed (FPS).
        - Loads the latest TensorBoard log directory matching 'DQN_*' and extracts the final policy loss.
        - Prints all metrics and handles missing or unreadable log files gracefully.
    """
    print("\n--- Metric 2: Training Statistics ---")
    try:
        # Get Time and FPS from Monitor logs
        log_files = sorted(glob.glob(os.path.join(log_directory, "env_*.csv")))
        if log_files:
            all_logs = pd.concat([pd.read_csv(f, comment="#") for f in log_files], ignore_index=True)
            total_time_seconds = all_logs["t"].iloc[-1]
            total_timesteps = all_logs["l"].sum()
            average_fps = total_timesteps / total_time_seconds
            print(f"   - Total Training Time: {total_time_seconds / 60:.2f} minutes")
            print(f"   - Average Training Speed: {average_fps:.2f} FPS")
        else:
            print("⚠️ Could not find Monitor logs for timing data.")

        # Get Final Loss from TensorBoard logs
        tb_log_dir_list = glob.glob(os.path.join(log_directory, "../tb/DQN_*/"))
        if tb_log_dir_list:
            tb_log_dir = sorted(tb_log_dir_list)[-1]
            ea = event_accumulator.EventAccumulator(tb_log_dir, size_guidance={'scalars': 0})
            ea.Reload()
            loss_data = ea.Scalars('train/loss')
            if loss_data:
                print(f"   - Final Policy Loss: {loss_data[-1].value:.4f}")
            else:
                print("⚠️ Could not find loss data in TensorBoard logs.")
        else:
            print("⚠️ Could not find TensorBoard log directory.")
    except Exception as e:
        print(f"❌ Could not calculate training stats: {e}")


def calculate_operational_metrics(results: dict):
    """
    Calculates system performance and operational efficiency metrics.

    Args:
        results (dict): Dictionary containing simulation results with keys:
            - initial_counts: dict of initial bike counts per station
            - live_counts: dict of final bike counts per station
            - thresh: dict of target bike counts per station
            - optimized_journeys: list of truck journey dicts, each with 'legs'
            - graph: networkx graph representing the network
            - station_to_node_map: mapping from station to graph node

    Functionality:
        - Computes initial and final network imbalance scores and reduction percentage.
        - Calculates total fleet distance travelled by trucks (in km).
        - Determines truck utilisation rate (percentage of trucks with multi-leg journeys).
        - Uses multiprocessing for efficient pathfinding between journey legs.
        - Prints all calculated metrics.
    """
    print("\n--- Metric 3: Operational & System Performance ---")

    # Extract data from the results dictionary
    initial_counts = results["initial_counts"]
    live_counts = results["live_counts"]
    thresh = results["thresh"]
    optimized_journeys = results["optimized_journeys"]
    G = results["graph"]
    station_to_node = results["station_to_node_map"]

    # System-Level Performance
    initial_imbalance = sum(abs(c - thresh.get(s, 0)) for s, c in initial_counts.items())
    final_imbalance = sum(abs(c - thresh.get(s, 0)) for s, c in live_counts.items())
    imbalance_reduction = ((
                                   initial_imbalance - final_imbalance) / initial_imbalance) * 100 if initial_imbalance > 0 else 0
    print("\n[System-Level Performance]")
    print(f"   - Initial Network Imbalance Score: {initial_imbalance} bikes")
    print(f"   - Final Network Imbalance Score: {final_imbalance} bikes")
    print(f"   - Imbalance Reduction: {imbalance_reduction:.2f}%")

    # Operational Efficiency
    total_distance_km = 0
    multi_leg_trucks = 0
    if optimized_journeys:
        pathfinding_tasks = [(leg["src"], leg["tgt"]) for journey in optimized_journeys for leg in journey["legs"]]
        with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker,
                                  initargs=(G, station_to_node)) as pool:
            path_results = pool.map(find_path_worker, pathfinding_tasks)
        path_dict = {(src, tgt): nodes for src, tgt, nodes in path_results if nodes}

        for journey in optimized_journeys:
            if len(journey['legs']) > 1:
                multi_leg_trucks += 1
            for leg in journey['legs']:
                path_nodes = path_dict.get((leg['src'], leg['tgt']))
                if path_nodes:
                    path_length_meters = sum(
                        G.edges[u, v, 0]['length'] for u, v in zip(path_nodes[:-1], path_nodes[1:]))
                    total_distance_km += path_length_meters / 1000

        truck_utilisation_rate = (multi_leg_trucks / len(optimized_journeys)) * 100 if optimized_journeys else 0
        print("\n[Operational Efficiency]")
        print(f"   - Total Fleet Distance Travelled: {total_distance_km:.2f} km")
        print(f"   - Truck Utilisation Rate (multi-leg journeys): {truck_utilisation_rate:.2f}%")


def plot_task_prioritization(results: dict):
    """
    Plots a histogram and KDE of the hours when the agent identified the need for a transfer.

    Args:
        results (dict): Dictionary containing simulation results. Must include a 'transfers' key,
                        where each value is a dict with a 'first_hour' field indicating the hour
                        (0-23) when a transfer was first needed.

    Functionality:
        - Extracts the hour of first need for each transfer from the results.
        - Plots a histogram of task density by hour using seaborn.
        - Overlays a KDE curve to show demand trends.
        - Adds axis labels, title, grid, and legend.
        - Handles missing transfer data gracefully.
    """
    print("\n--- Metric 4: Task Prioritisation Analysis ---")
    transfers = results.get("transfers")
    if transfers:
        hours_of_need = [data['first_hour'] for _, data in transfers.items()]
        plt.figure(figsize=(12, 6))
        sns.histplot(hours_of_need, bins=24, binrange=(0, 24), stat='density', color='#FF8040', edgecolor='black',
                     label='Task Density')
        sns.kdeplot(hours_of_need, color='#0046FF', linewidth=2, label='Demand Trend (KDE)')
        plt.xlabel("Hour of Day (24h format)")
        plt.ylabel("Density of Tasks")
        plt.title("Agent Task Prioritisation: Time of Identified Need")
        plt.xticks(range(0, 25, 2))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
    else:
        print("⚠️ No transfer data available to plot task prioritization.")


# ==============================================================================
# Main Analysis Function
# ==============================================================================

def analyse_simulation_results(results_path="simulation_results.pkl", log_directory="/kaggle/working/logs"):
    """
    Loads simulation results and generates all performance metrics.

    Args:
        results_path (str): Path to the pickled simulation results file.
        log_directory (str): Path to the directory containing training and TensorBoard logs.

    Functionality:
        - Loads simulation results from a pickle file.
        - Plots the agent's learning curve from training logs.
        - Calculates and prints training statistics (time, FPS, loss).
        - Computes operational metrics (imbalance, fleet distance, truck utilization).
        - Plots task prioritization analysis (transfer need by hour).
        - Handles missing files and errors gracefully.
    """
    try:
        # Load the simulation results file
        print(f"--- Loading Saved Results from {results_path} ---")
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        print("✅ Successfully loaded simulation data.")

        # Generate each metric
        plot_learning_curve(log_directory)
        calculate_training_stats(log_directory)
        calculate_operational_metrics(results)
        plot_task_prioritization(results)

    except FileNotFoundError:
        print(f"❌ ERROR: Could not find '{results_path}'.")
        print("   Please run the main simulation script first to generate the results file.")
    except Exception as e:
        print(f"An error occurred during metrics calculation: {e}")


if __name__ == "__main__":
    analyse_simulation_results()
