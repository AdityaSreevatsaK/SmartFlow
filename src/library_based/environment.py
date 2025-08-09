import random
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import simulate_bike_counts


def compute_reward(
        source_station: str, target_station: str, bike_counts: np.ndarray,
        thresholds: dict, stations_list: list, capacities: dict
) -> float:
    """
    Computes a shaped reward for moving one bike from a source station to a target station.

    Args:
        source_station (str): Name of the source station.
        target_station (str): Name of the target station.
        bike_counts (np.ndarray): Array of current bike counts at each station.
        thresholds (dict): Minimum required bikes at each station.
        stations_list (list): List of all station names.
        capacities (dict): Maximum capacity for each station.

    Returns:
        float: The computed reward value. Returns -10.0 for invalid moves, positive reward for beneficial moves, and negative reward otherwise.
    """
    source_idx = stations_list.index(source_station)
    target_idx = stations_list.index(target_station)

    if bike_counts[source_idx] <= thresholds[source_station]:
        return -10.0
    if bike_counts[target_idx] >= capacities.get(target_station, 100):
        return -10.0

    target_need = thresholds[target_station] - bike_counts[target_idx]
    source_surplus = bike_counts[source_idx] - thresholds[source_station]

    reward = 0.0
    if target_need > 0:
        reward += min(source_surplus, target_need) * 2.0
    else:
        reward -= 5.0
    return reward


class BikeRedistributionEnv(gym.Env):
    def __init__(
            self, stations: List[str], thresholds: Dict[str, int], capacities: Dict[str, int],
            coords: Dict[str, tuple], demand_data: dict, max_steps: int, gamma: float
    ):
        """
        Initializes the BikeRedistributionEnv environment.

        Args:
            stations (List[str]): List of station names.
            thresholds (Dict[str, int]): Minimum required bikes at each station.
            capacities (Dict[str, int]): Maximum capacity for each station.
            coords (Dict[str, tuple]): Coordinates for each station.
            demand_data (dict): Hourly demand data for each station.
            max_steps (int): Maximum number of steps per episode.
            gamma (float): Discount factor for future rewards.

        Sets up the action and observation spaces, and initializes environment state variables.
        """
        super().__init__()
        self.stations = stations
        self.thresholds = thresholds
        self.capacities = capacities
        self.coords = coords
        self.demand_data = demand_data
        self.max_steps = max_steps
        self.gamma = gamma
        self.n_stations = len(stations)

        self.actions = [(i, j) for i in range(self.n_stations) for j in range(self.n_stations) if i != j]
        self.action_space = spaces.Discrete(len(self.actions))

        max_bikes = max(capacities.values()) if capacities else 50
        obs_shape = (self.n_stations + 1,)
        low_bounds = np.zeros(obs_shape, dtype=np.int32)
        high_bounds = np.full(obs_shape, max_bikes, dtype=np.int32)
        high_bounds[-1] = 23

        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, shape=obs_shape, dtype=np.int32)
        self.counts = None
        self.step_count = None
        self.current_hour = 0

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (optional): Random seed for reproducibility.
            options (optional): Additional options for environment reset.

        Returns:
            observation (np.ndarray): The initial observation, consisting of bike counts for each station and the current hour.
            info (dict): An empty dictionary for compatibility with Gymnasium's API.
        """
        super().reset(seed=seed)
        init_counts = simulate_bike_counts(self.thresholds, self.capacities)
        self.counts = np.array([init_counts[s] for s in self.stations], dtype=np.int32)
        self.step_count = 0
        self.current_hour = 0
        observation = np.concatenate([self.counts, [self.current_hour]]).astype(np.int32)
        return observation, {}

    def step(self, action: int):
        """
        Executes one time step within the environment using the specified action.

        Args:
            action (int): The index of the action to perform, representing moving a bike from one station to another.

        Returns:
            observation (np.ndarray): The updated observation, including bike counts for each station and the current hour.
            reward (float): The reward received for the action taken.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Always False (no truncation logic implemented).
            info (dict): Additional information, including the executed action index.

        The function checks if the action is feasible, applies the action (moving a bike), updates bike counts based on demand data,
        enforces station capacity constraints, increments the step count and hour, and determines if the episode should terminate.
        """
        action = int(action)
        feasible_idxs = [idx for idx, (src, tgt) in enumerate(self.actions) if
                         (self.counts[src] > self.thresholds[self.stations[src]]) and (
                                 self.counts[tgt] < self.capacities.get(self.stations[tgt], 100))]

        if not feasible_idxs:
            reward = -2.0
            terminated = True
        else:
            if action not in feasible_idxs:
                action = random.choice(feasible_idxs)
            src_idx, tgt_idx = self.actions[action]
            reward = compute_reward(self.stations[src_idx], self.stations[tgt_idx], self.counts, self.thresholds,
                                    self.stations, self.capacities)
            self.counts[src_idx] -= 1
            self.counts[tgt_idx] += 1
            terminated = False

        info = {"exec_action": action}

        for i, station_name in enumerate(self.stations):
            if station_name in self.demand_data:
                hourly_demand = self.demand_data[station_name].get(self.current_hour, {'departures': 0, 'arrivals': 0})
                self.counts[i] -= round(hourly_demand['departures'])
                self.counts[i] += round(hourly_demand['arrivals'])

        self.counts = np.maximum(0, self.counts)
        station_capacities = np.array([self.capacities.get(s, 100) for s in self.stations], dtype=np.int32)
        self.counts = np.minimum(self.counts, station_capacities)

        self.step_count += 1
        self.current_hour = (self.current_hour + 1) % 24
        if self.step_count >= self.max_steps:
            terminated = True

        observation = np.concatenate([self.counts, [self.current_hour]]).astype(np.int32)
        return observation, reward, terminated, False, info
