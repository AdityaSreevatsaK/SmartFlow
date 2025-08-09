import logging
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .constants import GAMMA, LOG_INTERVAL, MAX_STEPS, TOTAL_TIME_STEPS
from .environment import BikeRedistributionEnv


class VerboseEvalCallback(EvalCallback):
    """
    Custom callback for Stable Baselines3 that logs a message when a new best model is saved during evaluation.

    This callback extends `EvalCallback` and adds logging functionality. When the mean reward from evaluation
    matches the best mean reward so far and the evaluation frequency criteria are met, it logs the path where
    the new best model is saved.

    Inherits from:
        stable_baselines3.common.callbacks.EvalCallback

    Methods:
        _on_step(): Called at each environment step during evaluation. Logs when a new best model is saved.
    """

    def _on_step(self) -> bool:
        """
        Called at each environment step during evaluation.

        Returns:
            bool: True if training should continue, False otherwise.

        Logs a message when a new best model is saved, specifically when the mean reward
        from evaluation matches the best mean reward so far and evaluation frequency criteria are met.
        """
        res = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward == self.best_mean_reward:
                path = os.path.join(self.best_model_save_path, "best_model.zip")
                logging.info(f"✅ Saved new best model to: {path}")
        return res


def train_or_load_model(top, thresholds, capacities, coords_all, demand_data, n_envs, device, model_path=MODEL_PATH):
    """
    Loads a pre-trained DQN model for bike redistribution if available, otherwise trains a new model.

    Args:
        top: List of station indices to include in the environment.
        thresholds: List of threshold values for each station.
        capacities: List of capacity values for each station.
        coords_all: List of coordinates for all stations.
        demand_data: Dictionary containing demand data for each station.
        n_envs: Number of parallel environments to use for training.
        device: Device string for PyTorch (e.g., "cpu" or "cuda").
        model_path: Path to save/load the trained model (default: `dqn_bike_redistrib.zip`).

    Returns:
        DQN: The trained or loaded Stable Baselines3 DQN model.
    """

    class EnvMaker:
        def __init__(self, demand_dict):
            self.demand_dict = demand_dict

        def __call__(self, i=None):
            env = BikeRedistributionEnv(
                stations=top, thresholds=thresholds, capacities=capacities,
                coords=coords_all, demand_data=self.demand_dict,
                max_steps=MAX_STEPS, gamma=GAMMA
            )
            return Monitor(env, filename=f"./logs/env_{i}.csv" if i is not None else None)

    env_maker = EnvMaker(demand_data)
    vec_env = SubprocVecEnv([lambda i=i: env_maker(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([lambda: env_maker(n_envs)])

    if os.path.exists(model_path):
        print(f"✅ Found existing model at: {model_path}. Loading model...")
        model = DQN.load(model_path, env=vec_env, device=device)
        print("   - Model loaded successfully. Skipping training.")
    else:
        print(f"⚠️ No model found at {model_path}. Starting new training run...")

        best_dir = "./logs/best_model/"
        os.makedirs(best_dir, exist_ok=True)
        eval_cb = VerboseEvalCallback(
            eval_env, best_model_save_path=best_dir, log_path="./logs/results/",
            eval_freq=500, n_eval_episodes=5, deterministic=True, verbose=1
        )

        model = DQN(
            policy="MlpPolicy", env=vec_env, policy_kwargs={"net_arch": [128, 128]},
            learning_rate=1e-4, buffer_size=50_000, learning_starts=1000,
            batch_size=32, gamma=GAMMA, train_freq=4, target_update_interval=1000,
            tensorboard_log="./tb/", verbose=1, device=device
        )

        print("\nTraining the new time-aware agent...")
        model.learn(total_timesteps=TOTAL_TIME_STEPS, log_interval=LOG_INTERVAL, callback=eval_cb)
        model.save(model_path)
        print(f"✅ Training finished. Final model saved to: {model_path}")

    return model
