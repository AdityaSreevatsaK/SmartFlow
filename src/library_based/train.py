import logging
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .constants import GAMMA, LOG_INTERVAL, MAX_STEPS, MODEL_PATH, TOTAL_TIME_STEPS
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


def train_or_load_model(
        top, thresholds, capacities, coordinates_all, demand_data,
        n_envs, device, model_path=MODEL_PATH, seed_value=0
):
    """
    Loads a pre-trained DQN model for bike redistribution if available, or trains a new one using Stable Baselines3.

    Args:
        top: List of station identifiers.
        thresholds: List of threshold values for each station.
        capacities: List of capacity values for each station.
        coordinates_all: List of coordinates for all stations.
        demand_data: Dictionary containing demand data for the environment.
        n_envs: Number of parallel environments to use for training.
        device: Device to run the model on (e.g., "cpu" or "cuda").
        model_path: Path to save/load the trained model (default: "DQN_Inventory_Model.zip").
        seed_value: Random seed for reproducibility (default: 0).

    Returns:
        model: The trained or loaded DQN model.
    """
    log_dir = f"results/metrics/logs_seed_{seed_value}/"
    best_model_dir = os.path.join(log_dir, "best_model/")
    tb_log_dir = f"results/metrics/tb_seed_{seed_value}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    class EnvMaker:
        # This __init__ method is the constructor. It's needed to "catch" the data.
        def __init__(self, demand_dict):
            self.demand_dict = demand_dict

        # The __call__ method then uses the stored data to create an environment.
        def __call__(self, i=None):
            env = BikeRedistributionEnv(
                stations=top, thresholds=thresholds, capacities=capacities,
                coordinates=coordinates_all, demand_data=self.demand_dict,
                max_steps=MAX_STEPS, gamma=GAMMA
            )
            monitor_path = os.path.join(log_dir, f"env_{i}.csv") if i is not None else None
            return Monitor(env, filename=monitor_path)

    env_maker = EnvMaker(demand_data)
    vec_env = SubprocVecEnv([lambda i=i: env_maker(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([lambda: env_maker(n_envs)])

    if os.path.exists(model_path):
        print(f"✅ Found existing model at: {model_path}. Loading model...")
        model = DQN.load(model_path, env=vec_env, device=device)
        print("   - Model loaded successfully. Skipping training.")
    else:
        print(f"⚠️ No model found at {model_path}. Starting new training run...")
        eval_cb = VerboseEvalCallback(
            eval_env, best_model_save_path=best_model_dir, log_path=os.path.join(log_dir, "results/"),
            eval_freq=500, n_eval_episodes=5, deterministic=True, verbose=1
        )

        model = DQN(
            policy="MlpPolicy", env=vec_env, policy_kwargs={"net_arch": [128, 128]},
            learning_rate=1e-4, buffer_size=50_000, learning_starts=1000,
            batch_size=32, gamma=GAMMA, train_freq=4, target_update_interval=1000,
            tensorboard_log=tb_log_dir, verbose=1, device=device,
            seed=seed_value
        )

        print(f"\\nTraining the new time-aware agent (Seed: {seed_value})...")
        model.learn(total_timesteps=TOTAL_TIME_STEPS, log_interval=LOG_INTERVAL, callback=eval_cb)
        model.save(model_path)
        print(f"✅ Training finished. Final model saved to: {model_path}")

        vec_env.close()

    return model
