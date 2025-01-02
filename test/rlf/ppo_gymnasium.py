#pip install stable-baselines3
#pip install gymnasium



import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env

# Define a custom environment for the extraction process
class DataExtractionEnv(gym.Env):
    def __init__(self):
        super(DataExtractionEnv, self).__init__()
        # Define action and observation space
        # Action space: Adjustments to prompt engineering (discrete for simplicity)
        self.action_space = gym.spaces.Discrete(5)  # E.g., 5 different adjustments
        # Observation space: Accuracy metrics (e.g., Exact Match Score, Similarity Score)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        # Initialize state and parameters
        self.state = None
        self.reset()

    def step(self, action):
        # Simulate feedback based on action
        exact_match_score = self.state[0] + np.random.uniform(-0.05, 0.1)  # Random adjustment
        similarity_score = self.state[1] + np.random.uniform(-0.05, 0.1)  # Random adjustment

        # Penalize bad adjustments (actions not improving scores)
        reward = exact_match_score * similarity_score - abs(action - 2) * 0.05

        # Update state
        self.state = np.clip([exact_match_score, similarity_score], 0, 1)

        # Check if the task is complete
        done = bool(self.state[0] >= 0.95 and self.state[1] >= 0.95)

        # Provide additional info (can be empty)
        info = {}
        return np.array(self.state), reward, done, info

    def reset(self):
        # Reset state to initial values
        self.state = np.random.uniform(0.2, 0.4, size=(2,))
        return np.array(self.state)

# Create and wrap the environment
env = make_vec_env('DataExtractionEnv',n_envs=4, seed=0)

# Define the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model1 = DQN("MlpPolicy", env, verbose=1)


# Train the agent
print("Training the agent...")
model.learn(total_timesteps=10000)

# Evaluate the trained agent
print("Evaluating the trained agent...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Save the model
model.save("ppo_data_extraction_agent")
print("Model saved as 'ppo_data_extraction_agent'.")

# Test the agent
print("Testing the agent...")
obs = env.reset()
for _ in range(50):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("Task completed successfully!")
        break
