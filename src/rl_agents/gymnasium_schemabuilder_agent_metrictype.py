
import sys
import os


# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#


import gymnasium as gym
from langchain.output_parsers import RegexParser
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import numpy as np




class GymnasiumAgent:
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__

    def __init__(self, model, env, metric_type="perplexity"):
        self.model = model
        self.env = env
        self.docs = self.get_docs(env)
        self.metric_type = metric_type  # Add metric type to track which environment we're using

        self.instructions = f"""
Your goal is to maximize your return, i.e., the sum of the rewards you receive.
I will give you an observation, reward, termination flag, truncation flag, and the return so far, formatted as:

Observation: <observation>
Reward: <reward>
Termination: <termination>
Truncation: <truncation>
Return: <sum_of_rewards>

The observation is the current {metric_type} score. For perplexity, lower is better. For semantic match, higher is better.
You will respond with an action (0-4), formatted as:

Action: <action>

where you replace <action> with your actual action.
"""
        self.action_parser = RegexParser(
            regex=r"Action: (.*)", output_keys=["action"],
        )

    def interact(self):
        observation, _ = self.env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            # Format observation for better readability
            obs_dict = {
                self.metric_type.capitalize(): observation[0]
            }
            print("\nCurrent State:")
            for metric, value in obs_dict.items():
                print(f"{metric}: {value:.4f}")

            # Generate a response (action) using the model
            response = self.model([
                SystemMessage(content=self.instructions),
                HumanMessage(content=f"""
Current observation: {obs_dict}
Current total reward: {total_reward:.4f}
Task completed: {terminated}

Based on these metrics, what action (0-4) would you take to improve the {self.metric_type}?
Remember to respond ONLY with "Action: <number>"
""")
            ])

            try:
                action = int(self.action_parser.parse(response.content)['action'])
                if not (0 <= action <= 4):
                    raise ValueError("Action must be between 0 and 4")
            except (ValueError, KeyError) as e:
                print(f"Invalid response from model: {response.content}")
                print("Defaulting to action 0")
                action = 0

            # Perform action in the environment
            observation, reward, terminated, info = self.env.step(action)
            total_reward += reward

            print(f"\nAction taken: {action}")
            print(f"Reward: {reward:.4f}")
            print(f"Terminated: {terminated}")
            print(f"Total Return: {total_reward:.4f}")
            print("Metrics:", info)

        print("\nTask completed successfully!")


# # For perplexity-based environment
# env_perplexity = SchemaBuilderEnv(baseprompt, document_text, groundtruth)
# agent_perplexity = GymnasiumAgent(model, env_perplexity, metric_type="perplexity")
# agent_perplexity.interact()

# # For semantic-based environment
# env_semantic = SchemaBuilderEnvSemantic(baseprompt, document_text, groundtruth)
# agent_semantic = GymnasiumAgent(model, env_semantic, metric_type="semantic")
# agent_semantic.interact()