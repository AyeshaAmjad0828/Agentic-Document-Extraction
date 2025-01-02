import numpy as np
import random

class Actor:
    def __init__(self, actions, learning_rate=0.01):
        """
        Initializes the actor for policy approximation.
        
        Args:
            actions (list): List of possible actions (prompt strategies).
            learning_rate (float): Learning rate for the policy updates.
        """
        self.actions = actions
        self.learning_rate = learning_rate
        self.policy = {action: 1 / len(actions) for action in actions}  # Initialize uniform probabilities

    def choose_action(self):
        """
        Chooses an action based on the current policy probabilities.
        
        Returns:
            str: Selected action.
        """
        action = random.choices(list(self.policy.keys()), weights=list(self.policy.values()))[0]
        return action

    def update_policy(self, action, advantage):
        """
        Updates the policy probabilities based on the advantage.
        
        Args:
            action (str): Action taken.
            advantage (float): Advantage calculated by the critic.
        """
        # Update policy probability for the taken action
        for a in self.actions:
            if a == action:
                self.policy[a] += self.learning_rate * advantage * (1 - self.policy[a])
            else:
                self.policy[a] -= self.learning_rate * advantage * self.policy[a]
        # Normalize probabilities to maintain a valid probability distribution
        total = sum(self.policy.values())
        for a in self.actions:
            self.policy[a] /= total

class Critic:
    def __init__(self, learning_rate=0.01, discount_factor=0.9):
        """
        Initializes the critic for value approximation.
        
        Args:
            learning_rate (float): Learning rate for the value updates.
            discount_factor (float): Discount factor for future rewards.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.value_table = {}  # Value estimates for each state

    def get_value(self, state):
        """
        Returns the value estimate for a given state.
        
        Args:
            state (str): The current state.
        
        Returns:
            float: Value estimate for the state.
        """
        return self.value_table.get(state, 0.0)

    def update_value(self, state, reward, next_state):
        """
        Updates the value estimate based on the received reward and the next state.
        
        Args:
            state (str): Current state.
            reward (float): Reward received.
            next_state (str): State transitioned into.
        """
        current_value = self.get_value(state)
        next_value = self.get_value(next_state)
        # TD error (Temporal Difference)
        td_error = reward + self.discount_factor * next_value - current_value
        # Update the value table
        self.value_table[state] = current_value + self.learning_rate * td_error
        return td_error

class DataExtractionAgent:
    def __init__(self, actions, accuracy_threshold=0.8):
        """
        Initializes the agent with actor and critic components.
        
        Args:
            actions (list): Possible actions (prompt strategies).
            accuracy_threshold (float): Accuracy threshold for reward assignment.
        """
        self.actions = actions
        self.accuracy_threshold = accuracy_threshold
        self.actor = Actor(actions)
        self.critic = Critic()

    def get_state(self, accuracy_score):
        """
        Determines the state based on the accuracy score.
        
        Args:
            accuracy_score (float): The similarity or match score of extracted data.
        
        Returns:
            str: State identifier based on the score range.
        """
        if accuracy_score >= self.accuracy_threshold:
            return "high_accuracy"
        else:
            return "low_accuracy"

    def get_reward(self, accuracy_score):
        """
        Calculates the reward based on the extraction accuracy.
        
        Args:
            accuracy_score (float): The similarity or match score of extracted data.
        
        Returns:
            float: Reward value.
        """
        if accuracy_score >= self.accuracy_threshold:
            return 1  # Reward for successful extraction
        else:
            return -1  # Penalty for low accuracy

    def adjust_prompt(self, action):
        """
        Adjusts the prompt engineering strategy based on the chosen action.
        
        Args:
            action (str): Selected action that defines a prompt adjustment strategy.
        
        Returns:
            str: Adjusted prompt based on the chosen strategy.
        """
        # Define your prompt adjustment strategies here
        if action == "strategy_1":
            return "Rephrase prompt for clarity."
        elif action == "strategy_2":
            return "Add additional instructions to specify data fields."
        elif action == "strategy_3":
            return "Use examples in prompt to clarify expected format."
        else:
            return "Default prompt structure."

    def update_agent(self, state, action, reward, next_state):
        """
        Updates the agent's actor and critic based on received feedback.
        
        Args:
            state (str): Current state.
            action (str): Action taken.
            reward (float): Reward received.
            next_state (str): Next state.
        """
        # Update critic and get TD error
        td_error = self.critic.update_value(state, reward, next_state)
        # Use TD error as advantage for the actor
        self.actor.update_policy(action, td_error)

# Simulation function to illustrate agent's behavior
def simulate_agent_interaction(agent, accuracy_scores):
    """
    Simulates the agent's behavior in a sequence of extractions with varying accuracy scores.
    
    Args:
        agent (DataExtractionAgent): The reinforcement learning agent.
        accuracy_scores (list): List of extraction accuracy scores for simulation.
    """
    for score in accuracy_scores:
        # Step 1: Determine the current state
        state = agent.get_state(score)
        
        # Step 2: Choose an action using the actor's policy
        action = agent.actor.choose_action()
        
        # Step 3: Calculate reward based on accuracy score
        reward = agent.get_reward(score)
        
        # Step 4: Take action and get the next state
        prompt_adjustment = agent.adjust_prompt(action)
        print(f"Action Taken: {action} -> {prompt_adjustment}")
        
        next_state = agent.get_state(score)  # In real scenarios, this would be after prompt adjustment
        
        # Step 5: Update agent (actor and critic)
        agent.update_agent(state, action, reward, next_state)
        
        # Logging
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        print(f"Updated Policy: {agent.actor.policy}")
        print(f"Updated Value Table: {agent.critic.value_table}\n")

# Define actions (prompt strategies) the agent can take
actions = ["strategy_1", "strategy_2", "strategy_3"]

# Instantiate the agent
agent = DataExtractionAgent(actions, accuracy_threshold=0.85)

# Simulated accuracy scores from extractions
accuracy_scores = [0.75, 0.9, 0.7, 0.95, 0.8, 0.85]

# Run the simulation
simulate_agent_interaction(agent, accuracy_scores)
