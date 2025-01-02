import numpy as np
import random

class DataExtractionAgent:
    def __init__(self, actions, threshold=0.8, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initializes the agent with RL parameters.
        
        Args:
            actions (list): Possible actions (e.g., different prompt strategies).
            threshold (float): Accuracy threshold for rewarding successful extraction.
            learning_rate (float): Learning rate for Q-learning.
            discount_factor (float): Discount factor for future rewards.
            epsilon (float): Exploration rate for epsilon-greedy policy.
        """
        self.actions = actions
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # Q-values for each state-action pair
    
    def get_state(self, accuracy_score):
        """
        Determines the state based on the accuracy score.
        
        Args:
            accuracy_score (float): The similarity or match score of extracted data.
        
        Returns:
            str: State identifier based on the score range.
        """
        if accuracy_score >= self.threshold:
            return "high_accuracy"
        else:
            return "low_accuracy"
    
    def choose_action(self, state):
        """
        Chooses an action based on the current state, using epsilon-greedy policy.
        
        Args:
            state (str): The current state.
        
        Returns:
            str: Selected action.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: randomly choose an action
            return random.choice(self.actions)
        else:
            # Exploitation: choose the action with the highest Q-value for the current state
            return self.get_best_action(state)
    
    def get_best_action(self, state):
        """
        Returns the best action for a given state based on Q-values.
        
        Args:
            state (str): The current state.
        
        Returns:
            str: Best action for the given state.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning formula.
        
        Args:
            state (str): Current state.
            action (str): Action taken.
            reward (float): Reward received.
            next_state (str): State transitioned into.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.actions}
        
        # Q-learning formula
        best_future_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.learning_rate * (reward + self.discount_factor * best_future_q - current_q)
    
    def get_reward(self, accuracy_score):
        """
        Calculates the reward based on the extraction accuracy.
        
        Args:
            accuracy_score (float): The similarity or match score of extracted data.
        
        Returns:
            float: Reward value.
        """
        if accuracy_score >= self.threshold:
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
        
        # Step 2: Choose an action based on the current state
        action = agent.choose_action(state)
        
        # Step 3: Calculate reward based on accuracy score
        reward = agent.get_reward(score)
        
        # Step 4: Take action and get the next state
        prompt_adjustment = agent.adjust_prompt(action)
        print(f"Action Taken: {action} -> {prompt_adjustment}")
        
        next_state = agent.get_state(score)  # In real scenarios, this would be after prompt adjustment
        
        # Step 5: Update Q-table
        agent.update_q_table(state, action, reward, next_state)
        
        # Logging
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        print(f"Updated Q-Table: {agent.q_table}\n")

# Define actions (prompt strategies) the agent can take
actions = ["strategy_1", "strategy_2", "strategy_3"]

# Instantiate the agent
agent = DataExtractionAgent(actions, threshold=0.85)

# Simulated accuracy scores from extractions
accuracy_scores = [0.75, 0.9, 0.7, 0.95, 0.8, 0.85]

# Run the simulation
simulate_agent_interaction(agent, accuracy_scores)
