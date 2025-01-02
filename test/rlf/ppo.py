!pip install torch


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Actor network to predict actions based on current state.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        """
        Critic network to estimate the value function.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.2):
        """
        PPO Agent with actor and critic networks.
        """
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        """
        Choose action based on the policy (actor network).
        """
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs

    def compute_advantage(self, rewards, values):
        """
        Compute advantage using Generalized Advantage Estimation (GAE).
        """
        advantage = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * gae
            advantage.insert(0, gae)
        return torch.tensor(advantage, dtype=torch.float32)

    def update(self, states, actions, rewards, next_states):
        """
        Update policy and value networks using PPO.
        """
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        advantage = self.compute_advantage(rewards, values.tolist() + [next_values[-1]])

        # Update actor network using PPO objective
        old_action_probs = self.actor(states).gather(1, actions.unsqueeze(-1)).squeeze()
        for _ in range(10):  # PPO update iterations
            action_probs = self.actor(states).gather(1, actions.unsqueeze(-1)).squeeze()
            ratio = action_probs / (old_action_probs + 1e-10)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        # Update critic network to minimize the TD error
        critic_loss = (rewards + self.gamma * next_values - values).pow(2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

# Simulation function for the PPO agent with document extraction task
def simulate_agent(agent, accuracy_scores, num_epochs=5):
    for epoch in range(num_epochs):
        states, actions, rewards, next_states = [], [], [], []
        
        for i in range(len(accuracy_scores) - 1):
            state = [accuracy_scores[i]]
            next_state = [accuracy_scores[i + 1]]
            
            action, _ = agent.choose_action(state)
            reward = 1 if accuracy_scores[i] >= agent.gamma else -1
            
            # Store in memory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
        
        # Update agent
        agent.update(states, actions, rewards, next_states)
        
        # Logging
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("Updated Policy:", [agent.actor(torch.tensor(s, dtype=torch.float32)).detach().numpy() for s in states])

# Define state and action dimensions based on accuracy scores and prompt strategies
state_dim = 1  # Using accuracy score as the state
action_dim = 3  # Assume three prompt strategies for illustration

# Instantiate the PPO agent
agent = PPOAgent(state_dim, action_dim)

# Simulated accuracy scores
accuracy_scores = [0.8, 0.9, 0.75, 0.95, 0.82, 0.88]

# Simulate agent behavior
simulate_agent(agent, accuracy_scores)
