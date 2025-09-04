"""
Deep Q-Network (DQN) implementation for training and evaluating an RL agent for continuous State Spaces.

This implementation solves reinforcement learning problems with continuous state spaces by using a neural network model 
to approximate the Q-value function. The Q-value functions estimates the expected cumulative reward for taking each action 
in a given state. The neural network takes the state as input and outputs Q-values for all possible actions.

Key Components:
1. QNetwork: A neural network model that approximates the Q-value function Q(s, a).
2. Experience Replay: A buffer that stores past experiences (state, action, reward, next_state, done) to break correlation
3. Target Network: A separate neural network to stabilize training by providing consistent target Q-values.
4. Epsilon-Greedy Policy: A strategy to balance exploration and exploitation during training.
"""


import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    """
    Neural Network model to approximate the Q-value function.
    The network takes the state as input and outputs Q-values for all possible actions.
    For CartPole, the input is a 4-dimensional state and the output is a 2-dimensional action space.

    Architecture:
    - Input Layer: Size equal to state space (4 for CartPole)
    - Hidden Layers: Two hidden layers with 24 neurons and ReLU activation
    - Output Layer: Size equal to action space (2 for CartPole), outputs Q-values for each action
    """
    def __init__(self, state_size, action_size):
        """
        Initialize the QNetwork model.

        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor

        Returns:
            torch.Tensor: Q-values for each action
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for training and evaluating on environments with continuous state spaces.
    Implements experience replay, target network, and epsilon-greedy policy for exploration.
    Agent interacts with the environment, stores experiences, and learns from them.

    The agent uses the QNetwork to approximate the Q-value function and updates it using the Bellman equation.
    It uses an experience replay buffer to store past experiences and samples mini-batches for training. 
    The experience replay helps to break correlation between consecutive experiences and improves learning stability.
    A target network is used to provide stable target Q-values during training, which is updated periodically.
    """
    def __init__(self, state_size, action_size):
        """
        Initialize the DQNAgent with hyperparameters, networks, and replay buffer.

        Args:
            state_size (int): Dimension of the state space
            action_size (int): Dimension of the action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate. Start with 100% exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration probability
        self.learning_rate = 0.001  # Learning rate for optimizer
        self.batch_size = 64  # Mini-batch size for training
        self.update_target_freq = 5  # Frequency (in episodes) to update the target network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_network()

    def update_target_network(self):
        """
        Update the target network by copying weights from the Q-network.
        This is done periodically to stabilize training by providing consistent target Q-values.
        The target network provide stable Q-value targets during training, which helps to reduce oscillations and divergence.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in the replay buffer.

        Args:
            state (array-like): Current state of the environment
            action (int): Action taken in the current state
            reward (float): Reward received after taking the action
            next_state (array-like): State of the environment after taking the action
            done (bool): Whether the episode has ended after taking the action
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Select an action using an epsilon-greedy policy.

        With probability epsilon, select a random action (exploration).
        Otherwise, select the action with the highest Q-value (exploitation).

        Args:
            state (array-like): Current state of the environment
        Returns:
            int: Selected action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): # Disable gradient calculation for inference
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        """
        Train the Q-network using a mini-batch of experiences from the replay buffer.

        This method:
        1. Samples a random mini-batch of experiences from the replay buffer.
        2. Computes Q-value targets using the target network.
        3. Computes current Q-value estimates using the main network.
        """
        if len(self.memory) < self.batch_size:
            return # Not enough samples to train
        
        minibatch = random.sample(self.memory, self.batch_size)

        # Process each experience in the mini-batch
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            # Calculate target Q-value
            # If the episode is done, target is just the reward
            target = reward
            if not done:
                # If the episode is not done, add discounted max future Q-value from target network
                with torch.no_grad():
                    target += self.gamma * torch.max(self.target_network(next_state)).item()

            # Get current Q-value estimate from the main network
            current_q = self.q_network(state)[0][action]

            # Compute loss between current Q-value and target
            loss = self.loss_fn(current_q, torch.tensor(target).to(self.device))

            # Backpropagation to update the Q-network
            self.optimizer.zero_grad() # Clear previous gradients
            loss.backward() # Compute gradients
            self.optimizer.step() # Update network weights

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# Training the DQN Agent
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]  # 4 for CartPole
    action_size = env.action_space.n  # 2 for CartPole

    agent = DQNAgent(state_size, action_size)
    episodes = 2000 # Number of episodes to train for
    scores = []

    # Training loop
    for episode in range(episodes):
        # Reset the environment at the start of each episode
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)  # Select action using epsilon-greedy policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)  # Store experience
            state = next_state
            total_reward += reward

        scores.append(total_reward)

        if episode % agent.update_target_freq == 0:
            agent.update_target_network()  # Update target network periodically

        print(f"Episode {episode+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Train the agent if we have enough experiences
        if len(agent.memory) > agent.batch_size:
            agent.replay()

    env.close()

    # Plotting the scores over episodes
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Agent Performance on CartPole-v1')
    plt.show()
