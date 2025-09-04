import numpy as np
import gymnasium as gym


# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
env.reset()

# Initialize Q-table with zeros
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
learning_rate = 0.2
discount_factor = 0.99
num_episodes = 20000
epsilon = 0.9  # Exploration rate
render_threshold = 0.01

# For plotting metrics
rewards_per_episode = np.zeros(num_episodes)

for episode in range(1, num_episodes+1):
    # Reset the environment at the start of each episode
    state, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0

    # Check if we should start rendering
    if epsilon <= render_threshold and env.render_mode is None:
        env.close()
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human')
        print(f"Started rendering at episode {episode} with epsilon={epsilon}")
        env.reset()

    while not (terminated or truncated):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax(q_table[state]) # Exploit: best action from Q-table

        # Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update Q-value using the Q-learning formula
        old_value = q_table[state, action]
        # What is the best Q-value for the next state
        next_max = np.max(q_table[next_state])
        # Calculate the new target estimate
        new_estimate = reward + discount_factor * next_max
        # Update the Q-value towards the new estimate
        q_table[state, action] = old_value + learning_rate * (new_estimate - old_value)

        # Move to the next state
        state = next_state
        total_reward += reward
        rewards_per_episode[episode-1] = total_reward

    if episode % 200 == 0:
        print(f"Episode {episode}: Total Reward: {total_reward} Epsilon: {epsilon:.3f}")
    # Decay epsilon to reduce exploration over time
    epsilon = max(0.01, epsilon * 0.999)
    
    # Reset the environment after the episode ends
    state, info = env.reset()

env.close()

print("Training finished.\n")
# Display the learned Q-table
print("Learned Q-table:")
print(q_table)

# Let us see the success rate of the agent over time
success_rate = np.zeros(num_episodes)
for episode in range(num_episodes):
    # Calculate the success rate as the average reward over the last 100 episodes (rolling avg)
    success_rate[episode] = np.mean(rewards_per_episode[max(0, episode-100):(episode+1)])

import matplotlib.pyplot as plt
import seaborn as sns
plt.plot(success_rate)
plt.xlabel('Episode')
plt.ylabel('Success Rate (Rolling Average over 100 episodes)')
plt.title('Success Rate of Q-Learning Agent on FrozenLake')
plt.show()
