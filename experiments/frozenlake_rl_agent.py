import os
import time
import pickle
import numpy as np
import gymnasium as gym


class FrozenLakeAgent:
    def __init__(self, env_name="FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode=None):
        self.env_name = env_name
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        self.env = gym.make(env_name, map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.learning_rate = 0.2 # alpha: Controls how much new information overrides old information
        self.discount_factor = 0.99 # beta: Importance of future rewards
        self.epsilon = 0.9  # Exploration rate. Starts high and decays over time
        self.num_episodes = 20000
        self.rewards_per_episode = np.zeros(self.num_episodes)
        self.epsilon_decay = 0.999
        self.epsilon_threshold = 0.0001

    def train(self):
        for episode in range(1, self.num_episodes + 1):
            state, info = self.env.reset()
            terminated, truncated = False, False
            total_reward = 0

            while not (terminated or truncated):
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()  # Explore: random action
                else:
                    action = np.argmax(self.q_table[state])  # Exploit: best action from Q-table

                next_state, reward, terminated, truncated, info = self.env.step(action)

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_estimate = reward + self.discount_factor * next_max
                self.q_table[state, action] = old_value + self.learning_rate * (new_estimate - old_value)

                state = next_state
                total_reward += reward
                self.rewards_per_episode[episode - 1] = total_reward

            if episode % 1000 == 0:
                print(f"Episode {episode}: Total Reward: {total_reward} Epsilon: {self.epsilon:.4f}")

            # Decay epsilon to reduce exploration over time
            self.epsilon = max(self.epsilon_threshold, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def predict(self, num_episodes=10, render=False):
        if render:
            self.env.close()
            self.env = gym.make(self.env_name, map_name=self.map_name,
                                is_slippery=self.is_slippery, render_mode="human")

        results = []
        for episode in range(num_episodes):
            print(f"Predicting episode {episode + 1}/{num_episodes}")
            state, info = self.env.reset()
            terminated, truncated = False, False
            total_reward = 0

            while not (terminated or truncated):
                action = np.argmax(self.q_table[state])  # greedy
                new_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                state = new_state

                if render:
                    self.env.render()

            results.append({"Episode": episode, "Reward": total_reward})

        return results


if __name__ == "__main__":
    agent = FrozenLakeAgent()
    rewards = agent.train()

    model_filename = "frozenlake_agent.pkl"
    model_path = os.path.join("models", model_filename)
    agent.save(model_path)
    print("Training completed and model saved.")

    loaded_agent = FrozenLakeAgent.load(model_path)
    results = loaded_agent.predict(num_episodes=5, render=True)
    for r in results:
        print(r)
