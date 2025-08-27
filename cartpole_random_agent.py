import time
import gymnasium as gym

TIMESTEPS = 200

# Create the CartPole environment with rendering enabled
# Render mode "human" opens a window to visualize the environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment to its initial state. Returns the starting state.
state, _ = env.reset()

try:
    # Run the environment for a maximum of TIMESTEPS timesteps (it might end earlier)
    for timestep in range(TIMESTEPS):
        # Render the current state of the environment (show the animation)
        env.render()
        time.sleep(1)

        # Sample a random action from the action space
        # The action space for CartPole is discrete with two possible actions: 0 (move left) and 1 (move right)
        # Random agent policy: choose an action uniformly at random (0=left, 1=right)
        action = env.action_space.sample()

        # Take the action in the environment and observe the results
        # The agent interacts with the environment by taking an action
        # The environment returns the next state, reward, done flag, and additional information
        # next_state: the state after taking the action
        # reward: the reward received after taking the action
        # done: a boolean flag indicating if the episode has ended (e.g., pole fell or max steps reached) True if episode ended
        # info: a dictionary containing additional information about the environment
        next_state, reward, done, info, _ = env.step(action)

        print(f"Timestep: {timestep}, Action: {action}, Reward: {reward}, Done: {done}, State: {next_state}")

        # If the episode is done (e.g., pole fell or max steps reached), reset the environment
        if done:
            state, _ = env.reset()
            break
finally:
    # Close the environment and the rendering window when done
    env.close()