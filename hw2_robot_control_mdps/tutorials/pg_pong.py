import gymnasium as gym
# gym.wrappers.FlattenObservation

SEED = 42


def main() -> None:
    # Initialise the environment
    env = gym.make("CartPole-v1", render_mode="human")
    # env = gym.make("ALE/pong-v5", render_mode="human")

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=SEED)
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    episode_over = False
    total_reward = 0

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = (
            env.action_space.sample()
        )  # Random action for now - real agents will be smarter!

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
