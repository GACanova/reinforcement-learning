"""
Provides utilities for training reinforcement learning agents and analyzing their performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import train, EarlyStopCallback


def plot_rewards_per_episode(results, threshold=200):
    """
    Plots the rewards per episode with a threshold line indicating a performance benchmark.

    Args:
        results (list of dicts): A list containing dictionaries with 'episode' and 'rewards' keys.
        threshold (int, optional): The reward threshold to highlight on the plot. Default is 200.

    This function generates a line plot of rewards earned per episode and adds a horizontal
    line to denote the specified reward threshold, providing visual insight into the agent's
    performance relative to this threshold.
    """

    episodes = [x["episode"] for x in results]
    rewards = [x["rewards"] for x in results]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plt.xlabel("Episodes", fontsize=18)
    plt.ylabel("Rewards", fontsize=18)
    plt.plot(episodes, rewards, label="Rewards per Episode")
    plt.axhline(
        y=threshold, color="r", linestyle="--", label=f"Threshold at {threshold}"
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.legend()
    plt.show()


def time_to_learn(agent, iterations=100):
    """
    Measures the time (in episodes) taken for an agent to learn across specified iterations.

    Args:
        agent: The agent being trained and evaluated.
        iterations (int, optional): The number of times the training process is repeated. Default is 100.

    Returns:
        list: A list of integers representing the episode count at which the agent stopped training
              each iteration due to early stopping criteria.

    Each iteration involves training the agent until an early stopping criterion is met. The function
    tracks the episode count of the final training episode across all iterations.
    """

    time_to_learn_data = []

    early_stop_callback = EarlyStopCallback(threshold=200)

    for _ in range(iterations):
        agent.reset_simulation()

        results = train(
            agent,
            train_episodes=2000,
            evaluation_episodes=10,
            evaluation_interval=1,
            callbacks=early_stop_callback,
            verbose=False,
        )

        time_to_learn_data.append(results[-1]["episode"])

    return time_to_learn_data


def plot_time_to_learn(
    sarsa_ttl, dqn_ttl, actor_critic_ttl, continuous_actor_critic_ttl
):
    """
    Plots the distribution of time to learn across different agent types using KDE plots.

    Args:
        sarsa_ttl (list): Time to learn data for the SARSA agent.
        dqn_ttl (list): Time to learn data for the DQN agent.
        actor_critic_ttl (list): Time to learn data for the Actor-Critic agent.
        continuous_actor_critic_ttl (list): Time to learn data for the Continuous Actor-Critic agent.

    This function generates KDE plots for each agent type's time to learn, displaying distributions
    and highlighting the mean episodes to learn with vertical lines. This comparison helps in
    visualizing learning efficiency and consistency across different types of agents.
    """

    plt.figure(figsize=(10, 6))

    sns.kdeplot(sarsa_ttl, color="blue", label="SARSA", alpha=0.4, fill=True)
    sns.kdeplot(dqn_ttl, color="red", label="DQN", alpha=0.4, fill=True)
    sns.kdeplot(
        actor_critic_ttl, color="green", label="Actor-Critic", alpha=0.4, fill=True
    )
    sns.kdeplot(
        continuous_actor_critic_ttl,
        color="purple",
        label="Continuous Actor-Critic",
        alpha=0.4,
        fill=True,
    )

    plt.xlim([0, 600])

    mean_sarsa = sum(sarsa_ttl) / len(sarsa_ttl)
    mean_dqn = sum(dqn_ttl) / len(dqn_ttl)
    mean_actor_critic = sum(actor_critic_ttl) / len(actor_critic_ttl)
    mean_cont_actor_critic = sum(continuous_actor_critic_ttl) / len(
        continuous_actor_critic_ttl
    )

    plt.axvline(
        x=mean_sarsa,
        color="blue",
        linestyle="--",
        label=f"Mean SARSA: {round(mean_sarsa, 2)}",
    )
    plt.axvline(
        x=mean_dqn, color="red", linestyle="--", label=f"Mean DQN: {round(mean_dqn, 2)}"
    )
    plt.axvline(
        x=mean_actor_critic,
        color="green",
        linestyle="--",
        label=f"Mean Actor-Critic: {round(mean_actor_critic, 2)}",
    )
    plt.axvline(
        x=mean_cont_actor_critic,
        color="purple",
        linestyle="--",
        label=f"Mean Continuous Actor-Critic: {round(mean_cont_actor_critic, 2)}",
    )

    plt.legend()
    plt.title("Distribution of Time to Learn", fontsize=18)
    plt.xlabel("Episodes", fontsize=18)
    plt.ylabel("Density", fontsize=18)

    plt.show()
