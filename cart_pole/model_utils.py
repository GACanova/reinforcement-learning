"""
Provides functions and classes for training and evaluating reinforcement learning agents with optional callback support.
"""
from collections import deque
import numpy as np


def train(
    agent,
    train_episodes=1000,
    evaluation_episodes=10,
    evaluation_interval=100,
    callbacks=None,
    verbose=True,
):
    """
    Trains an agent over a specified number of episodes and evaluates it periodically.

    Args:
        agent: The agent to be trained and evaluated.
        train_episodes (int): Total number of training episodes.
        evaluation_episodes (int): Number of episodes to run during each evaluation.
        evaluation_interval (int): Number of episodes between each evaluation.
        callbacks (list of callables): Optional callbacks to execute during training.
        verbose (bool): If True, prints detailed progress.

    Returns:
        list: A list of dictionaries containing details about each training episode.

    The function trains the agent by repeatedly invoking its `step` method, evaluates its
    performance by calling the `evaluate` function, and applies any specified callbacks.
    """

    if callbacks is None:
        callbacks = []
    elif not isinstance(callbacks, list):
        callbacks = [callbacks]

    results = []

    for episode in range(1, train_episodes + 1):
        rewards, timesteps = agent.step()

        results.append({"episode": episode, "timesteps": timesteps, "rewards": rewards})

        if evaluation_interval and episode % evaluation_interval == 0:
            evaluation_rewards, evaluation_timesteps = evaluate(
                agent, episodes=evaluation_episodes
            )
            score = np.mean(evaluation_rewards)

            if verbose:
                print(f"Episode {episode}/{train_episodes}: Evaluation Score = {score}")

            for callback in callbacks:
                if callback(evaluation_rewards, evaluation_timesteps):
                    return results

    return results


def evaluate(agent, episodes=1):
    """
    Evaluates the agent's performance over a given number of episodes.

    Args:
        agent: The agent to be evaluated.
        episodes (int): Number of evaluation episodes.

    Returns:
        tuple: Two lists containing total rewards and timesteps per episode, respectively.

    This function assesses the agent by executing its `step` method with training
    set to False and aggregates the rewards and timesteps for each episode.
    """

    total_rewards = []
    total_timesteps = []

    for episode in range(0, episodes):
        rewards, timesteps = agent.step(train=False)
        total_rewards.append(np.sum(rewards))
        total_timesteps.append(np.sum(timesteps))

    return total_rewards, total_timesteps


class EarlyStopCallback:
    """
    A callback to stop training when all agents exceed a specified reward threshold.

    Attributes:
        threshold (int): The minimum reward threshold for early stopping.

    This callback is typically used during training to halt the training process
    if the agent consistently achieves a high level of performance.
    """

    def __init__(self, threshold=200):
        self.threshold = threshold

    def __call__(self, rewards, timesteps):
        if all(reward > self.threshold for reward in rewards):
            print(
                f"All agents reached the minimum reward threshold of {self.threshold}."
            )
            return True
        return False


class DivergenceCallback:
    """
    A callback to detect lack of improvement over a specified number of evaluations.

    Attributes:
        window_size (int): Number of evaluations to consider for detecting divergence.
        relative_threshold (float): The minimum relative improvement required to reset the no improvement count.
        no_improvement_threshold (int): Number of consecutive evaluations with insufficient improvement to trigger a stop.

    This callback is designed to halt the training if the model does not show significant improvement
    over a defined number of evaluations, indicating potential overfitting or convergence issues.
    """

    def __init__(
        self, window_size=10, relative_threshold=0.01, no_improvement_threshold=5
    ):
        self.scores = deque(maxlen=window_size)
        self.relative_threshold = relative_threshold
        self.window_size = window_size
        self.no_improvement_threshold = no_improvement_threshold
        self.best_score = -np.inf
        self.no_improvement_count = 0

    def __call__(self, rewards, timesteps):
        score = np.mean(rewards)
        self.scores.append(score)

        if len(self.scores) == self.window_size:
            recent_avg = np.mean(self.scores)
            past_avg = np.mean(list(self.scores)[: self.window_size // 2])

            # Check if there has been no significant relative improvement
            if (
                past_avg != 0
                and (recent_avg - past_avg) / past_avg < self.relative_threshold
            ):
                self.no_improvement_count += 1
                print(
                    f"No significant improvement detected. Recent avg: {recent_avg}, Past avg: {past_avg}, Count: {self.no_improvement_count}"
                )
            else:
                self.no_improvement_count = 0

            if self.no_improvement_count >= self.no_improvement_threshold:
                print(
                    f"Model has not shown significant improvement for {self.no_improvement_threshold} evaluation intervals."
                )
                return True

        return False
