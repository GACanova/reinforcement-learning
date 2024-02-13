"""
Implements the QLearningAgent class for reinforcement learning using Q-learning.
"""

from collections import defaultdict
import random
import numpy as np


class QLearningAgent:
    """
    Implements a Q-Learning agent for a specified environment. This agent
    learns to make decisions based on the Q-learning algorithm, aiming to
    maximize the cumulative reward over time through interaction with the
    environment.

    Attributes:
        env: The environment the agent interacts with, which must have `reset`
             and `step` methods compatible with the agent's expectations.
        actions (list): A list of possible actions the agent can take in the
                        environment.
        alpha (float): Learning rate, determines to what extent newly acquired
                       information overrides old information.
        epsilon (float): Exploration rate, the probability of choosing a random
                         action instead of the best action according to the
                         current policy.
        gamma (float): Discount factor, represents the importance of future rewards.
        state_action_function (defaultdict of float): A mapping from state-action
                                                      pairs to Q-values.

    Methods:
        get_epsilon_greedy_action(state): Chooses an action based on the epsilon-greedy
                                          policy for a given state.
        get_max_state_action_value(state): Returns the maximum Q-value for any action
                                           in the given state.
        run_simulation(n_episodes=1000): Runs a simulation of the environment over a
                                         specified number of episodes, updating the
                                         agent's knowledge of the state-action function.
    """

    def __init__(self, env, actions, alpha=0.5, epsilon=0.1, gamma=1.0):
        self.env = env
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_action_function = None

    def get_epsilon_greedy_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)

        action_values = np.array(
            [self.state_action_function[(state, action)] for action in self.actions]
        )
        best_action_index = np.argmax(
            np.random.random(action_values.shape[0])
            * (action_values == action_values.max())
        )  # random tiebreaker

        return self.actions[best_action_index]

    def get_max_state_action_value(self, state):
        action_values = np.array(
            [self.state_action_function[(state, action)] for action in self.actions]
        )

        return np.max(action_values)

    def run_simulation(self, n_episodes=1000):
        self.state_action_function = defaultdict(float)
        history = []

        for episode in range(0, n_episodes):
            state = self.env.reset()
            action = self.get_epsilon_greedy_action(state)

            done = False

            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.get_epsilon_greedy_action(next_state)

                self.state_action_function[state, action] += self.alpha * (
                    reward
                    + self.gamma * self.get_max_state_action_value(next_state)
                    - self.state_action_function[state, action]
                )

                state = next_state
                action = next_action

                history.append({"episode": episode, "reward": reward})

        return self.state_action_function, history
