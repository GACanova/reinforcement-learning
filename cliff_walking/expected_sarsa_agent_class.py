"""
Provides the ExpectedSarsaAgent class, implementing the Expected SARSA algorithm for
reinforcement learning.
"""

from collections import defaultdict
import random
import numpy as np


class ExpectedSarsaAgent:
    """
    Implements the Expected SARSA learning algorithm for reinforcement learning tasks.
    This agent learns optimal actions in an environment by averaging over all possible
    next actions' Q-values, rather than selecting a single next action, for updating
    Q-values. This results in a more stable learning process under certain conditions.

    Attributes:
        env: Environment to interact with, requiring `reset` and `step` functions.
        actions (list): Possible actions in the environment.
        alpha (float): Learning rate, adjusting the impact of new information.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        gamma (float): Discount factor for future rewards.
        state_action_function (defaultdict(float)): Estimated Q-values for state-action pairs.

    Methods:
        get_epsilon_greedy_action(state): Selects action using epsilon-greedy policy.
        get_expected_state_action_value(state): Computes expected Q-value in a state.
        run_simulation(n_episodes=1000): Runs episodes, updating Q-values and rewards.
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

    def get_expected_state_action_value(self, state):
        action_values = np.array(
            [self.state_action_function[(state, action)] for action in self.actions]
        )

        return np.mean(action_values)

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
                    + self.gamma * self.get_expected_state_action_value(next_state)
                    - self.state_action_function[state, action]
                )

                state = next_state
                action = next_action

                history.append({"episode": episode, "reward": reward})

        return self.state_action_function, history
