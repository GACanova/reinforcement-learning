"""
This module implements the SarsaAgent class, which uses the Sarsa learning algorithm
to interact with an environment and learn an optimal policy based on the rewards
received for actions taken in states. The Sarsa algorithm is an on-policy reinforcement
learning method that updates action-value estimations for policy improvement.
"""

from collections import defaultdict
import random
import numpy as np


class SarsaAgent:
    """
    Implements the Sarsa learning algorithm, an on-policy reinforcement learning method
    that updates an action-value function based on the current state, action, reward,
    next state, and next action.

    Attributes:
        env: The environment the agent interacts with. The environment should support
             reset() and step(action) methods.
        actions (list): A list of possible actions the agent can take in the environment.
        alpha (float): The learning rate, determining how quickly the agent updates its
                       value estimates with new information.
        epsilon (float): The exploration rate, determining the balance between exploring
                         new actions and exploiting known action values.
        gamma (float): The discount factor, indicating the importance of future rewards
                       compared to immediate rewards.
        state_action_function (defaultdict[float]): A mapping from state-action pairs to
                                                     their value estimates, representing the
                                                     learned policy.

    Methods:
        get_epsilon_greedy_action(state): Selects an action using an epsilon-greedy policy
                                          based on the current estimates of action values.
        run_simulation(n_episodes=1000): Executes a learning simulation over a specified
                                         number of episodes, allowing the agent to update
                                         its policy based on interactions with the environment.
                                         Returns the learned state-action function and a
                                         history of rewards for each episode.
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

    def run_simulation(self, n_episodes=1000):
        self.state_action_function = defaultdict(float)
        history = []

        for t in range(0, n_episodes):
            state = self.env.reset()
            action = self.get_epsilon_greedy_action(state)

            done = False

            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.get_epsilon_greedy_action(next_state)

                self.state_action_function[state, action] += self.alpha * (
                    reward
                    + self.gamma * self.state_action_function[next_state, next_action]
                    - self.state_action_function[state, action]
                )

                state = next_state
                action = next_action

                history.append({"episode": t, "reward": reward})

        return self.state_action_function, history
