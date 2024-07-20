"""
Implementation of a Semi-Gradient SARSA Agent for Reinforcement Learning.
"""

from collections import defaultdict
import random
import numpy as np


class SemiGradientSarsaAgent:
    """
    Implements a Semi-Gradient SARSA learning agent using tile coding for
    function approximation in reinforcement learning.

    This agent updates its value function based on the semi-gradient SARSA
    update rule, suitable for environments with continuous state spaces.

    Attributes:
        env (object): Environment where the agent operates.
        actions (list): Possible actions in the environment.
        tile_coder (object): Manages state representation using tile coding.
        epsilon (callable): Policy for action randomness.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        value_function_weights (defaultdict): Weights for the value function.

    Methods:
        get_state_action_value(state, action):
            Returns the value of a state-action pair.
        select_action(state, use_epsilon=True):
            Selects an action based on the current policy.
        update_value_function_weights(state, action, reward, next_state,
                                      next_action, done):
            Updates weights based on transitions observed.
        calculate_state_action_function():
            Computes the value function for all state-action pairs.
        step(train=True):
            Performs a step in the environment, optionally training the agent.
        reset_simulation():
            Resets the agent's internal state and value function.
    """

    def __init__(
        self,
        env,
        actions,
        tile_coder,
        epsilon,
        alpha=0.1,
        gamma=1.0,
    ):
        self.env = env
        self.actions = actions
        self.tile_coder = tile_coder
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.value_function_weights = defaultdict(float)

    def get_state_action_value(self, state, action):
        indices = self.tile_coder.get_tile_indices(state)

        state_action_value = 0

        for index in indices:
            state_action_value += self.value_function_weights[index, action]

        return state_action_value

    def select_action(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() < self.epsilon():
            return random.choice(self.actions)

        action_values = np.array(
            [self.get_state_action_value(state, action) for action in self.actions]
        )
        best_action_index = np.argmax(
            np.random.random(action_values.shape[0])
            * (action_values == action_values.max())
        )  # random tiebreaker

        return self.actions[best_action_index]

    def update_value_function_weights(
        self, state, action, reward, next_state, next_action, done
    ):
        if done:
            next_state_action_function = 0
        else:
            next_state_action_function = self.get_state_action_value(
                next_state, next_action
            )

        error = (
            reward
            + self.gamma * next_state_action_function
            - self.get_state_action_value(state, action)
        )

        indices = self.tile_coder.get_tile_indices(state)

        for index in indices:
            self.value_function_weights[index, action] += (
                self.alpha * error / len(indices)
            )

    def calculate_state_action_function(self):
        state_action_function = defaultdict(float)
        states = self.env.generate_state_grid()

        for action in self.actions:
            for state in states:
                state_action_function[(state, action)] = self.get_state_action_value(
                    state, action
                )

        return state_action_function

    def step(self, train=True):
        timesteps = 0
        rewards = 0
        state = self.env.reset()
        action = self.select_action(state, use_epsilon=train)

        done = False

        while not done:
            next_state, reward, done = self.env.step(action)
            next_action = self.select_action(next_state, use_epsilon=train)

            if train:
                self.update_value_function_weights(
                    state, action, reward, next_state, next_action, done
                )

            state = next_state
            action = next_action

            timesteps += 1
            rewards += reward

        self.epsilon.update()

        return rewards, timesteps

    def reset_simulation(self):
        self.value_function_weights = defaultdict(float)
        self.epsilon.reset()
