"""
Implementation of a Semi-Gradient SARSA Agent for Reinforcement Learning.
"""

from collections import defaultdict
import random
import numpy as np

from config import ACTIONS


class SemiGradientSarsaAgent:
    """
    A Semi-Gradient SARSA Agent for reinforcement learning tasks.

    This class implements the semi-gradient version of the SARSA (State-Action-Reward-State-Action)
    algorithm, a fundamental method in reinforcement learning that updates the policy based on the current
    policy's performance. The agent uses a tile coding approach for state representation, which helps in
    handling continuous state spaces.

    Attributes:
        env (object):
            An environment object complying with the OpenAI Gym interface.
        tile_coder (object):
            An instance of a tile coder that provides a discretized representation of states.
        actions (tuple):
            A tuple of possible actions in the environment.
        alpha (float):
            The learning rate.
        epsilon (float):
            The exploration rate, determining the trade-off between exploration and exploitation.
        gamma (float):
            The discount factor, weighing the importance of future rewards.
        value_function_weights (defaultdict):
            The learned weights for the state-action value function.
        episode_index (int):
            Counter for the number of episodes the agent has been trained.

    Methods:
        get_state_action_value(state, action):
            Returns the estimated value of a given state-action pair.
        get_epsilon_greedy_action(state):
            Selects an action based on the epsilon-greedy policy.
        update_value_function_weights(state, action, reward, next_state, next_action, done):
            Updates the weights of the value function based on the observed transition.
        calculate_state_action_function():
            Computes the state-action value function for all states and actions after training.
        run_simulation(n_episodes, reset):
            Runs a specified number of episodes for training the agent.
        reset_simulation():
            Resets the agent's state and learned weights to initial conditions for retraining.
    """

    def __init__(
        self, env, tile_coder, actions=ACTIONS, alpha=0.1, epsilon=0.1, gamma=1.0
    ):
        self.env = env
        self.tile_coder = tile_coder
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.value_function_weights = defaultdict(float)
        self.episode_index = 0
        self.history = []

    def get_state_action_value(self, state, action):
        indices = self.tile_coder.get_tile_indices(state)

        state_action_value = 0

        for index in indices:
            state_action_value += self.value_function_weights[index, action]

        return state_action_value

    def get_epsilon_greedy_action(self, state):
        if np.random.rand() < self.epsilon:
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

                if (
                    state[0] < self.env.position_bound[0]
                    or state[0] >= self.env.position_bound[1]
                ):
                    state_action_function[(state, action)] = 0

        return state_action_function

    def run_simulation(self, n_episodes=1000, reset=False):
        if reset:
            self.reset_simulation()

        for episode in range(0, n_episodes):
            self.episode_index += 1
            timestep = 0
            state = self.env.reset()
            action = self.get_epsilon_greedy_action(state)

            done = False

            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.get_epsilon_greedy_action(next_state)

                self.update_value_function_weights(
                    state, action, reward, next_state, next_action, done
                )

                state = next_state
                action = next_action

                timestep += 1
                self.history.append(
                    {
                        "episode": self.episode_index,
                        "timestep": timestep,
                        "reward": reward,
                    }
                )

        state_action_function = self.calculate_state_action_function()

        return state_action_function, self.history

    def reset_simulation(self):
        self.value_function_weights = defaultdict(float)
        self.history = []
        self.episode_index = 0
        self.timestep = 0
