"""
Module for the DynaQAgent class, implementing the Dyna-Q reinforcement learning algorithm. The DynaQAgent combines
direct interactions with an environment and simulated experiences for policy improvement. 

Dependencies:
- numpy: For array and numerical operations.
- collections.defaultdict: For initializing dictionaries with default values.
- random: For generating random numbers and selecting random actions.
"""


from collections import defaultdict
import random
import numpy as np


class DynaQAgent:
    """
    Implements the Dyna-Q algorithm for model-based reinforcement learning. This agent integrates direct experience
    with simulated experience to rapidly learn policies in environments. The agent can dynamically update its environment,
    perform actions based on an epsilon-greedy policy, and simulate experiences to update its knowledge.

    Attributes:
        env: The environment the agent interacts with, which must support reset and step functions.
        actions: A list of possible actions in the environment.
        alpha (float): The learning rate.
        epsilon (float): The exploration rate for epsilon-greedy action selection.
        gamma (float): The discount factor for future rewards.
        planning_steps (int): The number of steps to simulate from the learned model for each real step taken in the environment.
        state_action_function (defaultdict): A mapping from state-action pairs to values, representing the learned Q-values.
        model (dict): A learned model of the environment, mapping states (and actions from those states) to next states and rewards.
        history (list): A record of every step taken for all episodes, including the current episode index, timestep, and received reward.
        episode_index (int): The current episode index.
        timestep (int): The current timestep within the current episode.

    Methods:
        update_environment(env): Updates the agent's environment.
        get_epsilon_greedy_action(state): Returns an action based on the epsilon-greedy policy.
        get_max_state_action_value(state): Returns the maximum Q-value for all actions in a given state.
        update_state_action_function(state, action, reward, next_state): Updates the Q-value for a given state-action pair.
        simulate_experience(): Uses the internal model to simulate experiences and update the state-action function accordingly.
        update_model(state, action, reward, next_state): Updates the internal model of the environment based on observed transitions.
        reset_simulation(): Resets the agent's learning and model to initial state.
        run_simulation(n_episodes=100, reset=False, callback=None): Runs the simulation for a specified number of episodes,
            +optionally resetting the agent's state and using a callback function at each timestep.
    """

    def __init__(
        self, env, actions, alpha=0.5, epsilon=0.1, gamma=0.95, planning_steps=50
    ):
        self.env = env
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.planning_steps = planning_steps
        self.state_action_function = defaultdict(float)
        self.model = {}
        self.history = []
        self.episode_index = 0
        self.timestep = 0

    def update_environment(self, env):
        self.env = env

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

    def update_state_action_function(self, state, action, reward, next_state):
        self.state_action_function[state, action] += self.alpha * (
            reward
            + +self.gamma * self.get_max_state_action_value(next_state)
            - self.state_action_function[state, action]
        )

    def simulate_experience(self):
        for _ in range(self.planning_steps):
            state = random.choice(list(self.model.keys()))
            action = random.choice(self.actions)
            reward, next_state = self.model[state][action]

            self.update_state_action_function(state, action, reward, next_state)

    def update_model(self, state, action, reward, next_state):
        if state not in self.model:
            self.model[state] = {action: (reward, next_state)}

            for new_action in self.actions:
                if new_action != action:
                    self.model[state][new_action] = (0, state)
        else:
            self.model[state][action] = (reward, next_state)

    def reset_simulation(self):
        self.state_action_function = defaultdict(float)
        self.model = {}
        self.history = []
        self.episode_index = 0
        self.timestep = 0

    def run_simulation(self, n_episodes=100, reset=False, callback=None):
        if reset:
            self.reset_simulation()

        for _ in range(n_episodes):
            self.episode_index += 1

            state = self.env.reset()
            action = self.get_epsilon_greedy_action(state)

            done = False

            while not done:
                self.timestep += 1

                next_state, reward, done = self.env.step(action)
                next_action = self.get_epsilon_greedy_action(next_state)

                self.update_state_action_function(state, action, reward, next_state)
                self.update_model(state, action, reward, next_state)
                self.simulate_experience()

                state = next_state
                action = next_action

                self.history.append(
                    {
                        "episode": self.episode_index,
                        "timestep": self.timestep,
                        "reward": reward,
                    }
                )

                if callback:
                    callback(self)

        return self.state_action_function, self.history
