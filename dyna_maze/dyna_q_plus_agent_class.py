"""
Module implementing the DynaQPlusAgent class, which uses the Dyna-Q+ reinforcement learning algorithm. Dyna-Q+ 
enhances Dyna-Q by adding an exploration bonus for state-action pairs, promoting exploration of lesser-visited 
states.

Dependencies:
- numpy: For mathematical operations and handling arrays.
- collections.defaultdict: For creating dictionaries that can return a default value if a key hasn't been set.
- random: For generating random numbers and actions.
"""


from collections import defaultdict
import random
import numpy as np


class DynaQPlusAgent:
    """
    Implements the Dyna-Q+ algorithm for reinforcement learning, extending Dyna-Q by incorporating an exploration bonus.
    This bonus encourages the agent to explore not only unvisited states but also states it hasn't visited recently,
    enhancing its exploration efficiency in changing environments.

    Attributes:
        env: The environment the agent interacts with, supporting reset and step functions.
        actions: List of possible actions in the environment.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        gamma (float): Discount factor for future rewards.
        planning_steps (int): Number of steps to simulate from the learned model for each real step taken in the environment.
        exploration_rate (float): Rate at which the exploration bonus decreases with the time since last visit.
        state_action_function (defaultdict): Maps state-action pairs to values (Q-values).
        model (dict): Learned model of the environment, mapping states and actions to next states and rewards.
        last_state_action_update (defaultdict): Tracks the last timestep each state-action pair was updated.
        history (list): Records all steps taken, including episode index, timestep, and received reward.
        episode_index (int): Current episode index.
        timestep (int): Current timestep within the current episode.

    Methods:
        update_environment(env): Updates the agent's environment to a new one.
        get_epsilon_greedy_action(state): Returns an action based on the epsilon-greedy policy.
        get_max_state_action_value(state): Gets the maximum Q-value for all actions in a given state.
        update_state_action_function(state, action, reward, next_state, simulated_experience=False): Updates Q-values with an option
            for exploration bonus during simulated experiences.
        simulate_experience(): Simulates experiences to update the state-action function, including exploration bonus.
        update_model(state, action, reward, next_state): Updates the internal model with observed transitions.
        add_exploration_bonus(state, action): Calculates the exploration bonus for a state-action pair.
        reset_simulation(): Resets the agent's state to begin a new simulation.
        run_simulation(n_episodes=100, reset=False, callback=None): Runs simulation for n_episodes,
            optionally resetting the agent and using a callback function at each timestep.
    """

    def __init__(
        self,
        env,
        actions,
        alpha=0.5,
        epsilon=0.1,
        gamma=0.95,
        planning_steps=50,
        exploration_rate=0.001,
    ):
        self.env = env
        self.actions = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.planning_steps = planning_steps
        self.exploration_rate = exploration_rate
        self.state_action_function = defaultdict(float)
        self.model = {}
        self.last_state_action_update = defaultdict(lambda: 1)
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

    def update_state_action_function(
        self, state, action, reward, next_state, simulated_experience=False
    ):
        if simulated_experience:
            reward += self.add_exploration_bonus(state, action)

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

            self.update_state_action_function(
                state, action, reward, next_state, simulated_experience=True
            )

    def update_model(self, state, action, reward, next_state):
        if state not in self.model:
            self.model[state] = {action: (reward, next_state)}

            for new_action in self.actions:
                if new_action != action:
                    self.model[state][new_action] = (0, state)
        else:
            self.model[state][action] = (reward, next_state)

    def add_exploration_bonus(self, state, action):
        tau = self.timestep - self.last_state_action_update[state, action]
        bonus = self.exploration_rate * np.sqrt(tau)

        return bonus

    def reset_simulation(self):
        self.state_action_function = defaultdict(float)
        self.model = {}
        self.last_state_action_update = defaultdict(lambda: 1)
        self.history = []
        self.episode_index = 0
        self.timestep = 0

    def run_simulation(self, n_episodes=100, reset=False, callback=None):
        if reset:
            self.reset_simulation()

        for _ in range(0, n_episodes):
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
                self.last_state_action_update[state, action] = self.timestep

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
