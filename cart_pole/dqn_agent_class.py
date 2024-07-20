"""
Implements a reinforcement learning module using Deep Q-Networks (DQN).
"""

import numpy as np


class DQNAgent:
    """
    Implements a Deep Q-Network agent using experience replay and an epsilon-greedy policy.

    Attributes:
        env (object): Environment where the agent operates.
        actions (list): Possible actions the agent can take.
        model_handler (object): Manages neural network operations like prediction and training.
        replay_memory (object): Stores and samples transitions.
        epsilon (callable): Manages epsilon value for the epsilon-greedy strategy.
        gamma (float): Discount factor for future rewards, default is 1.0.
        batch_size (int): Number of transitions to sample from memory for training, default is 128.

    Methods:
        update_replay_memory(state, action_index, reward, next_state, done):
            Stores transitions in memory.

        predict_q_values(state):
            Returns Q-values for a given state using the model.

        get_max_q_values(states):
            Returns the maximum Q-value for each state in a batch.

        replay():
            Updates the model by sampling from replay memory and training on the batch.

        select_action(state, use_epsilon=True):
            Selects an action using epsilon-greedy strategy or the highest Q-value.

        step(train=True, max_timesteps=None):
            Executes actions in the environment until done or max timesteps are reached.

        reset_simulation():
            Resets simulation, clears memory, resets model weights, and resets epsilon.
    """

    def __init__(
        self,
        env,
        actions,
        model_handler,
        replay_memory,
        epsilon,
        gamma=1.0,
        batch_size=128,
    ):
        self.env = env
        self.model_handler = model_handler
        self.replay_memory = replay_memory
        self.actions = actions
        self.epsilon = epsilon
        self.num_actions = len(self.actions)
        self.gamma = gamma
        self.batch_size = batch_size

    def update_replay_memory(self, state, action_index, reward, next_state, done):
        self.replay_memory.store((state, action_index, reward, next_state, done))

    def predict_q_values(self, state):
        state = np.array(state)

        return self.model_handler.predict(state)

    def get_max_q_values(self, states):
        q_values = self.predict_q_values(states)

        return np.max(q_values, axis=-1)

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)

        states, actions, rewards, next_states, done = (
            np.array([transition[i] for transition in transitions])
            for i in range(len(transitions[0]))
        )

        target_q_values = rewards + (1 - done) * self.gamma * self.get_max_q_values(
            next_states
        )

        current_q_values = self.predict_q_values(states)
        current_q_values[np.arange(self.batch_size), actions] = target_q_values

        self.model_handler.train(states, current_q_values)

    def select_action(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() < self.epsilon():
            return np.random.choice(self.num_actions)

        return np.argmax(self.predict_q_values(state))

    def step(self, train=True, max_timesteps=None):
        timesteps = 0
        rewards = 0
        state = self.env.reset()

        action_index = self.select_action(state, use_epsilon=train)

        done = False

        while not done and (max_timesteps is None or timesteps < max_timesteps):
            next_state, reward, done = self.env.step(self.actions[action_index])
            next_action_index = self.select_action(next_state, use_epsilon=train)

            if train:
                self.update_replay_memory(state, action_index, reward, next_state, done)
                self.replay()

            state = next_state
            action_index = next_action_index

            timesteps += 1
            rewards += reward

        self.epsilon.update()

        return rewards, timesteps

    def reset_simulation(self):
        self.replay_memory.reset_memory()
        self.model_handler.reset_weights()
        self.epsilon.reset()
