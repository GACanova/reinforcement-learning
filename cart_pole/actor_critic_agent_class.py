"""
Provides an implementation of the Actor-Critic method for reinforcement learning.
"""

import numpy as np

class ActorCriticAgent:
    """
    An implementation of the Actor-Critic method using separate model handlers
    for the actor and the critic components, with experience replay.

    This agent learns policies based on the Actor-Critic framework where the
    critic evaluates the action taken by the actor by estimating the value function.

    Attributes:
        env (object): The environment where the agent interacts.
        critic_model_handler (object): Manages the critic's learning, predicting state values.
        actor_model_handler (object): Manages the actor's learning, determining actions.
        replay_memory (object): Stores experience tuples for later replay.
        actions (tuple, optional): A tuple of possible actions the agent can take.
                                   If empty, actions are used as received, which can be 
                                   continuous values or other forms.
        gamma (float): Discount factor for future rewards.
        batch_size (int): Number of experiences to sample from memory during training.
        episode_index (int): Counter for the number of episodes handled.
        history (list): Stores history of episodes for analysis.

    Methods:
        update_replay_memory(state, action_index, reward, next_state, done):
            Stores the transition in the memory.
        get_state_values(state):
            Computes the value for a given state using the critic.
        get_action(state):
            Determines the best action for a given state using the actor.
        replay():
            Performs training on a sampled batch from the replay memory.
        step(train=True):
            Executes a step or episode in the environment, optionally training the agent.
        reset_simulation():
            Resets the simulation environment and model weights.
    """
   
    
    def __init__(
        self,
        env,
        critic_model_handler,
        actor_model_handler,
        replay_memory,
        actions=(),
        gamma=1.0,
        batch_size=128,
    ):
        self.env = env
        self.critic_model_handler = critic_model_handler
        self.actor_model_handler = actor_model_handler
        self.replay_memory = replay_memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.episode_index = 0
        self.history = []

        if actions:
            self.map_action = lambda action: actions[action]
        else:
            self.map_action = lambda action: action

    def update_replay_memory(self, state, action_index, reward, next_state, done):
        self.replay_memory.store((state, action_index, reward, next_state, done))

    def get_state_values(self, state):
        return self.critic_model_handler.predict(state)

    def get_action(self, state):
        state = np.array([state])
        action = self.actor_model_handler.predict(state).numpy().squeeze()

        return action

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)

        states, actions, rewards, next_states, done = (
            np.array([transition[i] for transition in transitions]).reshape(
                self.batch_size, -1
            )
            for i in range(len(transitions[0]))
        )

        targets = rewards + self.gamma * self.get_state_values(next_states) * (1 - done)

        advantage = targets - self.get_state_values(states)

        self.actor_model_handler.train(states, actions, advantage)
        self.critic_model_handler.train(states, targets)

    def step(self, train=True):
        timesteps = 0
        rewards = 0
        state = self.env.reset()
        action = self.get_action(state)

        done = False

        while not done:
            next_state, reward, done = self.env.step(self.map_action(action))
            next_action = self.get_action(next_state)

            if train:
                self.update_replay_memory(state, action, reward, next_state, done)
                self.replay()

            state = next_state
            action = next_action

            timesteps += 1
            rewards += reward

        return rewards, timesteps

    def reset_simulation(self):
        self.replay_memory.reset_memory()
        self.actor_model_handler.reset_weights()
        self.critic_model_handler.reset_weights()
