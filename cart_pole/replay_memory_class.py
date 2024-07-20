"""This module defines the ReplayMemory class for managing experience replay in reinforcement learning."""
import random


class ReplayMemory:
    """
    A class to store experiences for reinforcement learning agents, which can be later
    used for training purposes. This is typically known as experience replay, where the
    memory stores a limited number of transitions (state, action, reward, next state, done).

    Attributes:
        memory_size (int): Maximum number of transitions to store in the memory.
        memory (list): Container for storing transitions.
        index (int): Current position in the memory to store a new transition.

    Parameters:
        memory_size (int): The size of the memory. Defaults to 10000.
    """

    def __init__(self, memory_size=10000):
        self.memory_size = memory_size
        self.memory = []
        self.index = 0

    def store(self, transition_variables):
        """
        Store a transition in the replay memory. If the memory is full, it will overwrite
        the oldest transition in a cyclic manner.

        Parameters:
            transition_variables (tuple): A transition tuple (state, action, reward, next state, done).
        """

        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.index] = transition_variables
        self.index = (self.index + 1) % self.memory_size

    def sample(self, batch_size=1):
        """
        Randomly samples a batch of transitions from the memory.

        Parameters:
            batch_size (int): The number of transitions to sample. Defaults to 1.

        Returns:
            list: A list of randomly sampled transitions.
        """

        if len(self.memory) < batch_size:
            raise ValueError(
                "Not enough elements in the memory to sample the requested batch size."
            )
        return random.sample(self.memory, batch_size)

    def reset_memory(self):
        """
        Clears the memory and resets the index.
        """

        self.memory = []
        self.index = 0

    def __len__(self):
        """
        Returns the current size of the memory.

        Returns:
            int: The number of items in the memory.
        """

        return len(self.memory)
