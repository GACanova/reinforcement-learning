"""
Provides a utility class for managing the decay of the epsilon parameter in reinforcement learning.
"""
import numpy as np
import matplotlib.pyplot as plt


class EpsilonDecay:
    """
    Manages the decay of epsilon in a reinforcement learning context, allowing
    for either linear or exponential decay.

    The class supports plotting the decay process, making it useful for
    visualization and analysis of the decay behavior over time.

    Attributes:
        epsilon_start (float): The initial value of epsilon.
        epsilon_end (float): The final value epsilon will approach.
        decay_method (str): The method of decay ('linear' or 'exponential').
        last_episode (int): The episode number at which decay adjustments stop.
        decay_rate (float): The decay rate; significant in exponential decay.

    Methods:
        __call__():
            Returns the current value of epsilon.

        reset():
            Resets epsilon to its starting value and resets the step count.

        update():
            Updates the epsilon value based on the decay method and increments
            the step count.

        plot_decay():
            Generates a plot showing how epsilon decays over time according
            to the specified decay method until it reaches epsilon_end or the
            step count surpasses last_episode.

        __str__():
            Provides a string representation of the current epsilon value.
    """

    def __init__(
        self,
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay_method="linear",
        last_episode=100,
        decay_rate=0.01,
    ):
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.decay_method = decay_method
        self.last_episode = last_episode
        self.step_count = 0

    def __call__(self):
        return self.epsilon

    def reset(self):
        self.epsilon = self.epsilon_start
        self.step_count = 0

    def update(self):
        if self.decay_method == "linear":
            self.epsilon = self.epsilon_start - (
                self.step_count / self.last_episode
            ) * (self.epsilon_start - self.epsilon_end)

        elif self.decay_method == "exponential":
            self.epsilon = self.epsilon_end + (
                self.epsilon_start - self.epsilon_end
            ) * np.exp(-self.decay_rate * self.step_count)

        self.epsilon = max(self.epsilon, self.epsilon_end)
        self.step_count += 1

    def plot_decay(self):
        epsilons = []
        self.reset()

        while self.epsilon > self.epsilon_end or self.step_count <= self.last_episode:
            epsilons.append(self.epsilon)
            self.update()

        plt.figure(figsize=(10, 5))
        plt.plot(epsilons, label=f"{self.decay_method.capitalize()} Decay")
        plt.title("Epsilon Decay Over Steps")
        plt.xlabel("Steps")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.grid(True)
        plt.show()

        self.reset()

    def __str__(self):
        return f"Epsilon: {self.epsilon}"
