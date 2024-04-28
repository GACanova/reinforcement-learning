"""
Mountain Car Environment for Reinforcement Learning.
"""

import numpy as np

from config import (
    INITIAL_POSITION_BOUND,
    INITIAL_VELOCITY,
    POSITION_BOUND,
    VELOCITY_BOUND,
    ACTION_EFFECT_SCALE,
    GRAVITY_SCALE,
)


class MountainCar:
    """
    A Mountain Car environment class for reinforcement learning.

    This class simulates the classic Mountain Car problem. An underpowered car must drive up a steep hill.
    The goal is to reach the top of the hill. The car's engine is not strong enough to climb the
    hill in a direct ascent, so it must build up momentum by driving back and forth.

    Attributes:
        initial_position_bound (tuple): Bounds for the car's initial position.
        initial_velocity (float): The car's starting velocity at the beginning of an episode.
        position_bound (tuple): The minimum and maximum positions of the car on the track.
        velocity_bound (tuple): The allowed range of velocities for the car.
        action_effect_scale (float): The effect of actions on the car's velocity.
        gravity_scale (float): The effect of gravity, modulating the car's acceleration based on its position.
        grid_resolution (int): The resolution of the grid for generating a discrete representation of the state space.
        x (float): The current position of the car.
        v (float): The current velocity of the car.

    Methods:
        generate_initial_position(): Returns a random initial position for the car within the initial position bounds.
        reset(): Resets the environment state to a new initial condition.
        step(action): Updates the environment state based on the provided action.
        generate_state_grid(): Generates a grid of possible states for the environment.
    """

    def __init__(
        self,
        initial_position_bound=INITIAL_POSITION_BOUND,
        initial_velocity=INITIAL_VELOCITY,
        position_bound=POSITION_BOUND,
        velocity_bound=VELOCITY_BOUND,
        action_effect_scale=ACTION_EFFECT_SCALE,
        gravity_scale=GRAVITY_SCALE,
        grid_resolution=100,
    ):
        self.initial_position_bound = initial_position_bound
        self.initial_velocity = initial_velocity
        self.position_bound = position_bound
        self.velocity_bound = velocity_bound
        self.action_effect_scale = action_effect_scale
        self.gravity_scale = gravity_scale
        self.grid_resolution = grid_resolution
        self.x = None
        self.v = None

    def generate_initial_position(self):
        return np.random.uniform(
            self.initial_position_bound[0], self.initial_position_bound[1]
        )

    def reset(self):
        self.x = self.generate_initial_position()
        self.v = self.initial_velocity

        return self.x, self.v

    def step(self, action):
        self.v += self.action_effect_scale * action - self.gravity_scale * np.cos(
            3 * self.x
        )
        self.v = np.clip(self.v, self.velocity_bound[0], self.velocity_bound[1])
        self.x = np.clip(self.x, self.position_bound[0], self.position_bound[1])
        self.x += self.v

        if self.x <= self.position_bound[0]:
            self.v = 0

        reward = -1
        done = False

        if self.x >= self.position_bound[1]:
            done = True

        return (self.x, self.v), reward, done

    def generate_state_grid(self):
        axes = [
            np.linspace(low, high, self.grid_resolution, endpoint=True)
            for low, high in zip(
                (self.position_bound[0], self.velocity_bound[0]),
                (self.position_bound[1], self.velocity_bound[1]),
            )
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        points = np.stack(grid, axis=-1).reshape(-1, len(axes))
        states = list(map(tuple, points))

        return states
