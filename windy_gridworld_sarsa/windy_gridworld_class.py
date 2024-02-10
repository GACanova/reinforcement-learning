"""
This module defines the WindyGridworld environment, a grid-based simulation where an
agent must navigate from a starting point to a goal location against the backdrop of
varying wind strengths affecting movement in certain columns.
"""

from point_class import Point
from config import INITIAL_STATE, GOAL_STATE, WIND_DISTRIBUTION


class WindyGridworld:
    """
    Implements the Windy Gridworld environment. The environment simulates a grid
    with a start and a goal state, where certain columns have an upward wind that affects
    the agent's movement.

    Attributes:
        initial_state (Point): The starting state of the environment.
        goal_state (Point): The goal state of the environment.
        state (Point): The current state of the agent in the environment.
        wind_distribution (list[Point]): The wind effect in each column of the grid.

    Methods:
        reset(): Resets the environment to the initial state.
        step(action): Performs an action in the environment, moving the agent and applying
                      wind effects. Returns the new state, a reward, and a flag indicating
                      if the goal state is reached.
    """

    def __init__(
        self,
        initial_state=INITIAL_STATE,
        goal_state=GOAL_STATE,
        wind_distribution=None,
    ):
        if wind_distribution is None:
            wind_distribution = WIND_DISTRIBUTION
        self.initial_state = Point.from_tuple(initial_state)
        self.goal_state = Point.from_tuple(goal_state)
        self.state = self.initial_state
        self.wind_distribution = [Point(0, y) for y in wind_distribution]

    def reset(self):
        self.state = self.initial_state
        return self.state.to_tuple()

    def step(self, action):
        self.state += self.wind_distribution[self.state.x]
        self.state += Point.from_tuple(action)
        reward = -1

        done = False

        if self.state == self.goal_state:
            reward = 0
            done = True

        return self.state.to_tuple(), reward, done
