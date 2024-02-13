"""
Implements the CliffWalking environment for reinforcement learning experiments.
"""

from point_class import Point
from config import INITIAL_STATE, GOAL_STATE, CLIFF_CELLS


class CliffWalking:
    """
    Defines the CliffWalking environment for reinforcement learning simulations,
    modeling a grid where an agent must navigate from an initial state to a goal
    state while avoiding cliffs. This environment is typically used to demonstrate
    the differences in learning strategies, particularly in handling penalties for
    falling off the cliff.

    Attributes:
        initial_state (Point): The starting point for the agent.
        goal_state (Point): The target point the agent aims to reach.
        cliff_cells (list of tuples): Coordinates marking the cliff locations.
        state (Point): The current position of the agent in the environment.

    Methods:
        reset(): Resets the environment to the initial state for a new episode.
        step(action): Moves the agent based on the action taken, returns the new state,
                      a reward signal, and a done flag indicating if the goal is reached
                      or the agent fell off the cliff.
    """

    def __init__(
        self, initial_state=INITIAL_STATE, goal_state=GOAL_STATE, cliff_cells=None
    ):
        if not cliff_cells:
            self.cliff_cells = CLIFF_CELLS

        self.initial_state = Point.from_tuple(initial_state)
        self.goal_state = Point.from_tuple(goal_state)
        self.state = self.initial_state

    def reset(self):
        self.state = self.initial_state
        return self.state.to_tuple()

    def step(self, action):
        self.state += Point.from_tuple(action)
        reward = -1

        done = False

        if self.state == self.goal_state:
            reward = 0
            done = True

        elif self.state.to_tuple() in self.cliff_cells:
            reward = -100
            self.state = self.initial_state

        return self.state.to_tuple(), reward, done
