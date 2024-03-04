"""
This module implements the Maze environment for reinforcement learning,
based on Example 8.1 from Barto's RL book. It defines a simple grid-based
maze where an agent must navigate from an initial state to a goal state,
avoiding blocked cells.

Dependencies:
- Point: A class from `point_class` module used to represent the positions (states) in the maze.
- INITIAL_STATE, GOAL_STATE, BLOCKED_CELLS: Configurations imported from `config` module specifying
the start point, end point, and obstacles within the maze, respectively.
"""

from point_class import Point
from config import INITIAL_STATE, GOAL_STATE, BLOCKED_CELLS


class Maze:
    """
    Represents a Maze environment for reinforcement learning tasks,
    as described in Example 8.1 of Barto's RL book.

    The Maze environment is a grid where an agent moves to reach a goal state while avoiding blocked cells.
    Actions move the agent in the grid, and reaching the goal state provides a reward.

    Attributes:
        initial_state (Point): The starting point of the agent in the maze, represented as a Point object.
        goal_state (Point): The target point the agent aims to reach, represented as a Point object.
        blocked_cells (list): A list of tuples representing the grid cells that are blocked and cannot be entered.
        state (Point): The current state of the agent in the maze, represented as a Point object.

    Methods:
        reset(self):
            Resets the environment to the initial state and returns it. This is typically used at the beginning
            of a new episode.

        step(self, action):
            Executes the given action and updates the environment's state. An action moves the agent in the maze.
            If an action leads to a blocked cell, the move is undone, and the state does not change. Reaching the
            goal state provides a reward and marks the episode as done.

            Parameters:
                action (tuple): A tuple representing the movement direction (e.g., (0, 1) for moving right).

            Returns:
                tuple: The new state as a tuple after taking the action.
                int: The reward for taking the action. Reaching the goal state gives a reward of 1, otherwise 0.
                bool: A flag indicating whether the goal state has been reached and the episode is done.
    """

    def __init__(
        self,
        initial_state=INITIAL_STATE,
        goal_state=GOAL_STATE,
        blocked_cells=BLOCKED_CELLS,
    ):
        self.initial_state = Point.from_tuple(initial_state)
        self.goal_state = Point.from_tuple(goal_state)
        self.blocked_cells = blocked_cells
        self.state = self.initial_state

    def reset(self):
        self.state = self.initial_state
        return self.state.to_tuple()

    def step(self, action):
        self.state += Point.from_tuple(action)
        reward = 0

        if self.state.to_tuple() in self.blocked_cells:
            self.state -= Point.from_tuple(action)

        done = False

        if self.state == self.goal_state:
            reward = 1
            done = True

        return self.state.to_tuple(), reward, done
