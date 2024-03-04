"""
This module defines the configurations for environments used in Examples 8.1 (Dyna Maze), 8.2 (Blocking Maze),
and 8.3 (Shortcut Maze) from Barto's reinforcement learning book. It details grid sizes, initial and goal states,
blocked cells, actions, and specific barriers for different maze configurations. These settings facilitate the simulation
of various maze navigation tasks, showcasing the application and effectiveness of reinforcement learning algorithms
like Dyna-Q and Dyna-Q+ in both static and dynamically changing environments.

Attributes:
    X_SIZE (int): Width of the maze grid.
    Y_SIZE (int): Height of the maze grid.
    STATES (list of tuples): All possible (x, y) positions in the maze.
    BLOCKED_CELLS (tuple of tuples): Positions of obstacles in the maze.
    INITIAL_STATE (tuple): Starting position in the Dyna Maze.
    GOAL_STATE (tuple): Target position in the Dyna Maze.
    ACTIONS (tuple of tuples): Possible actions as directional movements.
    BARRIER_CELLS_1 (tuple of tuples): Obstacles for the first setup in the Blocking Maze.
    BARRIER_CELLS_2 (tuple of tuples): Adjusted obstacles for the second setup in the Blocking Maze, demonstrating environment dynamics.
    INITIAL_STATE_BLOCKING_MAZE (tuple): Starting position for the Blocking Maze.
    GOAL_STATE_BLOCKING_MAZE (tuple): Target position for the Blocking Maze.
    BARRIER_CELLS_3 (tuple of tuples): Obstacles for the Shortcut Maze, illustrating a scenario with an emergent shortcut.
    INITIAL_STATE_SHORTCUT_MAZE (tuple): Starting position for the Shortcut Maze.
    GOAL_STATE_SHORTCUT_MAZE (tuple): Target position for the Shortcut Maze.
"""

X_SIZE = 9
Y_SIZE = 6
assert X_SIZE > 0 and Y_SIZE > 0, "Grid sizes must be positive integers."

STATES = [(x, y) for x in range(X_SIZE) for y in range(Y_SIZE)]

BLOCKED_CELLS = (
    (2, 2),
    (2, 3),
    (2, 4),
    (5, 1),
    (7, 3),
    (7, 4),
    (7, 5),
)

INITIAL_STATE = (0, 3)
GOAL_STATE = (8, 5)
assert (
    0 <= INITIAL_STATE[0] < X_SIZE and 0 <= INITIAL_STATE[1] < Y_SIZE
), "Initial state must be bounded."
assert (
    0 <= GOAL_STATE[0] < X_SIZE and 0 <= GOAL_STATE[1] < Y_SIZE
), "Goal state must be bounded."
assert INITIAL_STATE != GOAL_STATE, "Initial and goal states must be different."

assert (
    INITIAL_STATE not in BLOCKED_CELLS and GOAL_STATE not in BLOCKED_CELLS
), "Invalid initial or goal states."

ACTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))

###############################################################################
# Example 8.3 Blocking Maze

BARRIER_CELLS_1 = (
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
)

BARRIER_CELLS_2 = (
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
    (8, 2),
)

INITIAL_STATE_BLOCKING_MAZE = (3, 0)
GOAL_STATE_BLOCKING_MAZE = (8, 5)

assert (
    0 <= INITIAL_STATE_BLOCKING_MAZE[0] < X_SIZE
    and 0 <= INITIAL_STATE_BLOCKING_MAZE[1] < Y_SIZE
), "Initial state must be bounded."
assert (
    0 <= GOAL_STATE_BLOCKING_MAZE[0] < X_SIZE
    and 0 <= GOAL_STATE_BLOCKING_MAZE[1] < Y_SIZE
), "Goal state must be bounded."
assert (
    INITIAL_STATE_BLOCKING_MAZE != GOAL_STATE_BLOCKING_MAZE
), "Initial and goal states must be different."

assert (
    INITIAL_STATE_BLOCKING_MAZE not in BARRIER_CELLS_1
    and GOAL_STATE_BLOCKING_MAZE not in BARRIER_CELLS_1
), "Invalid initial or goal states."

assert (
    INITIAL_STATE_BLOCKING_MAZE not in BARRIER_CELLS_2
    and GOAL_STATE_BLOCKING_MAZE not in BARRIER_CELLS_2
), "Invalid initial or goal states."

#############################################################################################################
# Example 8.3 Shortcut Maze

BARRIER_CELLS_3 = (
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
)

INITIAL_STATE_SHORTCUT_MAZE = (3, 0)
GOAL_STATE_SHORTCUT_MAZE = (8, 5)

assert (
    0 <= INITIAL_STATE_SHORTCUT_MAZE[0] < X_SIZE
    and 0 <= INITIAL_STATE_SHORTCUT_MAZE[1] < Y_SIZE
), "Initial state must be bounded."
assert (
    0 <= GOAL_STATE_SHORTCUT_MAZE[0] < X_SIZE
    and 0 <= GOAL_STATE_SHORTCUT_MAZE[1] < Y_SIZE
), "Goal state must be bounded."
assert (
    INITIAL_STATE_SHORTCUT_MAZE != GOAL_STATE_SHORTCUT_MAZE
), "Initial and goal states must be different."

assert (
    INITIAL_STATE_SHORTCUT_MAZE not in BARRIER_CELLS_3
    and GOAL_STATE_SHORTCUT_MAZE not in BARRIER_CELLS_3
), "Invalid initial or goal states."
