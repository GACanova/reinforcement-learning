"""
Configures the gridworld for the cliff-walking task, illustrating
differences between Sarsa, Q-learning, and Expected Sarsa methods. 
The environment is a 2D grid where an agent moves to avoid cliffs 
and reach a goal.

Attributes:
    X_SIZE (int): Grid width (columns). Positive integer.
    Y_SIZE (int): Grid height (rows). Positive integer.
    STATES (list): All grid states as (x, y) pairs.
    CLIFF_CELLS (list): Cliff locations as (x, y) pairs. Stepping here
        results in a penalty.
    INITIAL_STATE (tuple): Starting point of the agent (x, y).
    GOAL_STATE (tuple): Target point for the agent (x, y).
    ACTIONS (list): Actions agent can take, represented as (x, y) moves.

Notes:
    - Grid indexed from (0,0) at the top-left corner.
    - Agent aims to go from INITIAL_STATE to GOAL_STATE without
      falling off the cliff.
    - Assertions validate the integrity of the configuration, ensuring
      grid size is positive, states are within bounds, and initial/goal
      states are valid and not on cliffs.

This setup supports experimentation with on-policy (Sarsa), off-policy
(Q-learning), and Expected Sarsa learning algorithms, providing a
foundation to compare their performance in a controlled environment.
"""

X_SIZE = 12
Y_SIZE = 4
assert X_SIZE > 0 and Y_SIZE > 0, "Grid sizes must be positive integers."

STATES = [(x, y) for x in range(X_SIZE) for y in range(Y_SIZE)]

CLIFF_CELLS = [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
]

assert len(CLIFF_CELLS) <= X_SIZE

INITIAL_STATE = (0, 0)
GOAL_STATE = (11, 0)
assert (
    0 <= INITIAL_STATE[0] < X_SIZE and 0 <= INITIAL_STATE[1] < Y_SIZE
), "Initial state must be bounded."
assert (
    0 <= GOAL_STATE[0] < X_SIZE and 0 <= GOAL_STATE[1] < Y_SIZE
), "Goal state must be bounded."
assert INITIAL_STATE != GOAL_STATE, "Initial and goal states must be different."

assert (
    INITIAL_STATE not in CLIFF_CELLS and GOAL_STATE not in CLIFF_CELLS
), "Invalid initial or goal states."

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
