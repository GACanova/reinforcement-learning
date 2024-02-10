X_SIZE = 10
Y_SIZE = 7
assert X_SIZE > 0 and Y_SIZE > 0, "Grid sizes must be positive integers."

STATES = [(x, y) for x in range(X_SIZE) for y in range(Y_SIZE)]

INITIAL_STATE = (0, 3)
GOAL_STATE = (7, 3)
assert (
    0 <= INITIAL_STATE[0] < X_SIZE and 0 <= INITIAL_STATE[1] < Y_SIZE
), "Initial state must be bounded."
assert (
    0 <= GOAL_STATE[0] < X_SIZE and 0 <= GOAL_STATE[1] < Y_SIZE
), "Initial state must be bounded."
assert INITIAL_STATE != GOAL_STATE, "Initial and goal states must be different."

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
KINGS_ACTIONS = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
]  # up, down, right, left, up-right, up-left, down-rigt, down-left
KINGS_ACTIONS_STOP = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
]  # stop, up, down, right, left, up-right, up-left, down-rigt, down-left

WIND_DISTRIBUTION = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
assert len(WIND_DISTRIBUTION) == X_SIZE
