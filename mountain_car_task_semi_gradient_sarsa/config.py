"""
Mountain Car Environment Constants and Tile Coder Configuration.

Constants:
----------
ACTIONS : tuple
    The set of possible actions for the agent in the environment where:
    -1 represents full throttle reverse,
     0 represents no throttle,
     1 represents full throttle forward.

POSITION_BOUND : tuple
    The lower and upper bounds of the car's position on the track.

VELOCITY_BOUND : tuple
    The minimum and maximum velocities the car can attain.

INITIAL_POSITION_BOUND : tuple
    The range for initializing the car's position at the start of each episode.

INITIAL_VELOCITY : float
    The car's initial velocity at the start of each episode.

ACTION_EFFECT_SCALE : float
    The scaling factor that determines the effect of the car's throttle on its velocity.

GRAVITY_SCALE : float
    The scaling factor that modulates the effect of gravity on the car, influencing
    how its position on the hill affects its acceleration.

NUM_TILINGS : int
    The number of overlapping tilings used in the tile coder for state representation.

TILES_PER_DIMENSION : tuple
    The number of tiles in each dimension (position, velocity) for the tile coder.

LOWER_BOUNDS : tuple
    The lower bounds for the tile coder, derived from POSITION_BOUND and VELOCITY_BOUND.

UPPER_BOUNDS : tuple
    The upper bounds for the tile coder, derived from POSITION_BOUND and VELOCITY_BOUND.

Examples:
---------
These constants can be used to initialize the Mountain Car environment and to configure the
tile coder for an agent learning to solve the task. By adjusting these values, one can control
the granularity of state representation and the dynamics of the car's movement within the environment.

"""
ACTIONS = (-1, 0, 1)

POSITION_BOUND = (-1.2, 0.5)
VELOCITY_BOUND = (-0.07, 0.07)
INITIAL_POSITION_BOUND = (-0.6, -0.4)
INITIAL_VELOCITY = 0

ACTION_EFFECT_SCALE = 0.001  # Scaling factor for the action's impact on velocity
GRAVITY_SCALE = 0.0025       # Modulates the effect of gravity based on the car's position

NUM_TILINGS = 8
TILES_PER_DIMENSION = 8, 8
LOWER_BOUNDS = POSITION_BOUND[0], VELOCITY_BOUND[0]
UPPER_BOUNDS = POSITION_BOUND[1], VELOCITY_BOUND[1]