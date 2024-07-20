"""
This module defines constants and parameters for a simulated cart-pole environment
and configurations for neural network models used in reinforcement learning algorithms.

The module specifies initial conditions, bounds, and dynamics properties for the
simulation of a cart-pole system, such as angles, angular velocities, positions,
and linear velocities. It also sets up the action space and other simulation parameters
like timesteps and physical properties of the cart and pole.

Additionally, parameters for tile coding feature representation and artificial neural
network (ANN) configurations are provided to support learning algorithms.

Constants:
    - Initial conditions and bounds for the theta (pole angle), theta dot (angular velocity),
      x (cart position), and x dot (cart velocity).
    - Action-related constants including available actions, action timestep duration,
      and force magnitude applied to the cart.
    - Physical properties of the system such as masses, lengths, and friction coefficients.
    - Parameters for tile coding such as the number of tilings and the tiles per dimension.
    - ANN parameters for building neural network models including input dimensions,
      output dimensions, hidden units, and activation functions.
"""


import numpy as np
from torch import nn

################ ENVIRONMENT #####################

THETA_0 = 0
THETA_DOT_0 = 0
THETA_BOUNDS = (-12.0 * np.pi / 180, 12.0 * np.pi / 180)
THETA_DOT_BOUNDS = (-4.0 * np.pi, 4.0 * np.pi)

X_0 = 0
X_DOT_0 = 0
X_BOUNDS = (-2.4, 2.4)
X_DOT_BOUNDS = (-5, 5)

ACTIONS = (-1, 1)
N_ACTIONS = len(ACTIONS)
ACTION_TIMESTEP = 0.02

INTEGRATION_TIMESTEP = 0.01
MAX_TIMESTEPS = 1000
CART_MASS = 1
POLE_MASS = 0.1
HALF_POLE_LENGTH = 0.5
CART_FRICTION_COEFFICIENT = 0.0005
POLE_FRICTION_COEFFICIENT = 0.000002
FORCE_MAGNITUDE = 10.0 * 0.02  # To be consistent with gym library and actions (-1, 1)
G = 9.8

################ TILE CODING #################

NUM_TILINGS = 10
TILES_PER_DIMENSION = 24, 24, 24, 24
LOWER_BOUNDS = THETA_BOUNDS[0], THETA_DOT_BOUNDS[0], X_BOUNDS[0], X_DOT_BOUNDS[0]
UPPER_BOUNDS = THETA_BOUNDS[1], THETA_DOT_BOUNDS[1], X_BOUNDS[1], THETA_DOT_BOUNDS[1]

################ ANN #################

INPUT_DIM = 4
OUTPUT_DIM = 2
HIDDEN_UNITS = (64, 64)
ACTIVATION = nn.ReLU()
