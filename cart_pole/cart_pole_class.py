"""
This module defines the CartPole class, which simulates the dynamics of a cart-pole system used in reinforcement learning and control system environments.
"""
import numpy as np
from config import (
    X_0,
    X_DOT_0,
    X_BOUNDS,
    THETA_0,
    THETA_DOT_0,
    THETA_BOUNDS,
    CART_MASS,
    CART_FRICTION_COEFFICIENT,
    POLE_MASS,
    POLE_FRICTION_COEFFICIENT,
    HALF_POLE_LENGTH,
    ACTION_TIMESTEP,
    INTEGRATION_TIMESTEP,
    MAX_TIMESTEPS,
    FORCE_MAGNITUDE,
    G,
)


class CartPole:
    """
    Simulation of a cart-pole balancing system, incorporating different integrators for motion simulation.

    This class models and simulates the dynamics of a cart-pole system, which consists of a pole attached by an un-actuated
    joint to a cart, which moves along a track. The system is controlled by applying a force to the cart,
    and the goal is to balance the pole vertically while keeping the cart within predefined bounds.

    Attributes:
        theta_0 (float): Initial angle of the pole with the vertical (in radians).
        theta_dot_0 (float): Initial angular velocity of the pole (in radians per second).
        theta_bounds (tuple): Lower and upper bounds for the pole angle (in radians).
        theta_dot_bounds (tuple, optional): Lower and upper bounds for the angular velocity of the pole (in radians per second).
        x_0 (float): Initial position of the cart on the track.
        x_dot_0 (float): Initial velocity of the cart along the track.
        x_bounds (tuple): Lower and upper bounds for the cart position.
        x_dot_bounds (tuple, optional): Lower and upper bounds for the velocity of the cart.
        cart_mass (float): Mass of the cart.
        mu_cart (float): Coefficient of friction for the cart.
        pole_mass (float): Mass of the pole.
        half_pole_length (float): Half the length of the pole (from pivot to tip).
        mu_pole (float): Coefficient of friction at the pole's pivot.
        force_magnitude (float): Magnitude of the force applied to the cart.
        g (float): Acceleration due to gravity.
        tau (float): Time step for action application.
        integrator (str): Type of integrator to use for simulation ('euler', 'semi_implicit_euler', 'verlet').
        dt (float): Integration time step.
        max_timesteps (int): Maximum number of timesteps for the simulation before automatic termination.

    Methods:
        calculate_force(f, theta_var, x_var): Calculates the forces and resulting accelerations for given states and action.
        euler_integration(f): Performs Euler integration for the system's dynamics.
        semi_implicit_euler_integration(f): Performs semi-implicit Euler integration.
        verlet_integration(f): Performs Verlet integration for more accurate simulation of the dynamics.
        integrate(action): Integrates the system's dynamics over one time step using the specified integrator.
        reset(lower_bound, upper_bound): Resets the system state to random initial conditions within specified bounds.
        step(action): Simulates one time step of the system, returning the new state, reward, and termination status.

    Raises:
        ValueError: If the action timestep (tau) is not an integer multiple of the integration timestep (dt) or is less than dt.
    """

    def __init__(
        self,
        theta_0=THETA_0,
        theta_dot_0=THETA_DOT_0,
        theta_bounds=THETA_BOUNDS,
        theta_dot_bounds=None,
        x_0=X_0,
        x_dot_0=X_DOT_0,
        x_bounds=X_BOUNDS,
        x_dot_bounds=None,
        cart_mass=CART_MASS,
        mu_cart=CART_FRICTION_COEFFICIENT,
        pole_mass=POLE_MASS,
        half_pole_length=HALF_POLE_LENGTH,
        mu_pole=POLE_FRICTION_COEFFICIENT,
        force_magnitude=FORCE_MAGNITUDE,
        g=G,
        tau=ACTION_TIMESTEP,
        integrator="verlet",
        dt=INTEGRATION_TIMESTEP,
        max_timesteps=MAX_TIMESTEPS,
    ):
        self.theta_0 = theta_0
        self.theta_dot_0 = theta_dot_0
        self.theta_bounds = theta_bounds
        self.theta_dot_bounds = theta_dot_bounds
        self.x_0 = x_0
        self.x_dot_0 = x_dot_0
        self.x_bounds = x_bounds
        self.x_dot_bounds = x_dot_bounds
        self.cart_mass = cart_mass
        self.mu_cart = mu_cart
        self.pole_mass = pole_mass
        self.mu_pole = mu_pole
        self.half_pole_length = half_pole_length
        self.force_magnitude = force_magnitude
        self.g = g
        self.integrator = integrator
        self.tau = tau
        self.dt = dt
        self.max_timesteps = max_timesteps

        if (
            abs(self.tau / self.dt - round(self.tau / self.dt)) > 1e-5
            or self.tau < self.dt
        ):
            raise ValueError(
                "The action timestep (tau) must be an integer multiple of the integration timestep (dt), and not less."
            )

        self.integration_steps = round(self.tau / self.dt)
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_moment = self.pole_mass * self.half_pole_length
        self.timestep_index = 0

        self.theta, self.theta_dot, self.theta_ddot = None, None, None
        self.x, self.x_dot, self.x_ddot = None, None, None
        self.state = None

        self.reset()

    def calculate_force(self, f, theta_var, x_var):
        theta, theta_dot, theta_ddot = theta_var
        x, x_dot, x_ddot = x_var

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        cart_normal = self.total_mass * self.g - self.pole_moment * (
            sin_theta * theta_ddot + cos_theta * theta_dot**2
        )

        sgn = np.sign(cart_normal * x_dot)

        tmp = (
            -f
            - self.pole_moment
            * (theta_dot**2)
            * (sin_theta + self.mu_cart * sgn * cos_theta)
        ) / self.total_mass

        theta_ddot_numerator = (
            self.g * sin_theta
            + cos_theta * (tmp + self.mu_cart * self.g * sgn)
            - self.mu_pole * theta_dot / self.pole_moment
        )

        theta_ddot_denominator = (
            4 * self.half_pole_length / 3
            - self.pole_moment
            * cos_theta
            * (cos_theta - self.mu_cart * sgn)
            / self.total_mass
        )

        theta_ddot = theta_ddot_numerator / theta_ddot_denominator

        x_ddot = (
            f
            + self.pole_moment * (sin_theta * theta_dot**2 - cos_theta * theta_ddot)
            - self.mu_cart * cart_normal * sgn
        ) / self.total_mass

        return theta_ddot, x_ddot

    def euler_integration(self, f):
        # Accelerations
        theta_var = self.theta, self.theta_dot, self.theta_ddot
        x_var = self.x, self.x_dot, self.x_ddot

        self.theta_ddot, self.x_ddot = self.calculate_force(f, theta_var, x_var)

        self.theta += self.theta_dot * self.dt
        self.theta_dot += self.theta_ddot * self.dt

        self.x += self.x_dot * self.dt
        self.x_dot += self.x_ddot * self.dt

    def semi_implicit_euler_integration(self, f):
        # Accelerations
        theta_var = self.theta, self.theta_dot, self.theta_ddot
        x_var = self.x, self.x_dot, self.x_ddot

        self.theta_ddot, self.x_ddot = self.calculate_force(f, theta_var, x_var)

        self.theta_dot += self.theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt

        self.x_dot += self.x_ddot * self.dt
        self.x += self.x_dot * self.dt

    def verlet_integration(self, f):
        # Accelerations
        theta_var = self.theta, self.theta_dot, self.theta_ddot
        x_var = self.x, self.x_dot, self.x_ddot

        theta_ddot, x_ddot = self.calculate_force(f, theta_var, x_var)

        # Next positions
        theta_next = (
            self.theta + self.theta_dot * self.dt + 0.5 * theta_ddot * self.dt**2
        )
        x_next = self.x + self.x_dot * self.dt + 0.5 * x_ddot * self.dt**2

        # Intermediate Velocities
        theta_dot_half = self.theta_dot + 0.5 * theta_ddot * self.dt
        x_dot_half = self.x_dot + 0.5 * x_ddot * self.dt

        # New accelerations
        theta_var_next = theta_next, theta_dot_half, self.theta_ddot
        x_var_next = x_next, x_dot_half, self.x_ddot

        theta_ddot_next, x_ddot_next = self.calculate_force(
            f, theta_var_next, x_var_next
        )

        # Next velocities
        theta_dot_next = theta_dot_half + 0.5 * theta_ddot_next * self.dt
        x_dot_next = x_dot_half + 0.5 * x_ddot_next * self.dt

        # Update variables
        self.theta, self.theta_dot, self.theta_ddot = (
            theta_next,
            theta_dot_next,
            theta_ddot_next,
        )
        self.x, self.x_dot, self.x_ddot = x_next, x_dot_next, x_ddot_next

    def integrate(self, action):
        for t in range(self.integration_steps):
            f = (
                (action * self.force_magnitude / self.dt) if t == 0 else 0
            )  # Remember that F = ΔP/Δt

            if self.integrator == "verlet":
                self.verlet_integration(f)
            elif self.integrator == "euler":
                self.euler_integration(f)
            elif self.integrator == "semi_implicit_euler":
                self.semi_implicit_euler_integration(f)

            else:
                raise ValueError(
                    f"Integrator name {self.integrator} not known. Options: 'euler', 'semi_implicit_euler', 'verlet'."
                )

    def reset(self, lower_bound=-0.05, upper_bound=0.05):
        self.theta, self.theta_dot, self.x, self.x_dot = np.random.uniform(
            lower_bound, upper_bound, size=4
        )
        self.theta_ddot, self.x_ddot = 0, 0
        self.state = self.theta, self.theta_dot, self.x, self.x_dot
        self.timestep_index = 0

        return self.state

    def step(self, action):
        self.timestep_index += 1
        self.integrate(action)

        if self.theta_dot_bounds is not None:
            assert (
                self.theta_dot_bounds[0] <= self.theta_dot <= self.theta_dot_bounds[1]
            ), "theta_dot is outside of bounds."

        if self.x_dot_bounds is not None:
            assert (
                self.x_dot_bounds[0] <= self.x_dot <= self.x_dot_bounds[1]
            ), "x_dot is outside of bounds."

        conditions = (
            self.theta_bounds[0] <= self.theta <= self.theta_bounds[1],
            self.x_bounds[0] <= self.x <= self.x_bounds[1],
        )

        if all(conditions):
            reward = 1
            done = self.timestep_index == self.max_timesteps
        else:
            reward = 0
            done = True

        self.state = self.theta, self.theta_dot, self.x, self.x_dot

        return self.state, reward, done
