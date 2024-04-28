"""
Module for Simulating and Visualizing the Performance of a Semi-Gradient SARSA Agent.

This module contains functions to calculate and visualize the optimal value function and policy from
a given state-action value function within a reinforcement learning environment. It also provides tools 
for running multiple simulations of an agent's performance across episodes and iterations, and for plotting 
the average number of steps taken per episode.

Functions:
    get_optimal_value_function_and_policy:
        Determines the optimal value and policy from the state-action function.
    plot_state_value_function:
        Plots the state value function as a 3D surface plot.
    steps_per_episode:
        Computes the average number of steps taken per episode over multiple simulations.
    plot_steps_per_episode:
        Plots the average steps per episode over a series of episodes.

Dependencies:
    numpy:
        Utilized for numerical operations and handling of multi-dimensional arrays.
    pandas:
        Used for data manipulation and aggregation when processing simulation results.
    matplotlib:
        Provides plotting functions for visualizing data.
    config:
        Contains configuration parameters like the list of possible actions (ACTIONS).
Note:
    This module is part of a reinforcement learning package and assumes the existence of an environment
    conforming to a specific interface, including methods like `generate_state_grid()` and `run_simulation()`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import ACTIONS


def get_optimal_value_function_and_policy(env, state_action_function, actions=ACTIONS):
    """
    Calculates the optimal value function and policy from a given state-action value function.

    This function iterates through all possible states in the environment, as provided by
    `env.generate_state_grid()`. For each state, it evaluates the possible actions and determines
    the optimal action and its value based on the provided state-action value function. The result
    is a policy dictating the best action to take in each state and a value function representing
    the expected return from following this policy.

    Parameters:
        env (Environment): The environment object which should provide a method `generate_state_grid()`
                           that returns a grid of all possible states in the environment.
        state_action_function (dict): A dictionary mapping (state, action) pairs to estimated values.
                                      This function assumes that state_action_function has been
                                      populated by a learning algorithm beforehand.
        actions (iterable, optional): A list or tuple of possible actions. Defaults to ACTIONS.

    Returns:
        tuple: A tuple containing two elements:
               1. state_value_function (numpy.ndarray): An array where each element represents
                  the maximum value of the state as determined by the optimal policy, reshaped
                  into a grid with dimensions corresponding to the state dimensions.
               2. policy (list): A list of optimal actions for each state. Each entry is either an
                  index corresponding to the best action in `actions` or None if all actions have
                  the same value, indicating no preference.

    Notes:
        - The state_value_function array includes both the coordinates of each state and their respective
          values in a structured format.
        - If all actions have the same value in a state (i.e., the policy is indifferent), the corresponding
          policy index for that state is set to None to indicate this indeterminacy.
    """

    state_value_function = []
    policy = []

    states = env.generate_state_grid()

    for state in states:
        action_values = []

        for action in actions:
            action_values.append(state_action_function.get((state, action), 0))

        if all(x == action_values[0] for x in action_values):
            policy.append(None)
        else:
            policy.append(np.argmax(action_values))

        state_value_function.append(np.max(action_values))

    dim = len(states[0])
    resolution = round(np.power(len(states), 1.0 / dim))
    states = np.array(states).reshape(resolution, resolution, dim)
    state_value_function = np.array(state_value_function).reshape(
        resolution, resolution
    )
    state_value_function = np.dstack((states, state_value_function[:, :, None]))

    return state_value_function, policy


def plot_state_value_function(state_value_function):
    """
    Plots the state value function as a 3D surface plot.

    This function takes a state value function array where each element contains
    the coordinates of the state in the first two dimensions and the value of the state
    in the third dimension. It visualizes the state value function using a 3D surface plot
    to help in analyzing the spatial distribution of values across states.

    Parameters:
        state_value_function (numpy.ndarray): A 3D numpy array where the first two dimensions
                                              are the coordinates of each state and the third
                                              dimension is the value of the state. This array
                                              is expected to be the output from the function
                                              `get_optimal_value_function_and_policy`.

    Notes:
        - The function uses a negative sign for the state values to plot the surface, which may
          be adjusted based on how you prefer to visualize high and low values (e.g., peaks vs pits).
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.view_init(elev=45, azim=135)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")

    ax.set_xlabel("Position", fontsize=16, labelpad=20)
    ax.set_ylabel("Velocity", fontsize=16, labelpad=20)
    ax.set_zlabel("Value", fontsize=16, labelpad=20)

    ax.set_xlim(0.6, -1.2)
    ax.set_ylim(0.07, -0.07)
    ax.set_zlim(
        -state_value_function[:, :, 2].max(), -state_value_function[:, :, 2].min()
    )

    ax.set_xticks([-1.2, 0.6])
    ax.set_yticks([0.07, -0.07])
    ax.set_zticks([0, 120])

    ax.plot_surface(
        state_value_function[:, :, 0],
        state_value_function[:, :, 1],
        -state_value_function[:, :, 2],
        cmap="plasma",
        alpha=0.9,
    )
    plt.show()


def steps_per_episode(agent, iterations=100, n_episodes=500):
    """
    Calculates the average number of steps per episode over a specified number of episodes and iterations for a given agent.
    This function runs multiple simulations to gather data on how many steps the agent takes to complete each episode,
    averaging the results to provide insights into the agent's efficiency and learning progress over time.

    Parameters:
        agent: The agent object that implements the `run_simulation` method, which simulates learning over a series of episodes.
        iterations (int): The number of times the simulation is repeated to ensure statistical significance of the results.
        n_episodes (int): The number of episodes to simulate in each iteration to evaluate the agent's performance.

    Returns:
        pandas.DataFrame: A DataFrame containing the average number of steps per episode for each episode number across all iterations.
                          It has two columns: 'episode' and 'avg_timesteps', where 'episode' is the episode number (1 to n_episodes)
                          and 'avg_timesteps' is the average number of steps taken to complete that episode across all iterations.

    Note:
        - Each iteration involves resetting the agent's state to ensure independent evaluation of performance across episodes.
        - This function assumes the `run_simulation` method of the agent returns a history list with information about each step
          taken in all episodes, including the episode number and timestep.
    """

    results = []

    for i in range(iterations):
        _, history = agent.run_simulation(n_episodes=n_episodes, reset=True)
        results.extend([dict(x, iteration=i) for x in history])

    df = (
        pd.DataFrame(results)
        .groupby(["iteration", "episode"], as_index=False)
        .agg(timesteps=("timestep", "count"))
        .groupby("episode", as_index=False)
        .agg(avg_timesteps=("timesteps", "mean"))
    )

    return df


def plot_steps_per_episode(results):
    """
    Plots the number of steps per episode over a series of episodes from simulation results.

    Parameters:
    - results (list of dicts): A list where each dict contains 'data' and 'param' keys.
      'data' should have 'x' (episode numbers) and 'y' (steps per episode) lists,
      and 'param' should contain matplotlib line plot parameters for customization.

    """

    plt.xlim([0, 500])
    plt.xlabel("Episodes", fontsize=16)

    plt.yticks(fontsize=12)
    plt.ylabel("Steps per episode", fontsize=16)
    plt.yscale("log")
    plt.ylim(90, 1000)

    ax = plt.gca()

    formatter = FuncFormatter(lambda value, tick_number: f"{int(value)}")
    ax.yaxis.set_major_formatter(formatter)

    tick_locations = [100, 200, 400, 1000]
    plt.gca().set_yticks(tick_locations)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for result in results:
        plt.plot(result["data"]["x"], result["data"]["y"], **result["param"])

    plt.legend(fontsize=16)
    plt.show()
