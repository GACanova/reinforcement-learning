"""
This module provides a suite of functions and utilities for visualizing and analyzing the performance of reinforcement learning agents in maze-like environments.
It includes methods for computing optimal value functions and policies, plotting trajectories and policies, and evaluating agent performance over multiple episodes. The module supports environments with blocked cells and dynamic changes, incorporating the Dyna-Q and Dyna-Q+ algorithms' principles.

Key functionalities include:
- Computing the optimal value function and policy based on a state-action value function.
- Calculating the average number of steps per episode to gauge agent efficiency.
- Visualizing agent trajectories within the maze, highlighting the start, goal, and blocked cells.
- Plotting the optimal policy and state values on the grid.
- Finding and visualizing an optimal trajectory given an optimal policy.
- Analyzing the cumulative reward per timestep to understand agent performance dynamics, especially before and after environmental changes.

Dependencies:
- numpy for numerical operations and array handling.
- pandas for data manipulation and analysis.
- matplotlib for plotting and visualizing data.
- matplotlib.patches for drawing shapes on plots.

This module assumes the availability of global configurations such as maze dimensions (X_SIZE, Y_SIZE), goal state, action set, and blocked cells from the `config` module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import X_SIZE, Y_SIZE, GOAL_STATE, ACTIONS, BLOCKED_CELLS


def get_optimal_value_function_and_policy(state_action_function, states, actions):
    """
    Computes the optimal value function and policy for a given set of states and actions,
    based on a provided state-action value function. This function iterates over all states
    and actions, determining the optimal action and its value for each state by maximizing
    the expected value of actions according to the state-action value function.

    Parameters:
        state_action_function (dict): A dictionary mapping (state, action) pairs to value estimates,
                                      representing the expected utility of taking a given action in a given state.
        states (list of tuples): The list of all possible states in the environment, where each
                                 state is represented as a tuple of integers.
        actions (list): The list of all possible actions available to the agent in the environment,
                        where each action is represented in a format specific to the environment's
                        action space (e.g., as integers, tuples, etc.).

    Returns:
        tuple: A tuple containing two elements:
               - state_value_function (numpy.ndarray): An array representing the maximum expected
                     value of each state under the optimal policy. The shape of this array matches
                     the dimensions of the environment's state space.
                - policy (numpy.ndarray): The optimal action (as an action index) for each state.
                    An entry of -1 indicates no preference among actions due to equal value.
                    The shape matches the state space dimensions.
    """

    shape = [max(state[dim] for state in states) + 1 for dim in range(len(states[0]))]
    state_value_function = np.zeros(shape)
    policy = -np.ones(shape=shape, dtype=int)

    for state in states:
        action_values = []

        for action in actions:
            action_values.append(state_action_function.get((state, action), 0))

        if all(x == action_values[0] for x in action_values):
            policy[state] = -1
        else:
            policy[state] = np.argmax(action_values)

        state_value_function[state] = np.max(action_values)

    return state_value_function, policy


def steps_per_episode(agent, iterations=100, n_episodes=50):
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


def plot_optimal_policy_and_state_values(
    state_value_function,
    optimal_policy,
    actions,
    arrows=True,
    goal_state=GOAL_STATE,
    blocked_cells=BLOCKED_CELLS,
):
    """
    Visualizes the optimal policy and state values on a grid. For each state, it shows
    the optimal action to take via arrows and the state value using color intensity.

    Parameters:
        state_value_function (numpy.ndarray): The value function of each state, obtained
                                              from the get_optimal_value_function_and_policy function.
        optimal_policy (numpy.ndarray): The optimal policy for each state, obtained
                                        from the get_optimal_value_function_and_policy function.
        actions (list of tuples): The list of possible actions, with each action represented
                                  as a tuple (dx, dy).
        arrows (bool): If True, the function plots arrows to indicate the optimal action
                       at each state. If False, it only plots the state values.
        goal_state (tuple): The coordinates of the goal state in the grid.

    Returns:
        None: This function does not return a value but visualizes the optimal policy
              and state values on a plot.

    Note:
        This function assumes that matplotlib.pyplot and matplotlib.patches are imported
        as plt and mpatches, respectively. It also assumes the existence of global variables
        X_SIZE and Y_SIZE for the dimensions of the grid.
    """

    v = state_value_function.T
    p = optimal_policy.T

    plt.figure(figsize=(X_SIZE, Y_SIZE))
    plt.xticks(np.arange(0, X_SIZE, 1))
    plt.yticks(np.arange(0, Y_SIZE, 1))
    plt.imshow(v, cmap="viridis", origin="lower")
    plt.title("State Values and Optimal Policy", fontsize=16)
    ax = plt.gca()

    arrows = [tuple(value * 0.25 for value in tup) for tup in actions]

    for i in range(X_SIZE):
        for j in range(Y_SIZE):
            if (i, j) in blocked_cells:
                plt.text(i, j, "X", ha="center", va="center", color="red", fontsize=12)
                rect = mpatches.Rectangle(
                    (i - 0.5, j - 0.5),
                    1,
                    1,
                    linewidth=0,
                    edgecolor="none",
                    facecolor="black",
                )
                ax.add_patch(rect)

            elif (i, j) == goal_state:
                plt.text(i, j, "G", ha="center", va="center", color="k", fontsize=16)

            else:
                if arrows:
                    action_index = p[j, i]

                    if action_index != -1:
                        dx, dy = arrows[action_index]
                        plt.arrow(
                            i, j, dx, dy, color="black", head_width=0.2, head_length=0.2
                        )

                else:
                    plt.text(
                        i, j, f"{v[j, i]:.2f}", ha="center", va="center", color="k"
                    )


def find_optimal_trajectory(env, optimal_policy, actions=None):
    """
    Finds an optimal trajectory from the current state to the goal state in a given environment
    using a specified optimal policy and action set. The function iterates up to a maximum of
    1000 steps to find a sequence of states leading to the goal, leveraging the optimal policy
    to select actions.

    Parameters:
    - env: The environment in which to find the optimal trajectory. The environment must have a
           `state` attribute that can be converted to a tuple with `to_tuple()` and a `step(action)`
           method that executes an action and returns the next state, a reward, and a done flag.
    - optimal_policy (numpy.ndarray or dict): A mapping from state tuples to action indices that
                                              specifies the optimal action to take from each state.
    - actions (list, optional): A list of possible actions. Defaults to ACTIONS if not specified.
                                Each action should be compatible with the `env.step(action)` method.

    Returns:
    - list of tuples: A list of state tuples representing the found trajectory from the initial
                      state to the goal state according to the optimal policy.

    Raises:
    - RuntimeError: If the function fails to find a trajectory to the goal within 1000 iterations,
                    indicating a potential issue such as an infinite loop in the policy.

    Note:
    This function assumes that the environment's `step` method and the `optimal_policy` are correctly
    implemented and that the goal is reachable within 1000 steps. It is primarily used for evaluating
    the effectiveness of a learned policy in deterministic environments.
    """

    if not actions:
        actions = ACTIONS

    state = env.state.to_tuple()
    states = [state]

    done = False

    for _ in range(1000):
        action_index = optimal_policy[state]
        state, _, done = env.step(actions[action_index])
        states.append(state)

        if done:
            break
    else:
        raise RuntimeError(
            "Failed to find a trajectory within 1000 iterations. Check for infinite loops."
        )

    return states


def plot_trajectory(trajectories, blocked_cells=BLOCKED_CELLS):
    """
    Plots the trajectory of an agent's path through a maze, highlighting blocked cells and marking the start and goal locations.

    This visualization helps in understanding the path taken by the agent, including how it navigates around blocked cells,
    and indicates the starting point ('S') and the goal ('G') within the maze.

    Parameters:
        trajectories (list of tuples): A list of (x, y) tuples representing the agent's path through the maze.
        blocked_cells (list of tuples): A list of (x, y) tuples indicating the positions of blocked cells within the maze.

    Note:
        The function assumes a pre-defined global maze size through `X_SIZE` and `Y_SIZE` variables for setting up the plot dimensions.
    """

    _, ax = plt.subplots(figsize=(X_SIZE, Y_SIZE))

    ax.set_xlim(0, X_SIZE)
    ax.set_ylim(0, Y_SIZE)

    ax.set_xticks(range(X_SIZE))
    ax.set_yticks(range(Y_SIZE))
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.grid(which="both", c="k", alpha=0.3)

    for i, j in blocked_cells:
        rect = mpatches.Rectangle(
            (i, j),
            1,
            1,
            linewidth=0,
            edgecolor="grey",
            facecolor="grey",
        )
        ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_alpha(0.2)

    for (x, y), (next_x, next_y) in zip(trajectories, trajectories[1:]):
        ax.annotate(
            "",
            xy=(next_x + 0.5, next_y + 0.5),
            xytext=(x + 0.5, y + 0.5),
            arrowprops={"arrowstyle": "->", "lw": 2},
        )

        if (x, y) == trajectories[0]:
            ax.text(
                x + 0.5,
                y + 0.5,
                "S",
                color="black",
                ha="center",
                va="center",
                fontsize=20,
                c="blue",
            )
        if (next_x, next_y) == trajectories[-1]:
            ax.text(
                next_x + 0.5,
                next_y + 0.5,
                "G",
                ha="center",
                va="center",
                fontsize=20,
                c="blue",
            )

    plt.show()


def create_callback(new_env, target_timestep):
    """
    Creates a callback function that updates the environment of a Dyna Maze instance at a specified timestep.

    Parameters:
        - new_env: The new environment configuration to update to.
        - target_timestep: The simulation timestep at which the environment should be updated.
    """

    def callback(instance):
        if instance.timestep == target_timestep:
            instance.update_environment(new_env)

    return callback


def cumulative_reward_per_timestep(
    agent, env1, env2, iterations=100, n_episodes=150, env_change_timestep=1000
):
    """
    Calculates the average cumulative reward per timestep over multiple episodes and iterations,
    considering an environment change at a specified timestep.

    This function runs a specified number of episodes for a given agent in an initial environment (env1),
    changes the environment to a new configuration (env2) at a predefined timestep, and repeats the process
    for a number of iterations. It tracks the cumulative reward received by the agent at each timestep,
    averaging the results over all iterations to provide insights into the agent's performance dynamics
    across different stages of the environment.

    Parameters:
    - agent: The agent that interacts with the environment.
    - env1: The initial environment configuration.
    - env2: The new environment configuration to switch to.
    - iterations (int): The number of times the simulation is run.
    - n_episodes (int): The number of episodes per iteration.
    - env_change_timestep (int): The timestep at which the environment changes from env1 to env2.

    Returns:
    - A pandas DataFrame with the average cumulative reward per timestep.
    """

    results = []

    callback = create_callback(env2, env_change_timestep)

    for i in range(iterations):
        agent.update_environment(env1)
        _, history = agent.run_simulation(
            n_episodes=n_episodes, reset=True, callback=callback
        )
        results.extend([dict(x, iteration=i) for x in history])

    df = pd.DataFrame(results)

    df["cumulative_reward"] = df.groupby(["iteration"])["reward"].transform("cumsum")

    df = df.groupby(["timestep"], as_index=False).agg(
        cumulative_reward=("cumulative_reward", "mean")
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

    plt.xlim([0, 50])
    plt.xlabel("Episodes", fontsize=16)

    plt.ylim([0, 1000])
    plt.yticks(fontsize=12)
    plt.ylabel("Steps per episode", fontsize=16)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for result in results:
        plt.plot(result["data"]["x"], result["data"]["y"], **result["param"])

    plt.legend(fontsize=16)


def plot_cumulative_reward_per_timestep(
    data, xlim=(0, 3000), ylim=(0, 150), env_change_timestep=1000
):
    plt.xlim(xlim)
    plt.xlabel("Time steps", fontsize=16)

    plt.ylim(ylim)
    plt.yticks(fontsize=12)
    plt.ylabel("Cumulative reward", fontsize=16)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for d in data:
        plt.plot(d["data"]["x"], d["data"]["y"], **d["param"])

    plt.axvline(x=env_change_timestep, color="k", linestyle="--")

    plt.legend(fontsize=16)
