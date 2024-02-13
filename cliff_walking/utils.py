"""
This module encompasses the implementation of various tools and utilities for analyzing
and visualizing the performance of reinforcement learning agents within a CliffWalking
environment. It includes functions for computing optimal value functions and policies,
plotting trajectories, analyzing average rewards per episode, and visualizing average
rewards and optimal policies.

Key Components:
- Functions to compute and visualize the optimal value function and policy given a
  state-action value function.
- Utilities for plotting the trajectory of an agent following an optimal policy in
  the CliffWalking environment, highlighting important features such as start and
  goal states, and cliff locations.
- Tools for evaluating and visualizing the performance of agents through average
  rewards per episode, aiding in the assessment of learning and decision-making
  strategies over time.

Dependencies:
- numpy for numerical operations.
- pandas for data manipulation and analysis.
- matplotlib for creating static, interactive, and animated visualizations.
"""


from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import X_SIZE, Y_SIZE, GOAL_STATE, ACTIONS, CLIFF_CELLS


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
               - policy (numpy.ndarray): An array representing the optimal action to take in each
                 state, expressed as an index into the list of actions. The shape of this array
                 matches the dimensions of the environment's state space.
    """

    shape = [max(state[dim] for state in states) + 1 for dim in range(len(states[0]))]
    state_value_function = np.zeros(shape)
    policy = -np.ones(shape=shape, dtype=int)

    for state in states:
        action_values = []

        for action in actions:
            action_values.append(
                state_action_function.get((state, action), float("-inf"))
            )

        state_value_function[state] = np.max(action_values)
        policy[state] = np.argmax(action_values)

    return state_value_function, policy


def episodes_per_timestep(history):
    """
    Plots the number of episodes completed over time steps, given a history of
    episodes and their respective time steps.

    Parameters:
        history (list of dicts): A history of episodes, where each dictionary contains
                                 'episode' and 'reward' keys.

    Returns:
        None: This function does not return a value but plots the cumulative number of
              episodes over time steps.
    """

    episodes, counts = np.unique([x["episode"] for x in history], return_counts=True)
    cumulative_counts = np.cumsum(counts)
    cumulative_counts = np.insert(cumulative_counts, 0, 0)
    episodes = np.insert(episodes, 0, 0)

    plt.xlim([0, 8000])
    plt.ylim([0, 200])
    plt.xlabel("Time steps", fontsize=16)
    plt.ylabel("Episodes", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.plot(cumulative_counts, episodes, c="red")


def plot_optimal_policy_and_state_values(
    state_value_function, optimal_policy, actions, arrows=True, goal_state=GOAL_STATE
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
            if v[j, i] == float("-inf"):
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

                    if actions[action_index] == (0, 0):
                        plt.text(
                            i,
                            j,
                            "STOP",
                            ha="center",
                            va="center",
                            color="red",
                            fontweight="bold",
                        )
                    else:
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


def plot_trajectory(trajectories, cliff_cells=None):
    """
    Plots the optimal policy trajectory from the start in CliffWalking.

    Visualizes the agent's path from start to goal, highlighting cliffs and
    trajectory. The grid shows movements as arrows, with 'S' for start and 'G'
    for goal.

    Parameters:
        trajectories (list): Agent's (x, y) path positions.
        cliff_cells (list, optional): Cliff (x, y) locations. Defaults to global.

    Generates a plot with the grid, cliffs in grey, agent's path, and labels for
    start/goal positions. 'The Cliff' is annotated for clarity.
    """

    if not cliff_cells:
        cliff_cells = CLIFF_CELLS

    _, ax = plt.subplots(figsize=(X_SIZE, Y_SIZE))

    ax.set_xlim(0, X_SIZE)
    ax.set_ylim(0, Y_SIZE)

    ax.set_xticks(range(X_SIZE))
    ax.set_yticks(range(Y_SIZE))
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.grid(which="both", c="k", alpha=0.3)

    for i, j in cliff_cells:
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

    cliff_center_x = (
        min(cliff_cells, key=itemgetter(0))[0] + max(cliff_cells, key=itemgetter(0))[0]
    ) / 2
    cliff_y_position = cliff_cells[0][1]
    ax.text(
        cliff_center_x + 0.5,
        cliff_y_position + 0.5,
        "The Cliff",
        ha="center",
        va="center",
        color="k",
        fontsize=25,
    )
    plt.show()


def avg_rewards_per_episode(agent, iterations=1000, n_episodes=500):
    """
    Calculates average rewards per episode over multiple iterations.

    Runs simulations to gather rewards per episode for a given agent, then averages
    these rewards across all iterations. Useful for analyzing the performance and
    learning progression of reinforcement learning algorithms.

    Parameters:
        agent: The agent implementing `run_simulation`.
        iterations (int, optional): Number of simulation runs. Default is 1000.
        n_episodes (int, optional): Episodes per simulation run. Default is 500.

    Returns:
        DataFrame: A pandas DataFrame with average rewards for each episode.

    This function extends the agent's history with iteration info, aggregates rewards
    by episode across iterations, and computes the mean reward per episode.
    """

    results = []

    for i in range(iterations):
        _, history = agent.run_simulation(n_episodes=n_episodes)
        results.extend([dict(x, iteration=i) for x in history])

    df = (
        pd.DataFrame(results)
        .groupby(["iteration", "episode"], as_index=False)
        .agg(reward=("reward", "sum"))
        .groupby(["episode"], as_index=False)
        .agg(reward=("reward", "mean"))
    )
    return df


def plot_avg_rewards(results):
    """
    Plots the average rewards per episode from simulation results.

    Visualizes the progression of sum of rewards during each episode over the course
    of simulations, aiding in the evaluation of agent learning performance. The
    function sets the plot's y-axis limits and labels, and iterates over multiple
    result sets to plot each as a separate line on the graph.

    Parameters:
        results (list of dicts): Each dict contains 'data' with 'x' (episodes)
        and 'y' (average rewards) arrays, and 'param' with plotting parameters.
    """

    plt.ylim([-100, -15])
    plt.yticks(np.arange(-100, -20, 25))

    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Sum of rewards during episode", fontsize=14)

    for result in results:
        plt.plot(result["data"]["x"], result["data"]["y"], **result["param"])

    plt.legend(fontsize=12)
