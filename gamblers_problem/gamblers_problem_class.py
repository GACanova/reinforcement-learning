import numpy as np

class GamblersProblem:
    """
    Represents the Gambler's Problem, a standard example of reinforcement learning involving a gambler 
    who aims to reach a target amount of capital by betting on a series of coin flips.

    The gambler has a chance 'p_h' of winning each bet, which increases their capital by the amount 
    wagered, and a (1 - p_h) chance of losing the bet, which decreases their capital by the wagered amount. 
    The gambler's goal is to reach a predefined maximum capital, at which point they win the game.

    This class implements the solution to this problem using value iteration, a dynamic programming 
    technique. The solution involves determining the optimal policy: a strategy that tells the gambler 
    how much to wager in each state (current capital level) to maximize the probability of reaching the 
    target capital.

    Attributes:
        p_h (float): The probability of the coin landing heads (winning the bet).
        max_capital (int): The target capital the gambler aims to reach.
        v (numpy.ndarray): The value function, representing the maximum probability of winning from each state.
        policy (numpy.ndarray): The policy function, representing the optimal wager in each state.

    Methods:
        initialize_value_function(): Initializes the value function to zeros with a value of 1 at the target capital.
        initialize_policy(): Initializes the policy to always bet one unit.
        max_return(state): Computes the maximum expected return for a given state and the corresponding action.
        value_iteration(): Performs the value iteration algorithm to find the optimal value function and policy.
    """
    
    def __init__(self, p_h=0.5, max_capital=100, seed=42):
        self.p_h = p_h
        self.max_capital = max_capital
        np.random.seed(seed)
        
    def initialize_value_function(self):
        self.v = np.zeros(self.max_capital + 1)
        self.v[self.max_capital] = 1
    
    def initialize_policy(self):
        self.policy = np.ones(self.max_capital + 1, dtype=int)
        
    def max_return(self, state):
        actions = []
        returns = []
        
        for action in range(1, min(state, self.max_capital - state) + 1):
            
            value = self.p_h * self.v[state + action] + (1 - self.p_h) * self.v[state - action]
            
            actions.append(action)
            returns.append(value)
        
        best_return = np.max(returns)
        best_action_index = np.argmax(np.random.random(len(returns)) * (returns==max(returns))) #Select best action with random tie breaking
        best_action = actions[best_action_index]
        
        return best_return, best_action
                
    def value_iteration(self):
        
        iterations = 0
        theta = 1e-4
        delta = theta + 1
        max_iterations = 1e8
        
        self.initialize_policy()
        self.initialize_value_function()
        
        while delta > theta and iterations < max_iterations:
            delta = 0 
            
            for state in range(1, self.max_capital):

                v_old = self.v[state]
                self.v[state], self.policy[state] = self.max_return(state)
                delta = max(delta, np.abs(v_old - self.v[state]))
            
            print(f"Value Iteration - Iteration {iterations}: Max ΔV = {delta:.6f}")
            iterations += 1
            
        if iterations < max_iterations:
            print("Value Iteration Converged")
        else:
            print("Value Iteration reached maximum iterations")
            
            
        for state in range(1, self.max_capital):
            self.v[state], self.policy[state] = self.max_return(state)
            
        return self.v, self.policy
