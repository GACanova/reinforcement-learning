import numpy as np
from poisson import Poisson


class JacksCarRental:
    def __init__(self, max_n_car=20, return_cost=2, rent_profit=10, lambda_request_1=3, lambda_request_2=4,
                 lambda_return_1=3, lambda_return_2=2, gamma=0.9, probability_threshold=1e-5):
        """
        Represents a car rental service managing two locations, designed to find an optimal policy
        for car distribution using reinforcement learning (specifically policy iteration).

        This class models the environment of a car rental business where the number of cars requested
        and returned at each location is stochastic, following a Poisson distribution. The goal is to
        maximize profits by determining the optimal number of cars that should be moved between the two
        locations overnight.
        
        Attributes:
            max_n_car (int): Maximum number of cars at each location.
            return_cost_per_car (int): Cost of moving a car between locations.
            rent_profit_per_car (int): Profit made from renting a single car.
            lambda_request_1 (float): Expected number of rental requests at location 1.
            lambda_request_2 (float): Expected number of rental requests at location 2.
            lambda_return_1 (float): Expected number of cars returned at location 1.
            lambda_return_2 (float): Expected number of cars returned at location 2.
            gamma (float): Discount factor for future rewards.
            probability_threshold (float): Threshold for considering Poisson probabilities as significant.
            poisson (Poisson): An instance of the Poisson class for probability calculations.
            v (numpy.ndarray): The value function, representing the value of each state.
            policy (numpy.ndarray): The current policy, representing the action to take at each state.
            actions (numpy.ndarray): The set of possible actions (number of cars to move overnight).

        Methods:
            state_generator(lambda_): Generates possible states based on the Poisson distribution until the probabilities fall below the probability_threshold.
            initialize_value_function(random=False): Initializes the value function, randomly or with zeros.
            create_actions(max_moved_car=5): Initializes the set of possible actions.
            initialize_policy(): Initializes the policy with no cars being moved.
            expected_return(state, action): Calculates the expected return for a state-action pair.
            policy_evaluation(): Evaluates the current policy, updating the value function.
            policy_improvement(): Improves the policy based on the updated value function.
            policy_iteration(): Applies policy iteration to find the optimal policy.
        """
        
        self.max_n_car = max_n_car
        self.return_cost_per_car = return_cost
        self.rent_profit_per_car = rent_profit
        self.lambda_request_1 = lambda_request_1
        self.lambda_request_2 = lambda_request_2
        self.lambda_return_1 = lambda_return_1
        self.lambda_return_2 = lambda_return_2
        self.gamma = gamma
        self.probability_threshold = probability_threshold
        self.poisson = Poisson()
        
    def state_generator(self, lambda_):
        n = 0
        while self.poisson.probability(n, lambda_) > self.probability_threshold:
            yield n
            n += 1
        
    def initialize_value_function(self, random=False):
        if random:
            self.v = np.random.normal(0, 1, (self.max_n_car + 1, self.max_n_car + 1))
        else:
            self.v = np.zeros((self.max_n_car + 1, self.max_n_car + 1))
        
    def create_actions(self, max_moved_car=5):
        self.actions = np.arange(-max_moved_car, max_moved_car + 1, 1)
        
    def initialize_policy(self):
        self.policy = np.zeros((self.max_n_car + 1, self.max_n_car + 1), dtype=int)
        
    def expected_return(self, state, action):
        n_cars_1 = min(state[0] - action, self.max_n_car)
        n_cars_2 = min(state[1] + action, self.max_n_car)
                
        return_cost = abs(action) * self.return_cost_per_car
        
        final_return = -return_cost 
        
        for n_request_1 in self.state_generator(self.lambda_request_1):
            for n_request_2 in self.state_generator(self.lambda_request_2):
                
                p_request_1 = self.poisson.probability(n_request_1, self.lambda_request_1) 
                p_request_2 = self.poisson.probability(n_request_2, self.lambda_request_2)
                
                n_rented_cars_1 = min(n_request_1, n_cars_1)
                n_rented_cars_2 = min(n_request_2, n_cars_2)
                
                rent_profit = (n_rented_cars_1 + n_rented_cars_2) * self.rent_profit_per_car
                            
                for n_returned_1 in self.state_generator(self.lambda_return_1):
                    for n_returned_2 in self.state_generator(self.lambda_return_2):
                        
                        p_returned_1 = self.poisson.probability(n_returned_1, self.lambda_return_1)
                        p_returned_2 = self.poisson.probability(n_returned_2, self.lambda_return_2)
                                                
                        final_n_cars_1 = min(n_cars_1 - n_rented_cars_1 + n_returned_1, self.max_n_car)
                        final_n_cars_2 = min(n_cars_2 - n_rented_cars_2 + n_returned_2, self.max_n_car)
                        
                        final_probability = p_request_1 * p_request_2 * p_returned_1 * p_returned_2
                                                
                        final_return += final_probability * (rent_profit + self.gamma*self.v[final_n_cars_1, final_n_cars_2])
                        
        return final_return
            
    
    def policy_evaluation(self):
        
        iterations = 0
        theta = 1e-2
        delta = theta + 1
        max_iterations = 1e6
        
        while delta > theta and iterations < max_iterations:
            delta = 0 
            for n_cars_1 in range(0, self.max_n_car + 1):
                for n_cars_2 in range(0, self.max_n_car + 1):
                    state = n_cars_1, n_cars_2

                    action = self.policy[state]
                    v_old = self.v[state]
                    self.v[state] = self.expected_return(state, action)
                    delta = max(delta, np.abs(v_old - self.v[state]))
                                        
            iterations += 1
            print(f"Policy Evaluation - Iteration {iterations}: Max ΔV = {delta:.6f}")
            
    def policy_improvment(self):
        policy_stable = True
        policy_changes = 0 
        
        for n_cars_1 in range(0, self.max_n_car + 1):
            for n_cars_2 in range(0, self.max_n_car + 1):

                state = n_cars_1, n_cars_2

                old_action = self.policy[state]
                returns = []

                for action in self.actions:
                    if 0 <= n_cars_1 - action <= self.max_n_car and 0 <= n_cars_2 + action <= self.max_n_car:
                        returns.append(self.expected_return(state, action))
                    else:
                        returns.append(float("-inf"))
                        
                self.policy[state] = self.actions[np.argmax(returns)]

                if self.policy[state] != old_action:
                    policy_stable = False
                    policy_changes += 1
                    
        print(f"Policy Improvement - Policy Changed in {policy_changes} States")
        return policy_stable
                    
                    
    def policy_iteration(self):
        
        self.create_actions()
        self.initialize_policy()
        self.initialize_value_function()
                
        policy_history = []
        value_function_history = []
        policy_history.append(np.array(self.policy))
        value_function_history.append(np.array(self.v))
        
        policy_stable = False
        max_iterations = 1e6
        iterations = 0
                
        while not policy_stable and iterations < max_iterations:
            
            self.policy_evaluation()
            policy_stable = self.policy_improvment()
            
            policy_history.append(np.array(self.policy))
            value_function_history.append(np.array(self.v))
            iterations += 1
            
            print(f"Policy Iteration - Iteration {iterations}: Policy Stable = {policy_stable}")
            print("------------------------------------------------\n")
            
        if policy_stable:
            print("Policy Iteration Converged")
        else:
            print("Policy Iteration reached maximum iterations")
            
        return policy_history, value_function_history
