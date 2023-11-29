from scipy.stats import poisson

class Poisson:
    """
    A caching class for efficiently computing and storing Poisson distribution probabilities.

    The Poisson class provides a mechanism to compute the probability mass function (pmf) values
    for the Poisson distribution and cache these values for future reference. This reduces
    computation time significantly when the same pmf values are needed repeatedly.
    
    Methods:
        __init__: Initializes the Poisson class with an empty dictionary for storing Poisson probabilities.
        probability(n, lambda_): Computes the Poisson probability for a given value 'n' and rate parameter 'lambda_'.

    Attributes:
        stored_values (dict): A dictionary to store computed Poisson probabilities.
    """
    
    def __init__(self):
        self.stored_values = {}
        
    def probability(self, n, lambda_):
        if (n, lambda_) not in self.stored_values:
            self.stored_values[n, lambda_] = poisson.pmf(n, lambda_)
        return self.stored_values[n, lambda_]
