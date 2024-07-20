"""
Provides a class for defining an objective function that integrates Optuna hyperparameter
optimization with a training process for reinforcement learning models.
"""
from model_utils import train


class ObjectiveFunction:
    """
    Defines an objective function for optimizing model hyperparameters using Optuna.

    This class is intended to be used with the Optuna optimization framework to dynamically
    create and evaluate models based on trial hyperparameters. It manages the lifecycle of
    a model's training and evaluation across multiple iterations to assess the overall
    performance.

    Attributes:
        create_model (callable): A factory function to create a new model instance per trial.
        callbacks (list, optional): List of callbacks to be used during model training.
        iterations (int): Number of times the training process should be repeated.
        train_episodes (int): Number of training episodes per iteration.
        evaluation_episodes (int): Number of episodes for evaluating the model performance.
        evaluation_interval (int): Interval at which the model is evaluated.

    Methods:
        __call__(trial):
            Called by the Optuna optimization framework for each trial, creating and evaluating
            a model based on trial-specified hyperparameters. It returns the cumulative number
            of episodes run across all iterations, which can be used as an objective measure.
    """

    def __init__(
        self,
        create_model,
        callbacks=None,
        iterations=10,
        train_episodes=500,
        evaluation_episodes=10,
        evaluation_interval=1,
    ):
        self.create_model = create_model
        self.callbacks = callbacks
        self.iterations = iterations
        self.train_episodes = train_episodes
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_interval = evaluation_interval

    def __call__(self, trial):
        model = self.create_model(trial)
        total_episodes = []

        for _ in range(self.iterations):
            results = train(
                model,
                train_episodes=self.train_episodes,
                evaluation_episodes=self.evaluation_episodes,
                evaluation_interval=self.evaluation_interval,
                callbacks=self.callbacks,
                verbose=False,
            )

            total_episodes.append(results[-1]["episode"])
            model.reset_simulation()

        return sum(total_episodes)
