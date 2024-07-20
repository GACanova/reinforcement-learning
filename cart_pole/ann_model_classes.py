"""This module defines neural network models for continuous and discrete action spaces using PyTorch."""
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from config import INPUT_DIM, OUTPUT_DIM, HIDDEN_UNITS, ACTIVATION, N_ACTIONS


class FCN(nn.Module):
    """
    A fully connected network (FCN) consisting of a series of linear layers separated by activation functions.

    Args:
        input_dim (int): Dimension of the input feature vector.
        output_dim (int): Dimension of the output.
        hidden_units (list of int): List specifying the number of units in each hidden layer.
        activation (nn.Module): An instance of a PyTorch activation function to use after each linear layer.
                                Example: nn.ReLU() or nn.ELU(alpha=1.0)
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_units=HIDDEN_UNITS,
        activation=ACTIVATION,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(activation)

        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_units[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SoftmaxFCN(nn.Module):
    """
    A neural network model designed for classification that outputs logits for softmax activation.
    Useful for scenarios requiring both class predictions and associated probabilities for tasks
    like sampling and probability density estimation.

    The architecture includes fully connected layers with specified activation functions, ending
    with a layer that outputs logits. These logits can be transformed into a probability distribution
    using softmax, supporting methods for log probability calculations and sampling.

    Args:
        input_dim (int): Dimension of the input feature vector.
        output_dim (int): Number of classes for output logits.
        hidden_units (list of int): Units in each hidden layer.
        activation (nn.Module): An instance of a PyTorch activation function to use after each linear layer.
                                Example: nn.ReLU() or nn.ELU(alpha=1.0)

    Methods:
        forward(x): Returns the logits from the input features.
        log_prob(x, indices): Computes log probabilities for indices from the logits' softmax.
        sample(x): Samples from the softmax distribution defined by the logits.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        output_dim=N_ACTIONS,
        hidden_units=HIDDEN_UNITS,
        activation=ACTIVATION,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(activation)

        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_units[-1], output_dim))

        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.feature_extractor(x)

        return logits

    def log_prob(self, x, indices):
        logits = self.forward(x)
        log_probs = F.log_softmax(logits, dim=-1).gather(1, indices)

        return log_probs

    def sample(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probabilities, 1).squeeze(-1)

        return sample


class GaussianFCN(nn.Module):
    """
    A neural network that outputs Gaussian distribution parameters: mean and
    logarithm of standard deviation. Suitable for regression or probabilistic
    modeling tasks where outputs are expected to have a Gaussian distribution.

    The architecture comprises fully connected layers with activation functions,
    followed by output layers predicting the mean and log standard deviation.

    Args:
        input_dim (int): Dimension of the input feature vector.
        output_dim (int): Dimension for the output, typically 1 for univariate.
        hidden_units (list of int): List of the number of units in each hidden layer.
        activation (nn.Module): PyTorch activation function instance used after each linear layer.
        output_scale (float): Scaling factor for the output of the sample method.

    Methods:
        forward(x):
            Performs the forward pass of the network, returning the mean and standard deviation.

        log_prob(x, targets):
            Computes the log probability of the targets under the Gaussian distribution defined by the model outputs.

        sample(x):
            Samples from the Gaussian distribution defined by the model outputs.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        output_dim=1,
        hidden_units=HIDDEN_UNITS,
        activation=ACTIVATION,
        output_scale=1.0,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(activation)
        self.output_scale = output_scale

        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(activation)

        self.feature_extractor = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(hidden_units[-1], output_dim)
        self.log_std = nn.Parameter(
            torch.full((output_dim,), torch.log(torch.tensor(0.1)))
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        mean = self.mean_layer(x)
        std = torch.clamp(torch.exp(self.log_std), 1e-3, 50)

        return mean, std

    def log_prob(self, x, targets):
        mean, std = self.forward(x)
        normal = distributions.Normal(mean, std)
        log_probs = normal.log_prob(targets).sum(-1, keepdim=True)

        return log_probs

    def sample(self, x):
        with torch.no_grad():
            mean, std = self.forward(x)
            normal = distributions.Normal(mean, std)
            sample = normal.sample()
            sample = torch.tanh(sample) * self.output_scale

        return sample
