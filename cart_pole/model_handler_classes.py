"""Module providing model handlers for training and managing reinforcement learning models."""
import numpy as np
import torch
from torch import nn


class ModelHandler:
    """
    Manages a PyTorch model for training and prediction. Handles device assignment,
    optimizes model parameters, and performs predictions. It automatically handles
    tensor conversions and moves data to the appropriate device.

    Attributes:
        model (torch.nn.Module): The model to manage.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_function (callable): Function to calculate the loss between outputs and targets.
        device (torch.device): Device on which the model is allocated.

    Methods:
        train(states, targets): Trains the model using provided states and targets.
        predict(state): Predict output for the given state.
        reset_weights(): Reinitializes model weights using Xavier uniform or zeros.
        _convert_to_tensor(data): Converts numpy arrays to PyTorch tensors.
    """

    def __init__(self, model, optimizer, loss_function=nn.MSELoss(), device=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.reset_weights()

    def train(self, states, targets):
        states = self._convert_to_tensor(states).to(self.device)
        targets = self._convert_to_tensor(targets).to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(states)
        loss = self.loss_function(outputs, targets.detach())

        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            state = self._convert_to_tensor(state).detach().to(self.device)
            output = self.model(state).cpu().numpy()

        return output

    def reset_weights(self):
        for param in self.model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def _convert_to_tensor(self, data):
        if data.dtype in [np.float32, np.float64]:
            return torch.from_numpy(data.astype(np.float32))
        if data.dtype in [np.int32, np.int64]:
            return torch.from_numpy(data.astype(np.int64))
        raise ValueError(
            f"Unsupported data type: {data.dtype}. Expected float32, float64, int32, or int64."
        )


class ActorModelHandler(ModelHandler):
    """
    Handles training for actor models in reinforcement learning scenarios, with gradient
    clipping defined at initialization. This handler manages a model that computes log
    probabilities for actions, optimizing these log probabilities weighted by advantages.

    Attributes:
        model (torch.nn.Module): The actor model which computes log probabilities.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device, optional): Device on which the model runs.
        clip_grad (float, optional): Maximum norm for gradient clipping.

    Methods:
        train(states, actions, advantages): Trains the model on given states,
            actions, and advantages. Applies gradient clipping if clip_grad is set.
        predict(state): Predicts an action for the given state.
    """

    def __init__(self, model, optimizer, clip_grad=None, device=None):
        super().__init__(model, loss_function=None, optimizer=optimizer, device=device)
        self.clip_grad = clip_grad

    def train(self, states, actions, advantages):
        states = self._convert_to_tensor(states).to(self.device)
        actions = self._convert_to_tensor(actions).to(self.device)
        advantages = self._convert_to_tensor(advantages).to(self.device)

        self.optimizer.zero_grad()

        log_probs = self.model.log_prob(states, actions)

        loss = -(log_probs * advantages.detach()).mean()

        loss.backward()

        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.clip_grad
            )
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            state = self._convert_to_tensor(state).to(self.device)
            action = self.model.sample(state).cpu()

        return action
