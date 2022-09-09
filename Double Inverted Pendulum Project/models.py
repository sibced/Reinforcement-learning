"models: neural networks to use in the project"

import torch
import torch.nn.functional as F

from torch import nn


_WIDTH = 128


class PolicyContinuousModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(9, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, 1),
            ]
        )
        self.log_std = torch.tensor(-0.69)
        # self.log_std = torch.nn.Parameter(self.log_std)  # just diminishes toward 0...

    def forward(self, value: torch.Tensor):
        if len(value.shape) == 1:
            value = value.view((1, -1))
        for linear in self.layers[:-1]:
            value = F.relu(linear(value))
        return torch.tanh(self.layers[-1](value))

    def get_policy(self, epsilon: float = 0.0):
        def policy_(value):
            action = self(torch.tensor(value, dtype=torch.float32))
            if epsilon == 0.0:
                return action

            noise = torch.rand_like(action) * 2 - 1
            prob = torch.rand_like(action)

            return torch.where(prob < epsilon, noise, action)

        return policy_

    def posterior(self, observation, action):
        "returns pi(action | observation)"
        prediction = self(observation)

        std = torch.exp(self.log_std)
        prefactor = 1 / (std * 2.50662827463)  # sqrt(2 pi)

        diff_2 = (action - prediction) ** 2
        arg_exp = (-0.5 / std**2) * diff_2

        return prefactor * torch.exp(arg_exp)


class PolicyDiscreteModel(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()

        self.values = torch.tensor(values, dtype=torch.float32)
        assert len(self.values.shape) == 1

        self.layers = nn.ModuleList(
            [
                nn.Linear(9, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, self.values.shape[0]),
            ]
        )

    def forward(self, value):
        for linear in self.layers[:-1]:
            value = F.relu(linear(value))
        # directly return the logits
        return self.layers[-1](value)

    def get_policy(self, epsilon: float = 0.0):
        def policy_(value):
            # policy assumes a single dimension array
            prediction = self(torch.tensor(value, dtype=torch.float32))
            idx = prediction.argmax(axis=-1)

            if epsilon > 0:
                noise = torch.randint_like(idx, high=self.values.shape[0])
                prob = torch.rand_like(idx, dtype=torch.float32)
                idx = torch.where(prob < epsilon, noise, idx)

            return self.values[idx].view((1,))

        return policy_

    def posterior(self, observation, action):
        idx = torch.argmin((action - self.values) ** 2, axis=-1)
        logits = self(observation)
        prob = F.softmax(logits, dim=-1)

        return torch.diagonal(prob[:, idx])


class ValueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(9, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, _WIDTH),
                nn.Linear(_WIDTH, 1),
            ]
        )

    def forward(self, value):
        for linear in self.layers[:-1]:
            value = F.relu(linear(value))
        # no activation on the last layer
        return self.layers[-1](value)
