import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[64, 64]):
        """Initialize parameters and build model.

        Params:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): List of unit counts in each layer of q-netwokr
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self._init_fc_layers(state_size, action_size, fc_units)

    def _init_fc_layers(self, state_size, action_size, fc_units):
        assert isinstance(fc_units, list), f'fc_units must be list, not {type(fc_units)}'
        assert len(fc_units) > 0, 'fc_units must contain at least one unit layer'

        self.fc_layers = list()
        self.fc_layers.append(nn.Linear(state_size, fc_units[0]))
        self.fc_layers += [nn.Linear(fc_units[idx], fc_units[idx + 1]) for idx in range(len(fc_units) - 1)]
        self.fc_layers.append(nn.Linear(fc_units[-1], action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        return self.fc_layers[-1](x)
