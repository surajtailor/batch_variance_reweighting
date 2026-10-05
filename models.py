import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ArchitectureConfig:
    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 10
    num_layers: int = 5
    activation_func: str = 'elu'
    shared: bool = False

class HatActivation(nn.Module):
    def __init__(self):
        super(HatActivation, self).__init__()

    def forward(self, x):
        return torch.where((x >= 0) & (x < 1), x, torch.where((x >= 1) & (x < 2), 2 - x, torch.zeros_like(x)))

class MVENetwork(nn.Module):
    def __init__(self,
                 input_size=1,
                 output_size = 3,
                 hidden_size=10,
                 num_layers=5,
                 training=True,
                 shared = False,
                 activation_func='elu'):
        super(MVENetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.training = training
        self.num_layers = num_layers
        self.output_size = output_size
        self.shared = False

        # Set activation function
        self.activation = self._get_activation_function(activation_func)

        if shared:
            # Shared trunk network
            self.shared_net = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=self.hidden_size)
            self.mu = nn.Sequential(self.shared_net, nn.Linear(hidden_size, output_size))
            self.var = nn.Sequential(self.shared_net, nn.Linear(hidden_size, output_size), nn.Softplus())
        else:
            # Define the networks
            self.mu = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=self.output_size)
            self.var = self._build_network(self.input_size, self.hidden_size, self.num_layers, self.activation, final_output_size=self.output_size, final_activation=nn.Softplus())

    def _get_activation_function(self, activation_func):
        """Returns the specified activation function."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leakyrelu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'hat': HatActivation(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation_func, nn.ELU())

    def _build_network(self, input_size, hidden_size, num_layers, activation, final_output_size=None, final_activation=None):
        """Builds a sequential network with a specified number of layers and activations."""
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        if final_output_size:
            layers.append(nn.Linear(hidden_size, final_output_size))
        if final_activation:
            layers.append(final_activation)
        return nn.Sequential(*layers)

    def init_weights(self, method="kaiming"):
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            if method == "kaiming":
                nn.init.kaiming_normal_(m.weight)
            elif method == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif method == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif method == "random":
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
            nn.init.zeros_(m.bias)

    def forward(self, input):
        var = self.var(input)
        mu = self.mu(input)
        return mu, var