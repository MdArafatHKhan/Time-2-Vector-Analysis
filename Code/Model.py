from periodic_activations import SineActivation, CosineActivation
from Data import ToyDataset
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Model, self).__init__()
        if activation == "sin":
            in_features = 1
            self.l1 = SineActivation(in_features, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(in_features, hiddem_dim)
        
        self.fc1 = nn.Linear(hiddem_dim, 2)
    
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x
