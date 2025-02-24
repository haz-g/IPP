import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from config import CONFIG

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        if CONFIG['CNN_preprocessing']:
            self.input_size = 128  # Output from ImpalaCNN
        else:
            obs_shape = (2,5,5,5)
            self.input_size = np.prod(obs_shape)

        # Two hidden layers with 64 units each and tanh activation
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(self.input_size).prod(), 64)),
            nn.Tanh(), 
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(self.input_size).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 4), std=0.01),
        )
        
    def get_value(self, x):
        x = x.reshape(x.size(0), -1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.reshape(x.size(0), -1) # converts from [batch_size, 2, 5, 5, 5] to [batch_size, 250]
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None: # this logic is to handle the dual functionality of this method for both action selection and policy updates
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

# for NN initialisation inline with original PPO implementation
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer