import wandb
import random
import numpy as np
import torch
import argparse

def parse_args(config):
    '''Parse command line arguments and update config dictionary.
    
    Supports the following arguments:
        --lr: Learning rate (float)
        --meta_ep_size: Size of meta episodes (int)
        --track: Enable wandb tracking (flag)
        --num_envs: Number of parallel environments (int)
        --anneal_lr: Enable learning rate annealing (flag)
    
    Args:
        config (dict): Configuration dictionary to update with parsed arguments
        
    Returns:
        None, modifies config dictionary in place
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=config['learning_rate'])
    parser.add_argument('--meta_ep_size', type=int, default=config['meta_ep_size'])
    parser.add_argument('--track', action='store_true', default=config['track'])
    parser.add_argument('--num_envs', type=int, default=config['num_envs'])
    parser.add_argument('--anneal_lr', action='store_true', default=config['anneal_lr'])
   
    args = parser.parse_args()
    
    # Update CONFIG with any command line arguments
    config.update(vars(args))
    
def setup_wandb(config):
    '''Initialize Weights & Biases tracking if enabled.
    
    Args:
        config (dict): Configuration dictionary containing wandb settings:
            - track: Whether to enable wandb tracking
            - wandb_project_name: Name of the wandb project
            - wandb_entity: Wandb username/organization
            - model_save_name: Name for saving the model
            
    Returns:
        None, initializes wandb if tracking is enabled
    '''
    if config['track']:
        wandb.init(project=config["wandb_project_name"], 
                   entity=config["wandb_entity"], 
                   config=config, 
                   name=config["model_save_name"], 
                   save_code=True)
            
def normalize_ratio_with_exp(a, b, exponent=2):
    '''Compute normalized ratio between two values raised to a power.
    
    Calculates (min(a,b)/max(a,b))^exponent. Returns 0 if max value is 0.
    
    Args:
        a (float): First value
        b (float): Second value
        exponent (float, optional): Power to raise the ratio to. Defaults to 2.
        
    Returns:
        float: Normalized ratio raised to the given exponent
    '''
    min_val = min(a, b)
    max_val = max(a, b)
    if max_val == 0:
        return 0
    ratio = (min_val / max_val) ** exponent
    return ratio

def seed_run(seed, torch_deterministic):
    '''Set random seeds for reproducibility.
    
    Sets seeds for Python's random, NumPy, and PyTorch random number generators.
    Also configures PyTorch's CUDNN backend for deterministic behavior if specified.
    
    Args:
        seed (int): Random seed to use
        torch_deterministic (bool): Whether to make PyTorch operations deterministic
            Note: Setting this to True may impact performance
        
    Returns:
        None, sets random seeds globally
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

def define_device(cuda):
    '''Determine the appropriate device (CPU/GPU) for running computations.
    
    Args:
        cuda (bool): Whether to use CUDA GPU acceleration if available
        
    Returns:
        torch.device: Device object for either 'cuda' if GPU is available and cuda=True,
                     or 'cpu' otherwise
    '''
    return torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

def load_pretrained_model(agent,load_model, model_path, device):
    '''Load a pretrained model from disk.
    
    Args:
        agent: PPO agent model
        model_path (str): Path to the model file
        device: PyTorch device
        
    Returns:
        agent: Loaded agent model
    '''
    if load_model:
        agent.load_state_dict(torch.load(f"src/models/{model_path}", map_location=device))
        print(f"Successfully loaded model from src/models/{model_path}")
        return agent
    return agent