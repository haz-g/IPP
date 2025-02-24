from config import CONFIG
import random
import gymnasium as gym
import sys
import pickle

def load_envs(env_list, test_env_list, verbose=True):
    '''Load training and testing environments from pickle files.
    
    Args:
        env_list (str): Path to pickle file containing training environments
        test_env_list (str): Path to pickle file containing testing environments
        verbose (bool, optional): Whether to print environment loading info. Defaults to True.
        
    Returns:
        tuple: Contains:
            - train_env_list (list): List of training gridworld environments
            - test_env_list (list): List of testing gridworld environments
            
    Example:
        >>> train_envs, test_envs = load_envs("train.pkl", "test.pkl")
        5000 training gridworlds loaded
        1000 testing gridworlds loaded
    '''
    sys.path.append('src')
    with open(env_list, "rb") as f:
        train_env_list = pickle.load(f)
    with open(test_env_list, "rb") as t:
        test_env_list = pickle.load(t)
    
    if verbose:
        print(f"{len(train_env_list)} training gridworlds loaded")
        print(f"{len(test_env_list)} testing gridworlds loaded")
        
    return train_env_list, test_env_list

# Function for parallel environment initialisation using SyncVectorEnv
def init_vec_envs(train_env_list):
    '''Initialize vectorized environments for parallel training.
    
    Creates a SyncVectorEnv object that runs multiple environments in parallel.
    Each environment is randomly sampled from the training environment list.
    Only supports discrete action spaces.
    
    Args:
        train_env_list (list): List of training environments to sample from
        
    Returns:
        gym.vector.SyncVectorEnv: Vectorized environment object running CONFIG['num_envs']
            parallel environments
            
    Raises:
        AssertionError: If the environment action space is not discrete
        
    Note:
        Uses CONFIG['num_envs'] to determine the number of parallel environments
    '''
    def make_env():
        idx = random.sample(range(len(train_env_list)), k=1)[0]
        return train_env_list[idx]

    # Parallel env setup - SyncVectorEnv expects a list of functions
    envs = gym.vector.SyncVectorEnv(
        [make_env for _ in range(CONFIG['num_envs'])],
    )
    
    # Ensure setup was correct
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    return envs