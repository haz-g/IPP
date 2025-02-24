import time
import numpy as np

CONFIG = {
    # --- SETUPS ---
    'torch_deterministic': True, # Try not to change
    'cuda': True, # Try not to change
    'track': True, # if toggled, this experiment will be tracked with Weights and Biases
    'wandb_project_name': 'IPP-second-paper-generalist',
    'wandb_entity': 'IPP-experiments',

    # --- ENV SPECIFIC ---
    'train_env_list': 'src/gridworld_construction/gridworlds_200.pkl',
    'test_env_list': 'src/gridworld_construction/gridworlds_test_40.pkl',
    'DREST_lambda_factor': 0.92,
    'meta_ep_size': 32, # After how many mini-env runs would you like to change to a new mini-env
    'num_steps': 128, # per policy rollout
    'Allow_coin_reward_in_traj': False, # If False reward will me accumilated and given as discounted DREST at end of traj only
    'num_envs': 4, # number of parallel envs

    # --- Logging ---
    'test_log_freq': 20, # How many times would you like to collect data on agent performance on a subset of envs
    'test_log_env_subset_size': 0.05, # What percentage of envs would you like to randomly sample for performance eval during training
    'test_on_test_envs': False, # Would you like to log data on test envs (set TRUE) or train envs (set FALSE) throughout training?

    # --- ALGO SPECIFIC ---
    'CNN_preprocessing': True, # If true observations will be pre-processed via IMPALA CNN
    'total_timesteps': 2_500_000,
    'learning_rate': 2.5e-4,
    'anneal_lr': True, # Exponentially decay learning rate for policy and value networks
    'decay_rate': 0.995, # after 'decay_steps' (calculated automatically) lr is decayed by this value
    'decay_depth': 0.4, # by the end of training the code will aim to exponentially decay lr to this amount of its original value
    'gamma': 0.99, # the discount factor gamma
    'gae_lambda': 0.95, # the lambda for the general advantage estimation
    'num_minibatches': 4,
    'update_epochs': 4, # the K epochs to update the policy
    'norm_adv': True, # Toggles advantages normalization
    'clip_coef': 0.2, # the surrogate clipping coefficient
    'clip_vloss': True, # Toggles whether or not to use a clipped loss for the value function, as per original PPO paper
    'ent_coef': 0.03, # coefficient of the entropy
    'vf_coef': 0.5, # coefficient of the value function
    'max_grad_norm': 0.5, # the maximum norm for the gradient clipping
    'target_kl': 0.01, # the target KL divergence threshold
}

# Create a unique run tag for logging and checkpoint naming.
RUN_TAG = time.strftime('%m%d-%H%M')

# Additional Configurations Auto-Complete
CONFIG['model_save_name'] = f'DREST_{RUN_TAG}'
CONFIG['seed'] = np.random.randint(0, 1000000)
CONFIG['batch_size'] = int(CONFIG['num_envs'] * CONFIG['num_steps'])
CONFIG['minibatch_size'] = int(CONFIG['batch_size'] // CONFIG['num_minibatches'])
CONFIG['num_iterations'] = CONFIG['total_timesteps'] // CONFIG['batch_size']
CONFIG['decay_steps'] = int(CONFIG['num_iterations'] / (np.log(CONFIG['decay_depth']) / np.log(CONFIG['decay_rate'])))