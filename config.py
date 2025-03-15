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
    'train_env_list': 'src/gridworld_construction/gridworlds_200_singletime.pkl',
    'test_env_list': 'src/gridworld_construction/gridworlds_test_40_singletime.pkl',
    'DREST_lambda_factor': 0.9,
    'meta_ep_size': 64, # After how many mini-env runs would you like to change to a new mini-env
    'num_steps': 2048, # per policy rollout
    'Allow_coin_reward_in_traj': False, # If False reward will me accumilated and given as discounted DREST at end of traj only
    'num_envs': 60, # number of parallel envs
    'penalise_wall_collisions': True, # If True agent penalised for colliding with walls

    # --- Logging ---
    'test_log_freq': 10, # How many times would you like to collect data on agent performance on a subset of envs
    'test_log_env_subset_size': 0.20, # What percentage of envs would you like to randomly sample for performance eval during training

    # --- ALGO SPECIFIC ---
    'CNN_preprocessing': False, # If true observations will be pre-processed via IMPALA CNN
    'total_timesteps': 1_000_000,
    'learning_rate': 0.000001,
    'anneal_lr': True, # Exponentially decay learning rate for policy and value networks
    'decay_rate': 0.990, # after 'decay_steps' (calculated automatically) lr is decayed by this value
    'decay_depth': 0.1, # by the end of training the code will aim to exponentially decay lr to this amount of its original value
    'gamma': 0.99, # the discount factor gamma
    'gae_lambda': 0.95, # the lambda for the general advantage estimation
    'num_minibatches': 32,
    'update_epochs': 10, # K epochs to update the policy
    'norm_adv': True, # Toggles advantages normalization
    'clip_coef': 0.2, # the surrogate clipping coefficient
    'clip_vloss': True, # Toggles whether or not to use a clipped loss for the value function, as per original PPO paper
    'initial_and_final_ent_coef': (0.01, 0.01), # the initial and final coefficient of the entropy (decays from first to second)
    'vf_coef': 0.5, # coefficient of the value function
    'max_grad_norm': 0.5, # the maximum norm for the gradient clipping
    'target_kl': None, # the target KL divergence threshold

    # --- Model Loading ---
    'load_model': True,
    'model_to_load': 'D141303U53N65.pt',
}

# Create a unique run tag for logging and checkpoint naming.
RUN_TAG = time.strftime('%d%H%M')

# Additional Configurations Auto-Complete
CONFIG['model_save_name'] = f'D{RUN_TAG}'
CONFIG['seed'] = 2744 # np.random.randint(0, 1000000)
CONFIG['batch_size'] = int(CONFIG['num_envs'] * CONFIG['num_steps'])
CONFIG['minibatch_size'] = int(CONFIG['batch_size'] // CONFIG['num_minibatches'])
CONFIG['num_iterations'] = CONFIG['total_timesteps'] // CONFIG['batch_size']