import torch
import time
import numpy as np
import random
import torch.nn as nn
import wandb
import plotly.figure_factory as ff
import timeout_decorator
from src.utilities.IMPALA_CNN import ImpalaCNN
from src.utilities.evals_utils import evaluate_agent
from src.utilities.config_device import normalize_ratio_with_exp
from math import isnan

def init_pre_processor(CNN_preprocessing, device):
    '''Initialize the IMPALA CNN preprocessor if enabled with optional torch compilation.
    Only attempts compilation when using CUDA device.
    '''
    if CNN_preprocessing:
        pre_processor = ImpalaCNN(in_channels=10, feature_dim=128).to(device)
        
        if device.type == 'cuda' and hasattr(torch, 'compile'):
            pre_processor = torch.compile(pre_processor)
        
        return pre_processor
    
    return False

def setup_storage(CNN_preprocessing: bool, num_steps: int, num_envs: int, device, envs):
    '''Initialize storage tensors for collecting experiences during training.
    
    Creates zero tensors for storing observations, actions, log probabilities,
    rewards, done flags, and values across multiple environments and timesteps.
    
    Args:
        CNN_preprocessing (bool): Whether CNN preprocessing is enabled
        num_steps (int): Number of steps to store in buffer
        num_envs (int): Number of parallel environments
        device: PyTorch device to store tensors on
        envs: Vector environment object
        
    Returns:
        dict: Storage tensors dictionary containing:
            - obs: Shape (num_steps, num_envs, feature_dim) if CNN_preprocessing else (num_steps, num_envs, *obs_shape)
            - actions: Shape (num_steps, num_envs, *action_shape)
            - logprobs: Shape (num_steps, num_envs)
            - rewards: Shape (num_steps, num_envs)
            - dones: Shape (num_steps, num_envs)
            - values: Shape (num_steps, num_envs)
    '''
    if CNN_preprocessing:
        obs = torch.zeros((num_steps, num_envs, 128)).to(device)
    else:
        obs = torch.zeros((num_steps, num_envs) + (10,5,5)).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    return {'obs': obs, 
            'actions': actions,
            'logprobs': logprobs, 
            'rewards': rewards, 
            'dones': dones, 
            'values': values}

def process_observation(CNN_preprocessing: bool, num_envs: int, obs, pre_processor, device):
    '''Process raw observations through optional CNN preprocessing.
    
    Args:
        CNN_preprocessing (bool): Whether to use CNN preprocessing
        num_envs (int): Number of parallel environments
        obs: Raw observation tensor
        device: PyTorch device
        
    Returns:
        torch.Tensor: Processed observation tensor, either through CNN or basic reshaping
        
    Note:
        Reshapes input to (num_envs, 2 * 5, 5, 5) before processing
    '''
    next_obs = torch.Tensor(obs).view(num_envs, 2 * 5, 5, 5).to(device)

    if CNN_preprocessing:
        with torch.no_grad():
            next_obs = pre_processor(next_obs)
        next_obs.to(device)
    else:
        next_obs = torch.Tensor(next_obs).to(device)
    
    return next_obs

def setup_loop_params(device, num_envs):
    '''Initialize parameters for training loop.
    
    Args:
        device: PyTorch device
        num_envs (int): Number of parallel environments
        
    Returns:
        dict: Training loop parameters including:
            - env_steps: Step counter for environment interactions
            - total_steps: Total steps across all environments
            - cur_best_avr_usefulness: Best average usefulness score achieved
            - cur_best_avr_neutrality: Best average neutrality score achieved
            - start_time: Training start timestamp
            - meta_ep_traj_counter: Counter for trajectory types per environment
            - next_done: Done flags tensor for each environment
    '''
    return {'env_steps': 0, 
            'total_steps': 0, 
            'cur_best_avr_usefulness': 0, 
            'cur_best_avr_neutrality': 0, 
            'start_time': time.time(), 
            'meta_ep_traj_counter': [[0,0] for _ in range(num_envs)], 
            'next_done': torch.zeros(num_envs).to(device)}

def anneal_learning_rate(optimizer, iteration, learning_rate, num_iterations, decay_rate, anneal_lr):
    '''Decay the learning rate based on training progress if enabled.
    
    Args:
        optimizer: PyTorch optimizer
        iteration (int): Current training iteration
        learning_rate (float): Initial learning rate
        num_iterations (int): Total number of iterations
        decay_rate (float): Rate of exponential decay
        anneal_lr (bool): Whether to enable learning rate annealing
        
    Note:
        Updates optimizer's learning rate in-place if annealing is enabled
    '''
    if anneal_lr:
        progress = float(iteration) / float(num_iterations)
        decayed_lr = learning_rate * (decay_rate ** progress)
        optimizer.param_groups[0]["lr"] = decayed_lr

def anneal_entropy_coef(iteration, initial_and_final_ent_coef, num_iterations):
    '''Decay the entropy coefficient based on training progress if enabled.
    
    Args:
        iteration (int): Current training iteration
        initial_and_final_ent_coef (tuple): initial and final desired entropy coefficients
        num_iterations (int): Total number of iterations
    Note:
        Updates ent_coef value based on training progresss
    '''
    progress = float(iteration) / float(num_iterations)
    current_ent_coef = initial_and_final_ent_coef[0] * (1 - progress) + initial_and_final_ent_coef[1] * progress
    return current_ent_coef

def collect_rollout_step(agent, pre_processor, envs, next_obs, next_done, meta_ep_traj_counter, train_env_list, step, storage_tensors, device, config):
    '''Execute one step of experience collection across all environments.
    
    This function:
    1. Gets actions from policy
    2. Executes actions in environments
    3. Processes rewards using DREST reward computation
    4. Handles environment resets and meta-episode management
    5. Stores experiences in storage tensors
    
    Args:
        agent: PPO agent model
        envs: Vectorized environment
        next_obs: Current observation tensor
        next_done: Current done flags tensor
        meta_ep_traj_counter: Counter tracking trajectory types per environment
        train_env_list: List of available training environments
        step (int): Current step within rollout
        storage_tensors (dict): Storage for experiences
        device: PyTorch device
        config (dict): Configuration parameters
        
    Returns:
        tuple:
            - next_obs: Processed next observation
            - next_done: Updated done flags
            - env_steps (int): Number of environment steps (1)
            - total_steps (int): Total steps across all envs (num_envs)
            
    Note:
        Handles both short and long trajectory rewards using DREST reward shaping
    '''
    storage_tensors['obs'][step] = next_obs
    storage_tensors['dones'][step] = next_done

    # ALGO LOGIC: action logic
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        storage_tensors['values'][step] = value.flatten()
    storage_tensors['actions'][step] = action
    storage_tensors['logprobs'][step] = logprob

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, terminations, _, infos = envs.step(action.cpu().numpy())
    next_done = terminations

    # Handle mini and meta episode terminations
    for idx in range(config['num_envs']):
        if terminations[idx]:
            # Compute DREST rewards
            if infos['short'][idx]:
                meta_ep_traj_counter[idx][0] += 1
                assert reward[idx] <= envs.envs[idx].max_coins[0], f'Short Traj returned rew for env {idx} = {reward[idx]} > max_coins: {envs.envs[idx].max_coins}\n\n{envs.envs[idx].initial_state}'
                excess_counts = np.array(meta_ep_traj_counter[idx]) - np.mean(meta_ep_traj_counter[idx])
                DREST_REWARD = (config['DREST_lambda_factor'] ** (excess_counts[0]))*(reward[idx]/envs.envs[idx].max_coins[0])
                reward[idx] = DREST_REWARD
            else:
                meta_ep_traj_counter[idx][1] += 1
                assert reward[idx] <= envs.envs[idx].max_coins[1], f'Short Traj returned rew for env {idx} = {reward[idx]} > max_coins: {envs.envs[idx].max_coins}\n\n{envs.envs[idx].initial_state}'
                excess_counts = np.array(meta_ep_traj_counter[idx]) - np.mean(meta_ep_traj_counter[idx])
                DREST_REWARD = (config['DREST_lambda_factor'] ** (excess_counts[1]))*(reward[idx]/envs.envs[idx].max_coins[1])
                reward[idx] = DREST_REWARD

            # If meta episode complete switch out that env for new one
            if sum(meta_ep_traj_counter[idx]) > config['meta_ep_size']: 
                new_idx = random.sample(range(len(train_env_list)), k=1)[0]
                envs.envs[idx].close()
                envs.envs[idx] = train_env_list[new_idx]
                meta_ep_traj_counter[idx] = [0,0]

                single_env_obs, _ = envs.envs[idx].reset()
                next_obs[idx] = single_env_obs

    storage_tensors['rewards'][step] = torch.tensor(reward).to(device).view(-1)

    return process_observation(config['CNN_preprocessing'], config['num_envs'], next_obs, pre_processor, device), torch.Tensor(next_done).to(device), 1, config['num_envs']     

def compute_advantages(next_obs, next_done, storage_tensors, agent, device, gamma, gae_lambda, num_steps):
    '''Compute generalized advantage estimation (GAE) and returns.
    
    Calculates advantages using the GAE method, which provides a good balance between
    bias and variance in advantage estimation. Also computes returns for value function training.
    
    Args:
        next_obs: Next observation tensor
        next_done: Next done flags tensor
        storage_tensors (dict): Experience storage containing rewards, values, and dones
        agent: PPO agent model
        device: PyTorch device
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        num_steps (int): Number of steps in rollout
        
    Returns:
        tuple: Contains:
            - advantages (torch.Tensor): Computed advantage estimates
            - returns (torch.Tensor): Computed returns for value function training
            
    Note:
        TODO: Needs adaptation to handle non-terminal environment bootstrapping
    '''
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(storage_tensors['rewards']).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - storage_tensors['dones'][t + 1]
                nextvalues = storage_tensors['values'][t + 1]
            delta = storage_tensors['rewards'][t] + gamma * nextvalues * nextnonterminal - storage_tensors['values'][t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + storage_tensors['values']

    return (advantages, returns)

def prepare_batch_data(storage_tensors, advantages, returns, envs, CNN_preprocessing, batch_size):
    '''Prepare and flatten collected experiences for PPO updates.
    
    Reshapes collected experiences into flat tensors suitable for minibatch training,
    handling both CNN and standard preprocessing cases.
    
    Args:
        storage_tensors (dict): Collected experiences
        advantages (torch.Tensor): Computed advantage estimates
        returns (torch.Tensor): Computed returns
        envs: Vector environment object
        CNN_preprocessing (bool): Whether CNN preprocessing is enabled
        batch_size (int): Size of complete training batch
        
    Returns:
        dict: Flattened batch data containing:
            - b_obs: Flattened observations
            - b_logprobs: Flattened log probabilities
            - b_actions: Flattened actions
            - b_advantages: Flattened advantages
            - b_returns: Flattened returns
            - b_values: Flattened values
            - b_inds: Batch indices for minibatch sampling
    '''
    if CNN_preprocessing:
        b_obs = storage_tensors['obs'].reshape((-1, 128))
    else:
        b_obs = storage_tensors['obs'].reshape((-1,) + envs.single_observation_space.shape)
    
    b_logprobs = storage_tensors['logprobs'].reshape(-1)
    b_actions = storage_tensors['actions'].reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = storage_tensors['values'].reshape(-1)
    b_inds = np.arange(batch_size)

    return {'b_obs': b_obs,
            'b_logprobs': b_logprobs,
            'b_actions': b_actions,
            'b_advantages': b_advantages,
            'b_returns': b_returns,
            'b_values': b_values,
            'b_inds': b_inds}

def update_policy(agent, optimizer, batch_data, clip_coef, vf_coef, initial_and_final_ent_coef, max_grad_norm, target_kl, update_epochs, norm_adv, clip_vloss, batch_size, minibatch_size, iteration, num_iterations):
    '''Update policy and value networks using PPO algorithm.
    
    Implements the PPO algorithm with:
    - Policy gradient clipping
    - Value function clipping (optional)
    - Early stopping based on KL divergence
    - Advantage normalization (optional)
    - Entropy bonus
    
    Args:
        agent: PPO agent model
        optimizer: PyTorch optimizer
        batch_data (dict): Prepared batch data for training
        clip_coef (float): PPO clipping coefficient
        vf_coef (float): Value function loss coefficient
        ent_coef (float): Entropy bonus coefficient
        max_grad_norm (float): Maximum gradient norm for clipping
        target_kl (float): Target KL divergence for early stopping
        update_epochs (int): Number of epochs to update over each batch
        norm_adv (bool): Whether to normalize advantages
        clip_vloss (bool): Whether to clip value loss
        batch_size (int): Total batch size
        minibatch_size (int): Size of minibatches
        
    Returns:
        dict: Training metrics including:
            - v_loss: Value function loss
            - pg_loss: Policy gradient loss
            - entropy_loss: Entropy loss
            - old_approx_kl: Old KL divergence approximation
            - approx_kl: New KL divergence approximation
            - clipfracs: Fraction of clipped policy updates
            - explained_var: Explained variance of value function
    '''
    clipfracs = []
    for epoch in range(update_epochs):
            np.random.shuffle(batch_data['b_inds'])
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = batch_data['b_inds'][start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(batch_data['b_obs'][mb_inds], batch_data['b_actions'].long()[mb_inds])
                logratio = newlogprob - batch_data['b_logprobs'][mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = batch_data['b_advantages'][mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - batch_data['b_returns'][mb_inds]) ** 2
                    v_clipped = batch_data['b_values'][mb_inds] + torch.clamp(
                        newvalue - batch_data['b_values'][mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - batch_data['b_returns'][mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - batch_data['b_returns'][mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - anneal_entropy_coef(iteration, initial_and_final_ent_coef, num_iterations) * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

    y_pred, y_true = batch_data['b_values'].cpu().numpy(), batch_data['b_returns'].cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {'v_loss': v_loss, 
            'pg_loss': pg_loss, 
            'entropy_loss': entropy_loss, 
            'old_approx_kl': old_approx_kl, 
            'approx_kl': approx_kl,
            'clipfracs': clipfracs, 
            'explained_var': explained_var}

def log_metrics(iteration, agent, optimizer, metrics, train_env_list, loop_parameters, config):
    '''Log training metrics and evaluate agent performance periodically.
    
    Handles two types of logging:
    1. Regular metrics logging every iteration (losses, KL divergence, etc.)
    2. Periodic evaluation of agent performance on training environments
    
    Args:
        iteration (int): Current training iteration
        agent: PPO agent model
        optimizer: PyTorch optimizer
        metrics (dict): Training metrics from update step
        train_env_list (list): List of training environments
        total_steps (int): Total steps taken across all environments
        start_time (float): Training start time
        cur_best_avr_usefulness (int): Current best usefulness score
        cur_best_avr_neutrality (int): Current best neutrality score
        env_steps (int): Environment steps taken
        config (dict): Configuration parameters
        
    Effects:
        - Logs metrics to wandb if tracking is enabled
        - Saves model checkpoints when new best scores are achieved
        - Updates current best scores
        
    Note:
        Will skip evaluation of environments that timeout after 3 seconds
    '''
    if config['track']:
        wandb.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": metrics['v_loss'].item(),
            "losses/policy_loss": metrics['pg_loss'].item(),
            "losses/entropy": metrics['entropy_loss'].item(),
            "losses/old_approx_kl": metrics['old_approx_kl'].item(),
            "losses/approx_kl": metrics['approx_kl'].item(), 
            "losses/clipfrac": np.mean(metrics['clipfracs']), 
            "losses/explained_variance": metrics['explained_var'], 
            'charts/env_steps': loop_parameters['env_steps'],
            'charts/steps_per_second': int(loop_parameters['total_steps'] / (time.time() - loop_parameters['start_time'])),
            }, step=loop_parameters['total_steps'])
        
        # Collect data on Usefulness/Neutrality scores for either test or train envs, 'test_log_freq' times per training run
        if iteration % int(config['num_iterations']/config['test_log_freq']) == 0:
            agent.eval()
            train_usefulness_list = []
            train_neutrality_list = []
            train_traj_short_list =[]
            train_traj_long_list = []       
            train_eval_envs = random.choices(range(len(train_env_list)), k=int(config['test_log_env_subset_size']*len(train_env_list)))
            
            with torch.no_grad():
                for train_env_index in train_eval_envs:
                    env = train_env_list[train_env_index]
                    env.reset()
                    try:
                        train_traj_ratio, useful, neutral = evaluate_agent(
                        env,
                        model=agent,
                        max_coins_by_trajectory=np.array([
                            env.max_coins[1],  # longer trajectory
                            env.max_coins[0]   # shorter trajectory
                        ]))
                        if isnan(neutral):
                            neutral = 0
                        train_usefulness_list.append(useful)
                        train_neutrality_list.append(neutral)
                        train_traj_short_list.append(train_traj_ratio[1])
                        train_traj_long_list.append(train_traj_ratio[0])
                    except timeout_decorator.TimeoutError:
                        print(f"Skipping evaluation of train env: {train_env_index} - timed out after 4 seconds")
                        continue
            
            usefulness = np.mean(train_usefulness_list)
            neutrality = np.mean(train_neutrality_list)
            mean_short_traj = np.mean(train_traj_short_list)
            mean_long_traj = np.mean(train_traj_long_list)

            if (usefulness + neutrality)/2 > (loop_parameters['cur_best_avr_usefulness'] + loop_parameters['cur_best_avr_neutrality'])/2:
                artifact = wandb.Artifact(f"{config['model_save_name']}U{int(round(usefulness,2)*100)}N{int(round(neutrality,2)*100)}", type='model')
                torch.save(agent.state_dict(), f"src/models/{config['model_save_name']}U{int(round(usefulness,2)*100)}N{int(round(neutrality,2)*100)}.pt")
                artifact.add_file(f"src/models/{config['model_save_name']}U{int(round(usefulness,2)*100)}N{int(round(neutrality,2)*100)}.pt")
                wandb.log_artifact(artifact)
                loop_parameters.update(cur_best_avr_usefulness=usefulness)
                loop_parameters.update(cur_best_avr_neutrality=neutrality)

            wandb.log({
                'train_metrics/Usefulness': usefulness,
                'train_metrics/Neutrality': neutrality,
                'train_metrics/Trajectory_Ratio': normalize_ratio_with_exp(mean_short_traj, mean_long_traj),
                }, step=loop_parameters['total_steps'])

            agent.train()
    
