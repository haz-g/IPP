import torch.optim as optim
from config import CONFIG
from src.Generalist.PPO2 import Agent
from src.utilities.config_device import setup_wandb, seed_run, define_device, parse_args
from src.utilities.config_env import load_envs, init_vec_envs
from src.utilities.config_train_loop import (init_pre_processor, setup_storage, process_observation, setup_loop_params, anneal_learning_rate, 
                                             collect_rollout_step, compute_advantages, prepare_batch_data, update_policy,
                                             log_metrics)
from src.utilities.config_wrapup import wrapup_training

if __name__ == "__main__":

    # ----- DEVICE CONFIG -----

    parse_args(CONFIG)
    setup_wandb(CONFIG)
    seed_run(CONFIG['seed'], CONFIG['torch_deterministic'])
    device = define_device(CONFIG['cuda'])

    # ----- LOAD & INSTANTIATE TRAIN ENVIRONMENTS -----

    train_env_list, test_env_list = load_envs(CONFIG['train_env_list'], CONFIG['test_env_list'])
    envs = init_vec_envs(train_env_list)
    
    # ----- LOAD & INSTANTIATE PPO AGENT -----

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG['learning_rate'], eps=1e-5)
    pre_processor = init_pre_processor(CONFIG['CNN_preprocessing'], device)

    # ----- TRAINING LOOP -----

    # Setup Storage
    storage_tensors = setup_storage(CONFIG['CNN_preprocessing'], CONFIG['num_steps'], CONFIG['num_envs'], device, envs)

    # Loop parameters (TRY NOT TO MODIFY)
    loop_parameters = setup_loop_params(device, CONFIG['num_envs'])
    
    # Collect and process first observation
    next_obs, _ = envs.reset(seed=CONFIG['seed'])
    next_obs = process_observation(CONFIG['CNN_preprocessing'], CONFIG['num_envs'], next_obs, pre_processor, device)

    # Main training loop: iteration count defined as (total timesteps)/(batch size)
    for iteration in range(1, CONFIG['num_iterations'] + 1):
        #Decays LR if specified in CONFIG
        anneal_learning_rate(optimizer, iteration, CONFIG['learning_rate'], CONFIG['decay_steps'], CONFIG['decay_rate'], CONFIG['anneal_lr'])

        # ----- POLICY ROLLOUT: COLLECT N EXPERIENCES -----

        for step in range(0, CONFIG['num_steps']):
            
            # Get action, execute a step in envs and store experience
            next_obs, next_done, step_env_steps, step_total_steps = collect_rollout_step(agent, pre_processor, envs, next_obs, loop_parameters['next_done'], loop_parameters['meta_ep_traj_counter'], 
                                                                                         train_env_list, step, storage_tensors, device, CONFIG)
            loop_parameters['next_done'] = next_done
            loop_parameters['env_steps'] += step_env_steps
            loop_parameters['total_steps'] += step_total_steps           

        # ----- UPDATE POLICY -----
        
        # Compute advantages and returns from rollout
        advantages, returns = compute_advantages(next_obs, loop_parameters['next_done'], storage_tensors, agent, device, 
                                                 CONFIG['gamma'],CONFIG['gae_lambda'], CONFIG['num_steps'])

        # Prepare batched data for NN updates
        batch_data = prepare_batch_data(storage_tensors, advantages, returns, envs, CONFIG['CNN_preprocessing'], CONFIG['batch_size'])

        # Update policy
        metrics = update_policy(agent, optimizer, batch_data, CONFIG['clip_coef'], CONFIG['vf_coef'], CONFIG['ent_coef'], 
                                CONFIG['max_grad_norm'], CONFIG['target_kl'], CONFIG['update_epochs'], CONFIG['norm_adv'],
                                CONFIG['clip_vloss'], CONFIG['batch_size'], CONFIG['minibatch_size'])

        # ----- LOG METRICS -----
        log_metrics(iteration, agent, optimizer, metrics, train_env_list, loop_parameters, CONFIG)

    # ----- RUN FINAL EVAL AND LOG FINAL POLICY -----
    wrapup_training(agent, test_env_list, envs, CONFIG)