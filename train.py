import torch.optim as optim
from config import CONFIG
from src.Generalist.PPO2 import Agent
from src.utilities.config_device import setup_wandb, seed_run, define_device, parse_args, load_pretrained_model
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
    agent = load_pretrained_model(agent, CONFIG['load_model'], CONFIG['model_to_load'], device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG['learning_rate'], eps=1e-5)
    pre_processor = init_pre_processor(CONFIG['CNN_preprocessing'], device)

    # ----- TRAINING INITIALISATION -----

    storage_tensors = setup_storage(CONFIG['CNN_preprocessing'], CONFIG['num_steps'], CONFIG['num_envs'], device, envs)
    loop_parameters = setup_loop_params(device, CONFIG['num_envs'])
    next_obs, _ = envs.reset(seed=CONFIG['seed'])
    next_obs = process_observation(CONFIG['CNN_preprocessing'], CONFIG['num_envs'], next_obs, pre_processor, device)

    # ----- MAIN TRAINING LOOP ----- 

    # One iteration involves a collection of 'num_steps' experiences, an --UPDATE-- to the NN based on these and a --LOG-- of current NN performance
    for iteration in range(1, CONFIG['num_iterations'] + 1):
        
        # Handles decay of LR if specified in CONFIG
        anneal_learning_rate(optimizer, iteration, CONFIG['learning_rate'], CONFIG['num_iterations'], CONFIG['decay_rate'], CONFIG['anneal_lr'])

        # Rollout policy and collect 'num_steps' experiences
        for step in range(0, CONFIG['num_steps']):
            
            # Below, we get an action, execute a step in envs and store experiences to our 'storage_tensors' until they're full
            next_obs, next_done, step_env_steps, step_total_steps = collect_rollout_step(agent, pre_processor, envs, next_obs, loop_parameters['next_done'], loop_parameters['meta_ep_traj_counter'], 
                                                                                         train_env_list, step, storage_tensors, device, CONFIG)
            loop_parameters['next_done'] = next_done
            loop_parameters['env_steps'] += step_env_steps
            loop_parameters['total_steps'] += step_total_steps           

    # ----- UPDATE POLICY -----
        
        # Compute advantages and returns for this iteration's rollout
        advantages, returns = compute_advantages(next_obs, loop_parameters['next_done'], storage_tensors, agent, device, 
                                                 CONFIG['gamma'],CONFIG['gae_lambda'], CONFIG['num_steps'])

        # Prepare batched data for NN updates
        batch_data = prepare_batch_data(storage_tensors, advantages, returns, envs, CONFIG['CNN_preprocessing'], CONFIG['batch_size'])

        # Update policy
        metrics = update_policy(agent, optimizer, batch_data, CONFIG['clip_coef'], CONFIG['vf_coef'], CONFIG['initial_and_final_ent_coef'], 
                                CONFIG['max_grad_norm'], CONFIG['target_kl'], CONFIG['update_epochs'], CONFIG['norm_adv'],
                                CONFIG['clip_vloss'], CONFIG['batch_size'], CONFIG['minibatch_size'], iteration, CONFIG['num_iterations'])

    # ----- LOG METRICS -----

        # Evaluate this iteration's updated policy on a subset of training envs and store to wandb as artifact if performance is good
        log_metrics(iteration, agent, optimizer, metrics, train_env_list, loop_parameters, CONFIG)

    # ----- RUN FINAL EVAL AND LOG FINAL POLICY -----

    # Wrap up training by evaluating final policy on test envs and storing to wandb as artifact
    wrapup_training(agent, test_env_list, train_env_list, envs, CONFIG)