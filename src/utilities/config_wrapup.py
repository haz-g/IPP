import timeout_decorator
import torch
import plotly.figure_factory as ff
import numpy as np
import wandb
from src.utilities.evals_utils import evaluate_agent
from math import isnan

def wrapup_training(agent, test_env_list, train_env_list, envs, config):
    '''Perform final evaluation and cleanup after training completion.
    
    This function:
    1. Evaluates the trained agent on all test environments
    2. Computes and logs final performance metrics
    3. Creates a summary table of results
    4. Saves the final model state
    5. Closes environments and wandb logging
    
    Args:
        agent: Trained PPO agent model
        test_env_list (list): List of test environments for final evaluation
        envs: Vector environment object to close
        config (dict): Configuration dictionary containing:
            - track (bool): Whether to log to wandb
            - model_save_name (str): Base name for saving model
            
    Effects:
        - Logs final test metrics to wandb as a table
        - Saves final model state as a wandb artifact
        - Closes all environments
        - Ends wandb run if tracking was enabled
        
    Note:
        - Skips evaluation of environments that timeout after 3 seconds
        - The longer trajectory metrics are stored in index 0 of max_coins_by_trajectory,
          while shorter trajectory metrics are in index 1
    '''
    print('\n\nAGENT TRAINING COMPLETE!\n\nNow commencing final eval on test envs...\n')
    
    agent.eval()
    
    avr_train_usefulness, avr_train_neutrality, avr_train_short_traj, avr_train_long_traj = evaluate_on_train_envs(agent, train_env_list, verbose=False)
    avr_test_usefulness, avr_test_neutrality, avr_test_short_traj, avr_test_long_traj = evaluate_on_test_envs(agent, test_env_list, verbose=False)
    
    data_matrix = [['Metric', 'Train Env Scores', 'Test Env Scores'],
            ['Usefulness', avr_train_usefulness, avr_test_usefulness],
            ['Neutrality', avr_train_neutrality, avr_test_neutrality],
            ['% Short Trajectory', avr_train_short_traj, avr_test_short_traj],
            ['% Lng Trajectory', avr_train_long_traj, avr_test_long_traj]]  
     
    fig = ff.create_table(data_matrix)

    wandb.log({'final_metrics': fig})

    envs.close()

    if config['track']:
        artifact = wandb.Artifact(f"{config['model_save_name']}FIN", type='model')
        torch.save(agent.state_dict(), f"src/models/{config['model_save_name']}FIN.pt")
        artifact.add_file(f"src/models/{config['model_save_name']}FIN.pt")
        wandb.log_artifact(artifact)
        wandb.finish()

def evaluate_on_test_envs(agent, test_env_list, verbose=True):
    test_usefulness_list = []
    test_neutrality_list = []
    test_traj_short_list = []
    test_traj_long_list = []

    with torch.no_grad():
        for test_env in range(len(test_env_list)):
            env = test_env_list[test_env]
            env.reset()

            try:
                test_traj_ratio, useful, neutral = evaluate_agent(env, agent, np.array([env.max_coins[1],  env.max_coins[0]]))
                if isnan(neutral):
                    neutral = 0
                if isnan(useful):
                    useful = 0

                test_usefulness_list.append(useful)
                test_neutrality_list.append(neutral)
                test_traj_short_list.append(test_traj_ratio[1])
                test_traj_long_list.append(test_traj_ratio[0])
                
                if verbose:
                    print(f'Test Env {test_env}: USEFULNESS: {round(useful,2)}, NEUTRALITY:{round(neutral,2)}, Short/Long: [{round(test_traj_ratio[1],2)}, {round(test_traj_ratio[0],2)}]')

            except timeout_decorator.TimeoutError:

                if verbose:
                    print(f"Skipping evaluation of test env {test_env} - timed out after 4 seconds")

                continue

    return round(np.mean(test_usefulness_list),2), round(np.mean(test_neutrality_list),2), round(np.mean(test_traj_short_list)*100,2), round(np.mean(test_traj_long_list)*100,2)

def evaluate_on_train_envs(agent, train_env_list, verbose=True):
    train_usefulness_list = []
    train_neutrality_list = []
    train_traj_short_list = []
    train_traj_long_list = []

    with torch.no_grad():
        for train_env in range(len(train_env_list)):
            env = train_env_list[train_env]
            env.reset()

            try:
                train_traj_ratio, useful, neutral = evaluate_agent(env, agent, np.array([train_env.max_coins[1], train_env.max_coins[0]]))
                
                if isnan(neutral):
                    neutral = 0
                if isnan(useful):
                    useful = 0

                train_usefulness_list.append(useful)
                train_neutrality_list.append(neutral)
                train_traj_short_list.append(train_traj_ratio[1])
                train_traj_long_list.append(train_traj_ratio[0])

                if verbose:
                    print(f'Train Env {train_env}: USEFULNESS: {round(useful,2)}, NEUTRALITY:{round(neutral,2)}, Short/Long: [{round(train_traj_ratio[1],2)}, {round(train_traj_ratio[0],2)}]')
            
            except timeout_decorator.TimeoutError:

                if verbose:
                    print(f"Skipping evaluation of train env {train_env} - timed out after 4 seconds")
                
                continue
    
    return round(np.mean(train_usefulness_list),2), round(np.mean(train_neutrality_list),2), round(np.mean(train_traj_short_list)*100,2), round(np.mean(train_traj_long_list)*100,2)