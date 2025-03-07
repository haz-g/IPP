import timeout_decorator
import torch
import plotly.figure_factory as ff
import numpy as np
import wandb
from src.utilities.evals_utils import evaluate_agent
from math import isnan

def wrapup_training(agent, test_env_list, envs, config):
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
    test_usefulness_list = []
    test_neutrality_list = []
    test_traj_short = []
    test_traj_long = []
    with torch.no_grad():
        for test_env in test_env_list:
                test_env.reset()
                try:
                    test_traj_ratio, useful, neutral = evaluate_agent(
                        env=test_env,
                        model=agent,
                        max_coins_by_trajectory=np.array([
                            test_env.max_coins[1],  # longer trajectory
                            test_env.max_coins[0]   # shorter trajectory
                        ])
                    )
                    if isnan(neutral):
                        neutral = 0
                    test_usefulness_list.append(useful)
                    test_neutrality_list.append(neutral)
                    test_traj_short.append(test_traj_ratio[1])
                    test_traj_long.append(test_traj_ratio[0])
                    print(f'[{round(test_traj_ratio[1],2)},{round(test_traj_ratio[0],2)}], USE:{round(useful,2)}, NEUT:{round(neutral,2)}')
                except timeout_decorator.TimeoutError:
                        print(f"Skipping evaluation of test env - timed out after 3 seconds")
                        continue
    
    data_matrix = [['Metric', 'Score'],
            ['Usefulness', round(np.mean(test_usefulness_list),2)],
            ['Neutrality', round(np.mean(test_neutrality_list),2)],
            ['% Short Trajectory', round(np.mean(test_traj_short)*100,2)],
            ['% Lng Trajectory', round(np.mean(test_traj_long)*100,2)]]   
    fig = ff.create_table(data_matrix)

    wandb.log({'test_env_metrics': fig})

    envs.close()

    if config['track']:
        artifact = wandb.Artifact(f"{config['model_save_name']}FIN", type='model')
        torch.save(agent.state_dict(), f"src/models/{config['model_save_name']}FIN.pt")
        artifact.add_file(f"src/models/{config['model_save_name']}FIN.pt")
        wandb.log_artifact(artifact)
        wandb.finish()