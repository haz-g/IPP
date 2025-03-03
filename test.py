# Created to test models downloaded directly from wandb and stored in src/models/

import numpy as np
import pickle 
import torch
from src.Generalist.PPO2 import Agent 
from src.utilities.evals_utils import evaluate_agent
from config import CONFIG
import plotly.figure_factory as ff
import timeout_decorator
import sys
from math import isnan

sys.path.append('src')
with open(CONFIG['train_env_list'], "rb") as f:
    train_env_list = pickle.load(f)
with open(CONFIG['test_env_list'], "rb") as t:
    test_env_list = pickle.load(t)

model = 'DREST_0228-1754_USE_35_NEUT_81'
model_path = f"src/models/{model}.pt"

agent = Agent(test_env_list)
agent.load_state_dict(torch.load(model_path))
agent.eval()

print('\n\nTEST ENVS\n\n')
with torch.no_grad():
    test_usefulness_list = []
    test_neutrality_list = []
    test_traj_short = []
    test_traj_long = []
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
    print('\n\nTRAIN ENVS\n\n')
    train_usefulness_list = []
    train_neutrality_list = []
    train_traj_short =[]
    train_traj_long = []
    for train_env in train_env_list:
            train_env.reset()
            try:
                train_traj_ratio, useful, neutral = evaluate_agent(
                    env=train_env,
                    model=agent,
                    max_coins_by_trajectory=np.array([
                        train_env.max_coins[1],  # longer trajectory
                        train_env.max_coins[0]   # shorter trajectory
                    ])
                )
                if isnan(neutral):
                    neutral = 0
                train_usefulness_list.append(useful)
                train_neutrality_list.append(neutral)
                train_traj_short.append(train_traj_ratio[1])
                train_traj_long.append(train_traj_ratio[0])
                print(f'[{round(train_traj_ratio[1],2)},{round(train_traj_ratio[0],2)}], USE:{round(useful,2)}, NEUT:{round(neutral,2)}')
            except timeout_decorator.TimeoutError:
                    print(f"Skipping evaluation of train env - timed out after 3 seconds")
                    continue
            
data_matrix = [['Metric', 'Train Env Score', 'Test Env Score'],
            ['Usefulness', round(np.mean(train_usefulness_list),2), round(np.mean(test_usefulness_list),2)],
            ['Neutrality', round(np.mean(train_neutrality_list),2), round(np.mean(test_neutrality_list),2)],
            ['Avr. % Short Trajectory', round(np.mean(train_traj_short)*100,2), round(np.mean(test_traj_short)*100,2)],
            ['Avr % Lng Trajectory', round(np.mean(train_traj_long)*100,2), round(np.mean(test_traj_long)*100,2)]]  
fig = ff.create_table(data_matrix)
fig.update_layout(title_text=f"{model}", title_x=0.5)
fig.show()