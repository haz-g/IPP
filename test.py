import numpy as np
import pickle 
import torch
import pandas as pd
from src.Generalist.PPO2 import Agent 
from config import CONFIG
import plotly.figure_factory as ff
import sys
from src.utilities.config_wrapup import evaluate_on_test_envs, evaluate_on_train_envs

sys.path.append('src')
with open(CONFIG['train_env_list'], "rb") as f:
    train_env_list = pickle.load(f)
with open(CONFIG['test_env_list'], "rb") as t:
    test_env_list = pickle.load(t)

# Fill this with the relevant run id and models you'd like to evaluate from your /models dir
model_dict = {"Run ID": ["model1.pt", "model2.pt", "model3.pt"]}

for exp_id, model_list in model_dict.items():
    print(f'\nExperiment {exp_id}\n')

    for model in model_list:
        model_path = f"src/models/{model}"
        
        try:
            agent = Agent(test_env_list)
            agent.load_state_dict(torch.load(model_path))
            agent.eval()

            avr_train_usefulness, avr_train_neutrality, avr_train_short_traj, avr_train_long_traj = evaluate_on_train_envs(agent, train_env_list, verbose=False)
            avr_test_usefulness, avr_test_neutrality, avr_test_short_traj, avr_test_long_traj = evaluate_on_test_envs(agent, test_env_list, verbose=False)

            print(f'Evaluated Model {model} - Train: U {avr_train_usefulness} | N {avr_train_neutrality} | [{avr_train_short_traj},{avr_train_long_traj}] | Test: U {avr_test_usefulness} | N {avr_test_neutrality} | [{avr_test_short_traj},{avr_test_long_traj}]')
        
        except Exception as e:
            print(f"Error evaluating model {model}: {e}")
            continue