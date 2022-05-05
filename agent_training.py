from cProfile import run
import gym, ray
#Import any agents you want to train, list found here: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
from ray.rllib.agents import ppo,dqn,a3c
from environments.defender_game_v3 import PowerGrid
import pypsa
import numpy as np
from ray.tune.registry import register_env
import pickle
import time
import argparse
from ray.tune.logger import pretty_print
from agents import RandomAgent
from scipy.special import comb
import os
import logging

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=False, help="Name of this test",default = '')
args = vars(ap.parse_args())
test_name = args['name']

#Initializing relevant variables
ray.init()
networks = ['networks/sample_grid.nc','networks/30line.nc','networks/60line.nc']
# lines_per_turn_tests = [1,3,5,7]
lines_per_turn_tests = [1]
test_time = time.strftime("%Y-%m-%d_%H-%M-%S")
test_dir = f'./results/MAS_training/{test_name}{test_time}/'
os.makedirs(test_dir,exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f'{test_dir}logs.log'),
        logging.StreamHandler()
    ]
)

#Runs our agent in the environment for specified number of iterations
def run_training(output_dir,agent,experiment_name,epochs = 500):
  consecutive_errors = 0
  total_errors = 0
  for i in range(epochs):
    try:
      logging.info(f"Experiment: {experiment_name}, Epoch: {i}")
      epoch_result = agent.train()
      logging.info(pretty_print(epoch_result))
      time.sleep(5)
      if i % 10 == 0:
        checkpoint = agent.save(f'{output_dir}checkpoint{i+1}')
        logging.info(f"checkpoint saved at: {checkpoint}")
      consecutive_errors = 0
    except Exception as e:
      total_errors +=1
      consecutive_errors +=1
      logging.error(e)
      logging.info(f'{total_errors} total errors and {consecutive_errors} consecutive errors')
      if consecutive_errors >=5:
        logging.error("5 Errors in a row. Moving on.")
        return



for network_path in networks:
  network = pypsa.Network(network_path)
  NUM_LINES = network.lines.shape[0]
  for lines_per_turn in lines_per_turn_tests:
    experiment_name = f'{NUM_LINES}lines_{lines_per_turn}per_turn'
    output_dir = f'{test_dir}{experiment_name}/'
    os.makedirs(output_dir,exist_ok=True)

    num_actions = comb(NUM_LINES,lines_per_turn,exact = True)

    #Use a uniform distribution
    attacker_distribution = np.ones(num_actions) / num_actions
    attacker_config = {
      'action_distribution':attacker_distribution
      }

    env_config = {
      'network':network,
      'agent_config':attacker_config,
      'agent_class':RandomAgent,
      'lines_per_turn':lines_per_turn
      }

    agent_config = {
      "env_config": env_config, 
      "num_workers": 8,
      "n_step": 5,
      "noisy": True,
      "num_atoms": 5,
      "v_min": 0,
      "v_max": 1000.0
      }

    agent = dqn.DQNTrainer(env=PowerGrid, config=agent_config)

    with open(f'{output_dir}/attacker_config.pkl', '+wb') as f:
      pickle.dump(attacker_config, f)
    with open(f'{output_dir}/env_config.pkl', '+wb') as f:
      pickle.dump(env_config, f)
    with open(f'{output_dir}/env.pkl', '+wb') as f:
      pickle.dump(PowerGrid, f) 
    with open(f'{output_dir}/agent_config.pkl', '+wb') as f:
      pickle.dump(agent_config, f)
    with open(f'{output_dir}/agent.pkl', '+wb') as f:
      pickle.dump(agent, f)
    
    logging.info(f"Beginning training for {experiment_name}")
    run_training(output_dir,agent,experiment_name = experiment_name)
    time.sleep(10)
    

