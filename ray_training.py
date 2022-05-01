import gym, ray
#Import any agents you want to train, list found here: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
from ray.rllib.agents import ppo,dqn,a3c
from defender_game import PowerGrid
import pypsa
import numpy as np
from ray.tune.registry import register_env
import pickle
import time
from ray.tune.logger import pretty_print
from agents import RandomAgent
from scipy.special import comb

ray.init()
network = pypsa.Network('lopf_grid.nc')
LINES = int(comb(network.lines.shape[0],2))
attack_distribution =  np.ones(LINES) / LINES

agent_config = {
  'action_distribution':attack_distribution
}

env_config = {
  'network':network,
  'agent_config':agent_config,
  'agent_class':RandomAgent,
  'lines_per_turn':2
  }

agent = dqn.DQNTrainer(env=PowerGrid, config={
    "env_config": env_config, 
    "num_workers": 1,
    "n_step": 5,
    "noisy": True,
    "num_atoms": 5,
    "v_min": 0,
    "v_max": 1000.0,
})

#Change the range to desired amount of training iterations
for i in range(1):
  # mean_rewards = []
  try:
    pop = agent.train()
    # mean_rewards.append()
    print(pretty_print(pop))
    time.sleep(5)
    if i % 10 == 0:
       checkpoint = agent.save()
       print("checkpoint saved at", checkpoint)
  except Exception as e:
    print(e)
    print("FUCK")
    print(i)

