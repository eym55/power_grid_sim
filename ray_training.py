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
from agents import RandomAgent,HurricaneAgent
from scipy.special import comb

ray.init()
network = pypsa.Network('lopf_grid.nc')
hurricane_path = [(0,0),(1,.5),(2,1),(3,1.5),(4,2),(4.5,2.7),(5,3),(6,3),(7,4),(8,4.5)]

agent_config = {
  'hurricane_path':hurricane_path
}

env_config = {
  'network':network,
  'agent_config':agent_config,
  'agent_class':HurricaneAgent,
  'lines_per_turn':1
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
for i in range(500):
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

