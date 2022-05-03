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

ray.init()
network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]

agent_config = {
  'action_distribution':attack_distribution
}

env_config = {
  'network':network,
  'agent_config':agent_config,
  'agent_class':RandomAgent
  }

agent = dqn.DQNTrainer(env=PowerGrid, config={
    "env_config": env_config, 
    "num_workers": 8,
    "n_step": 5,
    "noisy": True,
    "num_atoms": 5,
    "v_min": 0,
    "v_max": 1000.0,
})

#Change the range to desired amount of training iterations
for i in range(300):
  try:
    pop = agent.train()
    print(pretty_print(pop))
    time.sleep(5)
    if i % 10 == 0:
       checkpoint = agent.save()
       print("checkpoint saved at", checkpoint)
  except Exception as e:
    print(e)
    print("FUCK")
    print(i)

