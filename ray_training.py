import gym, ray
from ray.rllib.agents import ppo
from defender_game import PowerGrid
import pypsa
import numpy as np
from ray.tune.registry import register_env


ray.init()
network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]
agent = ppo.PPOTrainer(env=PowerGrid, config={
    "env_config": {'network':network,'attack_distribution':attack_distribution}, 
    "evaluation_num_workers": 1,
})
for _ in range(1):
  results = agent.train()
print(results)
agent.evaluate()
