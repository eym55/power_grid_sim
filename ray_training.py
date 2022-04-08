import gym, ray
from ray.rllib.agents import ppo
from defender_game import PowerGrid
import pypsa
import numpy as np
from ray.tune.registry import register_env
import pickle
import resource
resource.getrlimit(resource.RLIMIT_NOFILE)


ray.init()
network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]
agent = ppo.PPOTrainer(env=PowerGrid, config={
    "env_config": {'network':network,'attack_distribution':attack_distribution}, 
    "num_workers": 0,
    "vf_clip_param": 100,
})
history = []
for _ in range(1):
  try:
    pop = agent.train()
    history.append(pop)
  except Exception as e:
    print(e)
    print("FUCK")
with open('history.pkl', 'wb') as f:
  pickle.dump(history, f)
