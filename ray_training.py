import gym, ray
from ray.rllib.agents import ppo,dqn
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
agent = dqn.DQNTrainer(env=PowerGrid, config={
    "env_config": {'network':network,'attack_distribution':attack_distribution}, 
    "num_workers": 2,
    "n_step": 5,
    "noisy": True,
    "num_atoms": 5,
    "v_min": 0,
    "v_max": 500.0
})
history = []
for i in range(5):
  try:
    pop = agent.train()
    history.append(pop)
  except Exception as e:
    print(e)
    print("FUCK")
    print(i)
with open('history.pkl', 'wb') as f:
  pickle.dump(history, f)
