from mimetypes import init
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ray
from ray.rllib.agents import dqn
from defender_game import PowerGrid
import pypsa



np.random.seed(0)
network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]

env = PowerGrid({'network':network,'attack_distribution':attack_distribution})

ray.init()
agent = dqn.DQNTrainer(env=PowerGrid, config={
    "env_config": {'network':network,'attack_distribution':attack_distribution}, 
    "num_workers": 8,
    "n_step": 5,
    "noisy": True,
    "num_atoms": 5,
    "v_min": 0,
    "v_max": 10.0,
})

agent.restore('results/checkpoint_000251/checkpoint-251')
initial_state = env.reset()
actions = range(LINES)
action_distribution = agent.get_policy().compute_log_likelihoods(actions = actions, obs_batch = initial_state)
print(action_distribution)




