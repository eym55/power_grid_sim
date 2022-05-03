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
ray.init()
network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]
env = PowerGrid({'network':network,'attack_distribution':attack_distribution})
#Change dqn to the desired algorithm
#Change any config variables other than env_config
agent = dqn.DQNTrainer(env=PowerGrid, config={
    "env_config": {'network':network,'attack_distribution':attack_distribution},
    "num_workers": 8,
    "n_step": 5,
    "noisy": True,
    "num_atoms": 2,
    "v_min": 0,
    "v_max": 1000.0,
})

agent.load_checkpoint('results/checkpoint_000251/checkpoint-251')
initial_state = env.reset()
actions = range(LINES)
policy = agent.get_policy()
action_distribution = []
for i in range(10000):
  action_distribution.append(agent.compute_action(initial_state))
# action_distribution = policy.compute_log_likelihoods(actions = actions, obs_batch = initial_state)
print(action_distribution)