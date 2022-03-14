import gym
from game_env import PowerGrid
import pypsa
import numpy as np

network = pypsa.Network('lopf_grid.nc')
attack_distribution = np.random.dirichlet(np.ones(network.lines.shape[0]),size= 1)
env = PowerGrid(network,attack_distribution)
obs = env.reset()
print(obs)
for i in range(10):
    action = 3
    obs, rewards, done, info = env.step(action)
    env.render()