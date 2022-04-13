import gym
from cyber_attacker_game import PowerGrid
import pypsa
import numpy as np
from random import sample

network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]

env_one_per = PowerGrid(network,attack_distribution, 1)
env_two_per = PowerGrid(network,attack_distribution, 2)
env_3_per = PowerGrid(network,attack_distribution, 3)  
results_length= []

""" results_rewards=[] 
for episode in range(10):
  obs = env_one_per.reset()
  total_reward = 0
  for i in range(10):
    env_one_per.render()
    action = np.random.choice(range(LINES)) 
    print("action is", action) 
    obs, rewards, done, info = env_one_per.step(action)
    total_reward += rewards
    if done==True:
      break
  print(f"Agent made it {i+1} timesteps and had a total reward of {total_reward}")
  results_length.append(i)
  results_rewards.append(total_reward)
print(results_length,np.mean(results_length))
print(results_rewards,np.mean(results_rewards)) """

results_rewards=[] 
for episode in range(10):
  obs = env_two_per.reset()
  total_reward = 0
  for i in range(10):
    env_one_per.render()
    action = env_two_per.permutations[np.random.choice(len(env_two_per.permutations))]
    obs, rewards, done, info = env_two_per.step(action)
    total_reward += rewards
    if done==True:
      break
  print(f"Agent made it {i+1} timesteps and had a total reward of {total_reward}")
  results_length.append(i)
  results_rewards.append(total_reward)
print(results_length,np.mean(results_length))
print(results_rewards,np.mean(results_rewards))

