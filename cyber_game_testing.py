from pyexpat import model
import gym
from cyber_attacker_game import PowerGrid
import pypsa
import numpy as np
from random import sample
import sys

modelType = sys.argv[1] # takes either '1line' , '2line' or '3line' as arg

network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0] 

results_length= []
results_rewards=[] 

def oneLine(): 
  env_one_per = PowerGrid(network,attack_distribution, 1)

  for episode in range(10):
    obs = env_one_per.reset()
    total_reward = 0
    for i in range(10):
      env_one_per.render()
      action = np.random.choice(range(LINES)) 
      while action in env_one_per.removed_lines:
        action = action = np.random.choice(range(LINES)) 
      obs, rewards, done, info = env_one_per.step(action)
      total_reward += rewards
      if done==True:
        break
    print(f"Agent made it {i+1} timesteps and had a total reward of {total_reward}")
    results_length.append(i)
    results_rewards.append(total_reward)
  print(results_length,np.mean(results_length))
  print(results_rewards,np.mean(results_rewards))

def twoLine():
  env_two_per = PowerGrid(network,attack_distribution, 2)

  results_rewards=[] 
  for episode in range(10):
    obs = env_two_per.reset()
    total_reward = 0
    for i in range(10):
      env_two_per.render()
      action = np.random.choice(env_two_per.NUM_LINES, size=2)
      while (action[0] == action[1] or action[0] in env_two_per.removed_lines or action[1] in env_two_per.removed_lines): 
        action = np.random.choice(env_two_per.NUM_LINES, size=2)
        print('got 2 of the same, new attacker action is', action)
      obs, rewards, done, info = env_two_per.step(tuple(action))
      total_reward += rewards
      if done==True:
        break
    print(f"Agent made it {int(i)+1} timesteps and had a total reward of {str(total_reward)}")
    results_length.append(int(i))
    results_rewards.append(int(total_reward))
  print(results_length,np.mean(results_length))
  print(results_rewards,np.mean(results_rewards))

def threeLine():
  env_3_per = PowerGrid(network,attack_distribution, 3)  
  results_rewards=[] 

  for episode in range(10):
    obs = env_3_per.reset()
    total_reward = 0
    for i in range(10):
      env_3_per.render()
      action = np.random.choice(env_3_per.NUM_LINES, size=3)
      #print(action)
      while env_3_per.any_duplicates(action) or action[0] in env_3_per.removed_lines or action[1] in env_3_per.removed_lines or action[2] in env_3_per.removed_lines:
        action = np.random.choice(env_3_per.NUM_LINES, size=3)
        print('got 2+ of the same or some already removed lines, new attacker action is', action)
      obs, rewards, done, info = env_3_per.step(tuple(action))
      total_reward += rewards
      if done==True:
        break
    print(f"Agent made it {int(i)+1} timesteps and had a total reward of {str(total_reward)}")
    results_length.append(int(i))
    results_rewards.append(int(total_reward))
  print(results_length,np.mean(results_length))
  print(results_rewards,np.mean(results_rewards))

if modelType == '1line':
  oneLine()
elif modelType == '2line':
  twoLine()
else:
  threeLine()