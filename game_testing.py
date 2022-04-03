import gym
from game_env import PowerGrid
import pypsa
import numpy as np

network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]
env = PowerGrid(network,attack_distribution)
results_length= []
results_rewards=[]
for episode in range(2):
  obs = env.reset()
  total_reward = 0
  for i in range(10):
    env.render()
    action = np.random.choice(range(LINES))
    obs, rewards, done, info = env.step(action)
    total_reward += rewards
    env.render()
    if done==True:
      break
  print(f"Agent made it {i+1} timesteps and had a total reward of {total_reward}")
  results_length.append(i)
  results_rewards.append(total_reward)
print(results_length,np.mean(results_length))
print(results_rewards,np.mean(results_rewards))

