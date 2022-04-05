import gym
from defender_game import PowerGrid
import pypsa
import numpy as np

network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]
defend_distribution = np.random.dirichlet(np.ones(LINES),size= 1)[0]

aEnv = PowerGrid(network,attack_distribution)
dEnv = PowerGrid(network,defend_distribution)

results_length= []
results_rewards=[]

for episode in range(5):
  obs = aEnv.reset()
  dtrewards = 0
  atrewards = 0
  for i in range(10):
    dEnv.render()
    action = np.random.choice(range(LINES))
    obs, rewards, done, info = aEnv.step(action)
    dtrewards += rewards
    
    aEnv.render()
    action = np.random.choice(range(LINES))
    obs, rewards, done, info = aEnv.step(action)
    atrewards += rewards

    if done==True:
      break

  print(f"Agent made it {i+1} timesteps and had a total reward of {dtrewards}")
  results_length.append(i)
  results_rewards.append(dtrewards)
print(results_length,np.mean(results_length))
print(results_rewards,np.mean(results_rewards))


