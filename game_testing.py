import gym
from defender_game import PowerGrid
import pypsa
import numpy as np
from agents import RandomAgent,DQNAgent

network = pypsa.Network('lopf_grid.nc')
defender = RandomAgent()
agent_config = {
  'attack_distribution = '
}
env_config = {
  'network':network,
  'agent_config':attack_distribution}
env = PowerGrid()
results_length= []
results_rewards=[]
for episode in range(5):
  obs = env.reset()
  total_reward = 0
  done = False
  i=0
  while done == False:
    env.render()
    action = np.random.choice(range(LINES))
    obs, rewards, done, info = env.step(action)
    i+=1
    total_reward += rewards
  results_length.append(i)
  results_rewards.append(total_reward)
  print(f"\n\n\n Episode {episode} done \n\n\n")

import time
time.sleep(15)
print('\n\n\n\n\nDone')
print(results_length,np.mean(results_length))
print(results_rewards,np.mean(results_rewards))



