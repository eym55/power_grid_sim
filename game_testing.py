import gym
from defender_game import PowerGrid
import pypsa
import numpy as np
from agents import RandomAgent,DQNAgent

network = pypsa.Network('lopf_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]

agent_config = {
  'action_distribution':attack_distribution
}

env_config = {
  'network':network,
  'agent_config':agent_config,
  'agent_class':RandomAgent}
env = PowerGrid(env_config)
defend_distribution = np.random.dirichlet(np.ones(LINES),size= 1)[0]
defend_config = {
  'checkpoint_path':'results/checkpoint_000251/checkpoint-251',
  'agent_config':{
    "env_config": {'network':network,'attack_distribution':attack_distribution},
    "num_workers": 8,
    "n_step": 5,
    "noisy": True,
    "num_atoms": 2,
    "v_min": 0,
    "v_max": 1000.0,
},

}
defender = DQNAgent(env,defend_config)

results_length= []
results_rewards=[]
for episode in range(5):
  total_reward = 0
  done = False
  i=0
  obs = env.reset()
  action = defender.compute_action(obs)
  while done == False:
    obs, rewards, done, info = env.step(action)
    action = defender.compute_action(obs)
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



