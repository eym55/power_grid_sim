import gym
from environments.defender_game import PowerGrid
import pypsa
import numpy as np
from agents import RandomAgent,DQNAgent
from environments.defender_game_v1 import PowerGridV1
import pickle


np.random.seed(10)
network = pypsa.Network('networks/sample_grid.nc')
LINES = network.lines.shape[0]
attack_distribution =  np.random.dirichlet(np.ones(LINES),size= 1)[0]

#Make random attacker
attacker_agent_config = {
  'action_distribution':attack_distribution
}
#Make environment
env_config = {
  'network':network,
  'agent_config':attacker_agent_config,
  'agent_class':RandomAgent}
env = PowerGrid(env_config)

#Make defender
defend_config = {
  'checkpoint_path':'results/DQN_checkpoint_power_grid_v1/checkpoint-251',
  'agent_config':{
    "env_config": {'network':network,'attack_distribution':attack_distribution},
    "num_workers": 8,
    "n_step": 5,
    "noisy": True,
    "num_atoms": 2,
    "v_min": 0,
    "v_max": 1000.0,
    }
}
defender = DQNAgent(PowerGridV1,defend_config)


results_length= []
results_rewards=[]
for episode in range(1000):
  total_reward = 0
  done = False
  i=0
  obs = env.reset()
  action = 1
  while done == False:
    obs, rewards, done, info = env.step(action)
    action = defender.compute_action(obs)
    i+=1
    total_reward += rewards
  results_length.append(i)
  results_rewards.append(total_reward)
  print(f"\n\n\n Episode {episode} done \n\n\n")

print(f"\n\n\n heuristic Done \n\n\n")
with open('results/heuristic_rewards.pkl', 'wb') as f:
  pickle.dump(results_rewards, f)
with open('results/heuristic_lengths.pkl', 'wb') as f:
  pickle.dump(results_length, f)