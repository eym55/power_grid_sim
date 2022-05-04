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
num_episodes = 25
for episode in range(num_episodes):
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
  print(f"\n\n\n Episode {episode} done. Episode lasted {i} timesteps and had a cumulative reward of {total_reward} \n\n\n")

print(f"\n\n\n All {num_episodes} have completed. \n\n\n")
print(f"The average episode rewards was {np.mean(results_rewards)} and the mean episode length was {np.mean(results_length)} timesteps")
