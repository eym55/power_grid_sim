from asyncio import current_task
from pickletools import stringnl
import gym
from gym import spaces
from pypsa import Network
import pandas as pd
import numpy as np
import random
from ray.rllib.agents import dqn

class Agent():
  def __init__(self,game_env,agent_config):
    self.game_env = game_env
    #currently only handles discrete
    self.action_space = self.game_env.action_space
  def compute_action(self,state):
    pass
  def get_action_distribution(self,state):
    pass

class RandomAgent(Agent):
  def __init__(self,game_env,agent_config):
    self.game_env = game_env
    #currently only handles discrete
    self.action_space = self.game_env.action_space
    self.action_distribution = agent_config['action_distribution']

  def compute_action(self, state):
    current_distribution = state['lines'] * self.action_distribution
    current_distribution = current_distribution / current_distribution.sum()
    action = np.random.choice(self.action_space.n,p=current_distribution)
    return action

  def get_action_distribution(self, state):
    current_distribution = state['lines'] * self.action_distribution
    current_distribution = current_distribution / current_distribution.sum()
    return current_distribution

class DQNAgent(Agent):
  def __init__(self,game_env,agent_config):
    self.game_env = game_env
    #currently only handles discrete
    self.action_space = self.game_env.action_space
    self.checkpoint_path = agent_config['checkpoint_path']
    self.agent_config = agent_config['agent_config']
    self.agent = dqn.DQNTrainer(env = self.game_env,config = self.agent_config)
    self.agent.load_checkpoint(self.checkpoint_path)
  def compute_action(self,state):
    action = self.agent.compute_action(state)
    return action

class HurricaneAgent(Agent):
  def __init__(self,game_env,agent_config):
    self.game_env = game_env
    #currently only handles discrete
    self.action_space = self.game_env.action_space
    #list of points index by time step, points have an x,y value
    self.hurricane_path = agent_config['hurricane_path']
    self.time_step = 0
    self.lines = self.game_env.network.lines
    self.buses = self.network.buses
    self.lines['midpoint'] = self.lines.apply(lambda x:self._calculate_midpoint(x),axis=1)

  def _calculate_midpoint(self,row):
    bus_0 = row['bus0']
    bus_1 = row['bus1']
    bus_0_x = self.buses[bus_0]['x']
    bus_1_x = self.buses[bus_1]['x']
    bus_0_y = self.buses[bus_0]['y']
    bus_1_y = self.buses[bus_1]['y']
    return np.array[np.mean([bus_0_x,bus_1_x]),np.mean([bus_0_y,bus_1_y])]

  def compute_action(self,state):
    if self.time_step >= len(self.hurricane_path):
      raise 'Hurricane has passed'
    hurricane_location = np.array(self.hurricane_path[self.time_step])
    distances = self.lines['midpoint'].apply(lambda x: np.linalg.norm(x-hurricane_location))

    current_distribution = state['lines'] * distances
    current_distribution = current_distribution / current_distribution.sum()
    action = np.random.choice(self.action_space.n,p=current_distribution)

    self.time_step += 1
    return action

