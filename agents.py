from asyncio import current_task
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
    print(agent_config)
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