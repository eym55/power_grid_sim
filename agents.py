import gym
from gym import spaces
from pypsa import Network
import pandas as pd
import numpy as np
import random
from ray.rllib.agents import dqn

class Agent():
  def __init__(self,game_env):
    self.game_env = game_env
    #currently only handles discrete
    self.action_space = self.game_env.action_space
  def compute_action(self,state):
    pass
  def get_action_distribution(self,state):
    pass

class RandomAgent(Agent):
  def __init__(self,game_env,action_distribution = None):
    super(RandomAgent, self).__init__()
    if action_distribution:
      self.action_distribution = action_distribution
    else:
      self.action_distribution = np.ones(self.action_space.n)

  def compute_action(self, state):
    current_distribution = state['lines'] * self.action_distribution
    current_distribution = current_distribution / current_distribution.sum()
    action = np.random.choice(self.action_space.n,p=current_distribution)
    return action

class DQNAgent(Agent):
  def __init__(self,game_env,agent_config,agent_checkpoint):
    super(RandomAgent, self).__init__()
    self.agent = dqn.DQNTrainere(env = self.game_env,config = agent_config)
    self.agent.load_checkpoint(agent_checkpoint)
  def compute_action(self,state):
    action = self.agent.compute_action()
    return action