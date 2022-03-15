
import gym
from gym import spaces
from pypsa import Network
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import mpld3

import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


class PowerGrid(gym.Env):

  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, network: Network, attack_distribution,timesteps = 10):
    super(PowerGrid, self).__init__()
    #Keep track of timesteps and horizen
    self.timesteps = timesteps
    self.current_step = 0

    #Stor network and initial for reset
    self.INITIAL_NETWORK = network
    self.network = network.copy()

    #List of probabilities for each edge
    self.attack_distribution = attack_distribution

    self.NUM_LINES = self.network.lines.shape[0]
    #Status of each line, start active
    self.lines = np.ones(self.NUM_LINES,dtype = np.int8)
    self.removed_lines = {None}
    # Actions are defend line, each action correspoonds to the index of the line to defend.
    self.action_space = spaces.Discrete(network.lines.shape[0])
    #Observations are just lines whether they are up or down. 
    low = np.zeros(self.NUM_LINES,dtype = np.int8)
    high = np.ones(self.NUM_LINES,dtype = np.int8)
    self.observation_space = spaces.Box(low, high, dtype=np.int8)

  def step(self, action):
    done = False
    #Sample from attack distribution until we get a line thats not removed
    attacker_action = None
    while attacker_action in self.removed_lines:
      attacker_action = np.random.choice(self.NUM_LINES,p = self.attack_distribution)
    # If not defended, remove line and update network
    if action != attacker_action:
        lopf_status = self._apply_attack(attacker_action)
    else:
      lopf_status = ('ok',None)
    self.current_step +=1
    
    reward,isFailure = self._calculate_reward(lopf_status)
    #Check if network is infeasible 
    if isFailure:
      done = True
    #Check if network has no lines
    if self.network.lines.shape[0] == 0:
      done = True
    #Check if horizon reached
    if self.current_step == self.timesteps:
      done = True
    
    observation = self.lines
    
    return  observation, reward, done, {}

  def _attacked_line_to_line_name(self,attacked_line):
    return self.INITIAL_NETWORK.lines.index[attacked_line]

  def _apply_attack(self,attacked_line):
    self.lines[attacked_line] = 0
    self.removed_lines.add(attacked_line)
    line_to_remove = self._attacked_line_to_line_name(attacked_line)
    self.network.remove("Line",line_to_remove)
    try:
      lopf_status = self.network.lopf(pyomo=False,solver_name='gurobi',solver_options = {'OutputFlag': 0})
    except Exception as e:
      print(e)
      lopf_status = ('Failure',None)
    return lopf_status

  #Reward is -power not delivered
  def _calculate_reward(self,lopf_status):
    #If not feasible, return negative infinity and True
    if lopf_status[0] != 'ok':
      isFailure = True
      reward =-10000
    else:
      reward = self.network.loads['p_set'].sum()
      isFailure = False
    return reward, isFailure

  def reset(self):
    # Reset the state of the environment to an initial state
    self.network = self.INITIAL_NETWORK.copy()
    self.lines = np.ones(self.NUM_LINES,dtype=np.int8)
    self.removed_lines = {None}
    self.current_step = 0
    return self.lines
  #TODO add rendering here
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    busValue = list(self.network.buses.index)
    color = self.network.buses_t.p.squeeze()

    fig = plt.figure(figsize=(12, 6))

    data = self.network.plot(bus_colors=color, bus_cmap=plt.cm.RdYlGn, line_widths = 5.0, bus_sizes = .1)


    busTooltip = mpld3.plugins.PointHTMLTooltip(data[0], busValue,0,0,-50)
    fileName = "network" + str(3) + ".html" 
    mpld3.plugins.connect(fig,busTooltip)
    mpld3.save_html(fig, fileName)
    pass