import itertools
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

  def __init__(self, network: Network, attack_distribution, lines_per_turn):
    super(PowerGrid, self).__init__()
    #Keep track of timesteps and horizen
  
    self.lines_per = lines_per_turn
    timesteps = 12 / lines_per_turn # The more lines removable at each turn, the less timesteps the adversary has to work
    self.timesteps = timesteps
    self.current_step = 0

    #Stor network and initial for reset
    self.INITIAL_NETWORK = network
    self.network = network.copy()

    #List of probabilities for each edge
    self.attack_distribution = attack_distribution

    self.NUM_LINES = self.INITIAL_NETWORK.lines.shape[0]
    #Status of each line, start active
    self.lines = np.ones(self.NUM_LINES,dtype = np.int8)
    self.removed_lines = {None}
    # Actions are defend line(s), each action correspoonds to the index of the line to defend.
    
    if self.lines_per == 1:
      self.action_space = spaces.Discrete(network.lines.shape[0])
    elif self.lines_per == 2:
      combos = itertools.combinations(network.lines.shape[0], 2)
      self.action_space = spaces.Discrete(combos)
    elif self.lines_per == 3:
      combos = itertools.combinations(network.lines.shape[0], 3)
      self.action_space = spaces.Discrete(combos)

    #Observations are just lines whether they are up or down. 
    low = np.zeros(self.NUM_LINES,dtype = np.int8)
    high = np.ones(self.NUM_LINES,dtype = np.int8)
    self.observation_space = spaces.Box(low, high, dtype=np.int8)

  def step(self, defender_action): 
    done = False
    attacker_action = None
    
    if self.lines_per == 1: 
      #Sample from attack distribution until we get a line thats not removed
      while attacker_action in self.removed_lines:
        attacker_action = np.random.choice(self.NUM_LINES,p = self.attack_distribution)
      # If not defended, remove line and update network
      if defender_action != attacker_action:
          lopf_status = self._apply_attack(attacker_action)
      else:
        lopf_status = ('ok',None)
    
    if self.lines_per == 2:
      #Sample from attack distribution until we get a combo of lines none of which have been removed
      while any(item in attacker_action for item in self.removed_lines): 
        if self.network.lines.shape[0] == 1:
          attacker_action = np.random.choice(self.action_space, size = 1, p = self.attack_distribution)
        else: 
          attacker_action = np.random.choice(self.action_space, size = 2, p = self.attack_distribution)
      # If not defended, remove lines and update network
      
      if defender_action != attacker_action:
        for line in attacker_action: 
          if line in defender_action: #if this action is defended
            attacker_action.remove(line)
        lopf_status = self._apply_attack(attacker_action) #apply removal to lines that weren't defended
      else:
        lopf_status = ('ok',None)

    if self.lines_per == 3:
      #Sample from attack distribution until we get a combo of lines none of which have been removed
      while any(item in attacker_action for item in self.removed_lines): 
        if self.network.lines.shape[0] == 1:
          attacker_action = np.random.choice(self.action_space, size = 1, p = self.attack_distribution)
        elif self.network.lines.shape[0] == 2: 
          attacker_action = np.random.choice(self.action_space, size = 2, p = self.attack_distribution)
        else:
          attacker_action = np.random.choice(self.action_space, size = 3, p = self.attack_distribution)

      # If not defended, remove lines and update network
      if defender_action != attacker_action:
        for line in attacker_action: 
          if line in defender_action: #if this action is defended
            attacker_action.remove(line)
        lopf_status = self._apply_attack(attacker_action) #apply removal to lines that weren't defended
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

  def _apply_attack(self,attacked_lines):
    for line in attacked_lines:
      self.lines[line] = 0
      self.removed_lines.add(line)
    lines_to_remove = [self._attacked_line_to_line_name(line) for line in attacked_lines]
    self.network.remove("Line",lines_to_remove)
    try:
      lopf_status = self.network.lopf(pyomo=False,solver_name='gurobi',solver_options = {'OutputFlag': 0})
    except Exception as e:
      print(e)
      lopf_status = ('Failure',None)
    return lopf_status

  #Reward is power not delivered
  def _calculate_reward(self,lopf_status):
    #If not feasible, return positive 1000 and True
    if lopf_status[0] != 'ok':
      isFailure = True
      reward = np.inf
    else:
      reward = (self.INITIAL_NETWORK.generators_t.p.loc['now'].sum() - self.network.loads_t.p.loc['now'].sum()) #diff between inital (goal) generation and current generation
      isFailure = False
    return reward, isFailure

  """def choose_action(self):
    # choose an action randomly based on total q value proportion
      totq = sum(self.q_table)
      map = {}
      for line in self.network.lines:
        map[line.index] = self.q_table[line.index]/totq

    return np.random.choice(self.NUM_LINES, p = map)"""
    

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

    fig = plt.figure(figsize=(6, 3))

    data = self.network.plot(bus_colors=color, bus_cmap=plt.cm.RdYlGn, line_widths = 5.0, bus_sizes = .1)

    busTooltip = mpld3.plugins.PointHTMLTooltip(data[0], busValue,0,0,-50)
    fileName = "outputs/network" + str(self.current_step) + ".html"

    mpld3.plugins.connect(fig, busTooltip)

    html_fig = mpld3.fig_to_html(fig)

    #Writes the info we want there, then appends the fig html
    write_file = open(fileName, 'w')
    append_file = open(fileName, 'a')

    # TODO
    # add more detail about visualization here
    html_text = "<div><h1> This is Step: " + str(self.current_step) + " </h1></div>"

    write_file.write(html_text)
    write_file.close()

    del_axes_css = "<style>g.mpld3-xaxis, g.mpld3-yaxis {display: none;}</style>"
    append_file.write(html_fig)
    append_file.write(del_axes_css)
    append_file.close()
    pass