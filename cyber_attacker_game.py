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
logger.setLevel(logging.WARNING)
logging.getLogger("pypsa").setLevel(logging.WARNING)


class PowerGrid(gym.Env):

  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, network: Network, attack_distribution,lines_per_turn):
    super(PowerGrid, self).__init__()
    #Keep track of timesteps and horizen
    self.lines_per = lines_per_turn
    timesteps = 12 / lines_per_turn # The more lines removable at each turn, the less timesteps the adversary has to work
    self.timesteps = timesteps
    self.current_step = 0

    #Stor network and initial for reset
    self.INITIAL_NETWORK = network
    self.network = network.copy()
    self.initial_loads = self.INITIAL_NETWORK.loads['p_set'].sum()

    #List of probabilities for each edge
    self.attack_distribution = attack_distribution

    self.NUM_LINES = self.INITIAL_NETWORK.lines.shape[0]
    #Status of each line, start active
    self.lines = np.ones(self.NUM_LINES,dtype = np.int8)
    self.removed_lines = {None}
    # Actions are defend line, each action correspoonds to the index of the line to defend.
    if self.lines_per == 1:
      self.action_space = spaces.Discrete(network.lines.shape[0])
    elif self.lines_per == 2:
      self.action_space = spaces.Tuple((
        spaces.Discrete(network.lines.shape[0]),
        spaces.Discrete(network.lines.shape[0])))
    elif self.lines_per == 3:
      self.action_space = spaces.Tuple((
        spaces.Discrete(network.lines.shape[0]),
        spaces.Discrete(network.lines.shape[0]),
        spaces.Discrete(network.lines.shape[0])))
    #Observations are just lines whether they are up or down. 
    low = np.zeros(self.NUM_LINES,dtype = np.int8)
    high = np.ones(self.NUM_LINES,dtype = np.int8)
    self.observation_space = spaces.Box(low, high, dtype=np.int8)

  def any_duplicates(self, inpt_list):
    seen = []
    for elem in inpt_list:
      if elem in seen:
        return True
      else:
        seen.append(elem)
    return False


  def step(self, defender_action):
    done = False
    #Sample from attack distribution until we get a line thats not removed
    attacker_action = [None]
    defend_act = defender_action

    if self.lines_per == 1: 
      #Sample from attack distribution until we get a line thats not removed
      while attacker_action[0] in self.removed_lines:
        attacker_action = tuple([np.random.choice(self.NUM_LINES,p = self.attack_distribution)])
      # If not defended, remove line and update network
      if attacker_action != defender_action:
          lopf_status = self._apply_attack(attacker_action) 
      else:
        lopf_status = ('ok',None) 
    
    if self.lines_per == 2:
      #Sample from attack distribution until we get a combo of lines none of which have been removed   
      while (attacker_action[0] in self.removed_lines or attacker_action[1] in self.removed_lines or attacker_action[0] == attacker_action[1]):
        if self.network.lines.shape[0] == 1: 
          attacker_action = tuple(self.network.lines[0]) 
        else: 
          attacker_action = tuple(np.random.choice(self.NUM_LINES, size = 2, p = self.attack_distribution))
      # If not defended, remove lines and update network  
      if sorted(defend_act) != sorted(attacker_action): 
        for line in attacker_action: #for each line in the pair of lines   
          if line in defend_act: #if this action is defended 
            filtered = filter(lambda x: x != line, attacker_action)
            attacker_action = tuple(filtered) #remove from the list of lines to remove
        lopf_status = self._apply_attack(attacker_action) #apply removal to lines that weren't defended
      else:
        lopf_status = ('ok',None)

    if self.lines_per == 3:

      #Sample from attack distribution until we get a combo of lines none of which have been removed
      while (attacker_action[0] in self.removed_lines or attacker_action[1] in self.removed_lines or attacker_action[2] in self.removed_lines or self.any_duplicates(attacker_action)):
          if self.network.lines.shape[0] == 1:
            attacker_action = self.network.lines[0] 
          elif self.network.lines.shape[0] == 2: 
            attacker_action = np.random.choice(self.NUM_LINES, size = 2, p = self.attack_distribution)
          else:
            attacker_action = np.random.choice(self.NUM_LINES, size = 3, p = self.attack_distribution)
      # If not defended, remove lines and update network
      if sorted(defend_act) != sorted(attacker_action):
        for line in attacker_action:  #for each line in the triplet of lines
          if line in defender_action: #if this action is defended
            filtered = filter(lambda x: x != line, attacker_action)
            attacker_action = tuple(filtered) #remove from the list of lines to remove
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
    affected_nodes = []
    if len(attacked_lines) > 1: #if more than one line being removed enter loop
      for line in attacked_lines:
        self.lines[line] = 0
        self.removed_lines.add(line)
      lines_to_remove = [self._attacked_line_to_line_name(line) for line in attacked_lines]  
      for line in lines_to_remove:
        buses = self.network.lines.loc[line][['bus0','bus1']].values
        for bus in buses:
          affected_nodes.append(bus)
        self.network.remove("Line",line)
    else: #if one line being attacked
      self.lines[attacked_lines[0]] = 0
      self.removed_lines.add(attacked_lines[0])  
      lines_to_remove = self._attacked_line_to_line_name(attacked_lines[0])
      affected_nodes = self.network.lines.loc[lines_to_remove][['bus0','bus1']].values
      self.network.remove("Line",lines_to_remove)

    try:
      lopf_status = self.network.lopf(pyomo=False,solver_name='cbc')
      while lopf_status[0] != 'ok':
        lopf_status,affected_nodes = self._fix_infeasibility(affected_nodes)
    except Exception as e:
      print(e)
      lopf_status = ('Failure',None)
    return lopf_status

  # Helper method to iterativly remove loads until network is feasible.
  # If no feasible network can be found
  def _fix_infeasibility(self,affected_nodes):
    def snom_over_load(row):
      bus = row['bus']
      load = row['p_set']
      return self.network.lines[(self.network.lines['bus0'] == bus) | (self.network.lines['bus1'] == bus)]['s_nom'].sum() / load
    snom_to_load_ratios = self.network.loads.apply(lambda x: snom_over_load(x),axis=1).sort_values(ascending = True)
    #Remove any nodes that have cumulative s_nom < their load
    if snom_to_load_ratios.iloc[0] < 1:
      load_to_remove = snom_to_load_ratios.index[0]
      affected_nodes = affected_nodes[affected_nodes != self.network.loads.loc[load_to_remove].bus]
      self.network.remove('Load',load_to_remove)
      try:
        lopf_status = self.network.lopf(pyomo=False,solver_name='cbc')
      except Exception as e:
        print(e)
        lopf_status = ('Failure',None)
      return lopf_status, affected_nodes
    if len(affected_nodes) > 0:
      affected_loads = self.network.loads['bus'].isin(list(affected_nodes))
      if not snom_to_load_ratios.loc[affected_loads].empty:
        snom_to_load_ratios = snom_to_load_ratios.loc[affected_loads]
      load_to_remove = snom_to_load_ratios.index[0]
      affected_nodes = affected_nodes[affected_nodes != self.network.loads.loc[load_to_remove].bus]
      self.network.remove('Load',load_to_remove)
      try:
        lopf_status = self.network.lopf(pyomo=False,solver_name='cbc')
      except Exception as e:
        print(e)
        lopf_status = ('Failure',None)
      return lopf_status, affected_nodes
    else:
      load_to_remove = snom_to_load_ratios.index[0]
      self.network.remove('Load',load_to_remove)
      try:
        lopf_status = self.network.lopf(pyomo=False,solver_name='cbc')
      except Exception as e:
        print(e)
        lopf_status = ('Failure',None)
      return lopf_status, np.array([])

  #Reward is -power not delivered 
  def _calculate_reward(self,lopf_status):
    #If not feasible, return positive infinity and True
    if lopf_status[0] != 'ok':
      isFailure = True
      reward = 1000
    else:
      discount_factor = self.current_step / self.timesteps 
      base_reward = self.initial_loads - self.network.loads['p_set'].sum() #diff between initial load and current, essentially the damage done
      reward = base_reward - (base_reward * discount_factor) #reward for attacker becomes worse and worse every timestep that goes by.
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
    #ef render(self, mode='human', close=False):
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