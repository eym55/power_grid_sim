
import gym
from gym import spaces
from pypsa import Network
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import mpld3
import re
import itertools

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
    if len(inpt_list) != len(set(inpt_list)): # Since sets contain no duplicates
      return True
    else:
      return False

  def step(self, attacker_action):
    done = False
    defender_action = [None]
    attack_act =  attacker_action

    if self.lines_per == 1: 
      #Sample from attack distribution until we get a line thats not removed
      while defender_action[0] in self.removed_lines:
        defender_action = tuple([np.random.choice(self.NUM_LINES,p = self.attack_distribution)])
      # If not defended, remove line and update network
      if attacker_action != defender_action:
          lopf_status = self._apply_attack(tuple([attack_act])) 
      else:
        lopf_status = ('ok',None) 
    
    if self.lines_per == 2:
      print("lines are currently ", str(self.lines), "INTENDED attacker action is  ",attack_act, " at timestep ", self.current_step)
      #Sample from attack distribution until we get a combo of lines none of which have been removed   
      while (defender_action[0] in self.removed_lines or defender_action[1] in self.removed_lines or defender_action[0] == defender_action[1]):
        if self.network.lines.shape[0] == 1: 
          defender_action = tuple(self.network.lines[0]) 
        else: 
          defender_action = tuple(np.random.choice(self.NUM_LINES, size = 2, p = self.attack_distribution))
      # If not defended, remove lines and update network  
      if sorted(defender_action) != sorted(attack_act): 
        for line in attack_act: #for each line in the pair of lines   
          if line in defender_action: #if this action is defended 
            filtered = filter(lambda x: x != line, attack_act)
            attack_act = tuple(filtered) #remove from the list of lines to remove
        lopf_status = self._apply_attack(attack_act) #apply removal to lines that weren't defended
      else:
        lopf_status = ('ok',None)

    if self.lines_per == 3:

      #Sample from attack distribution until we get a combo of lines none of which have been removed
      while (defender_action[0] in self.removed_lines or defender_action[1] in self.removed_lines or defender_action[2] in self.removed_lines or self.any_duplicates(defender_action)):
          if self.network.lines.shape[0] == 1:
            defender_action = self.network.lines[0] 
          elif self.network.lines.shape[0] == 2: 
            defender_action = np.random.choice(self.NUM_LINES, size = 2, p = self.attack_distribution)
          else:
            defender_action = np.random.choice(self.NUM_LINES, size = 3, p = self.attack_distribution)
      # If not defended, remove lines and update network
      if sorted(defender_action) != sorted(attack_act):
        for line in attacker_action:  #for each line in the triplet of lines
          if line in defender_action: #if this action is defended
            filtered = filter(lambda x: x != line, attack_act)
            attack_act = tuple(filtered) #remove from the list of lines to remove
        print("attacker action at timestep", self.current_step, "is ", attack_act)
        lopf_status = self._apply_attack(attack_act) #apply removal to lines that weren't defended
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

  def _sc_optimize(self, removed_line_index):
    now = self.network.snapshots[0]
    line = self.network.lines.index[removed_line_index:removed_line_index + 1]
    try:
      sclopf_status = self.network.sclopf(now,branch_outages=line, solver_name='cbc')
    except Exception as e:
      print(e)
      sclopf_status = ('Failure',None)

    self.network.generators_t.p_set = self.network.generators_t.p_set.reindex(columns=self.network.generators.index)
    self.network.generators_t.p_set.loc[now] = self.network.generators_t.p.loc[now]
    return sclopf_status

  def _apply_attack(self,attacked_lines):
    affected_nodes = []
    if len(attacked_lines) > 1: #if more than one line being removed enter loop
      for line in attacked_lines:
        self.lines[line] = 0
        self.removed_lines.add(line)
      lines_to_remove = [self._attacked_line_to_line_name(line) for line in attacked_lines]  
      for line in lines_to_remove:
        affected_nodes.append(self.network.lines.loc[line][['bus0','bus1']].values)
        self.network.remove("Line",line) 
    else: #if one line being attacked
      self.lines[attacked_lines[0]] = 0
      self.removed_lines.add(attacked_lines[0])
      lines_to_remove = self._attacked_line_to_line_name(attacked_lines[0])
      affected_nodes.append(self.network.lines.loc[lines_to_remove][['bus0','bus1']].values)
      self.network.remove("Line",lines_to_remove)

    try:
      lopf_status = self.network.lopf(pyomo=False,solver_name='cbc')
      while lopf_status[0] != 'ok':
        lopf_status,affected_nodes = self._fix_infeasibility(affected_nodes)
    except Exception as e:
      print(e,)
      lopf_status = ('Failure',None)
    return lopf_status

  # Helper method to iterativly remove loads until network is feasible.
  # If no feasinle network can be found
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
    if affected_nodes.any():
      affected_loads = self.network.loads['bus'].isin(affected_nodes)
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
    #If not feasible, return negative infinity and True
    if lopf_status[0] != 'ok':
      isFailure = True
      reward = 0
    else:
      reward = self.network.loads.p_set.sum()
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
