import gym
from gym import spaces
from pypsa import Network
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import mpld3
import re
import copy
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logging.getLogger("pypsa").setLevel(logging.CRITICAL)
np.random.seed(10)
import agents
class PowerGrid(gym.Env):

  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, env_config):
    super(PowerGrid, self).__init__()
    #Keep track of timesteps and horizen
    if 'timesteps' in env_config:
      self.timesteps = env_config['timesteps']
    else: self.timesteps=15
    self.current_step = 0

    #Store network and initial for reset
    self.network = env_config['network']
    self.initial_lines = self.network.lines.copy()
    self.initial_loads = self.network.loads.copy()


    self.NUM_LINES = self.initial_lines.shape[0]
    #Status of each line, start active
    self.lines = np.ones(self.NUM_LINES,dtype = np.int8)
    self.removed_lines = {None}
    # Actions are defend line, each action correspoonds to the index of the line to defend.
    self.action_space = spaces.Discrete(self.NUM_LINES)
    #Observations are just lines whether they are up or down. 
    lines_obs_space = spaces.Box(np.zeros(self.NUM_LINES,dtype = np.int8), np.ones(self.NUM_LINES,dtype = np.int8), dtype=np.int8)
    loads_obs_space = spaces.Box(np.zeros(self.network.loads.shape[0],dtype = np.int8), np.full(self.network.loads.shape[0],self.network.loads['p_set'].max()), dtype=np.int8)
    self.observation_space = spaces.Dict({"lines": lines_obs_space, "loads":loads_obs_space})

    #Load adversary
    agent_config = env_config['agent_config']
    agent_class = env_config['agent_class']
    self.adversary_agent = agent_class(game_env=self,agent_config=agent_config)

  def step(self, action):
    done = False
    #get state and adversary action
    current_state ={'lines':self.lines,'loads':self.network.loads['p_set']}
    attacker_action = self.adversary_agent.compute_action(current_state)
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
    #Check if loads are all 0
    if reward == 0:
      done = True
    #Check if horizon reached
    if self.current_step == self.timesteps:
      done = True
    
    observation = {'lines':self.lines,'loads':self.network.loads['p_set']}
    return  observation, reward, done, {}

  def _attacked_line_to_line_name(self,attacked_line):
    return self.initial_lines.index[attacked_line]

  def _sc_optimize(self, removed_line_index):
    now = self.network.snapshots[0]
    line = self.network.lines.index[removed_line_index:removed_line_index + 1]
    try:
      sclopf_status = self.network.sclopf(now,branch_outages=line, solver_name='cbc')
    except Exception as e:
      sclopf_status = ('Failure',None)

    self.network.generators_t.p_set = self.network.generators_t.p_set.reindex(columns=self.network.generators.index)
    self.network.generators_t.p_set.loc[now] = self.network.generators_t.p.loc[now]
    return sclopf_status

  def _apply_attack(self,attacked_line):
    self.lines[attacked_line] = 0
    self.removed_lines.add(attacked_line)
    line_to_remove = self._attacked_line_to_line_name(attacked_line)

    affected_nodes = self.network.lines.loc[line_to_remove][['bus0','bus1']].values
    self.network.remove("Line",line_to_remove)
    try:
      lopf_status = self._call_lopf()
      if lopf_status[0] == 'Failure':
        raise(Exception)
      while lopf_status[0] != 'ok':
        lopf_status,affected_nodes = self._fix_infeasibility(affected_nodes)
    except Exception as e:
      lopf_status = ('Failure',None)
    return lopf_status

  # Helper method to iterativly remove loads until network is feasible.
  # If no feasinle network can be found
  def _fix_infeasibility(self,affected_nodes):
    def snom_over_load(row):
      bus = row['bus']
      load = row['p_set']
      return self.network.lines[(self.network.lines['bus0'] == bus) | (self.network.lines['bus1'] == bus)]['s_nom'].sum() / load
    snom_to_load_ratios = self.network.loads[self.network.loads != 0].apply(lambda x: snom_over_load(x),axis=1).sort_values(ascending = True)
    #Remove any nodes that have cumulative s_nom < their load
    if snom_to_load_ratios.iloc[0] < 1:
      load_to_remove = snom_to_load_ratios.index[0]
      affected_nodes = affected_nodes[affected_nodes != self.network.loads.loc[load_to_remove].bus]
      self.network.loads.at[load_to_remove,'p_set'] = 0
      lopf_status = self._call_lopf()
      return lopf_status, affected_nodes
    if affected_nodes.any():
      affected_loads = self.network.loads['bus'].isin(affected_nodes)
      if not snom_to_load_ratios.loc[affected_loads].empty:
        snom_to_load_ratios = snom_to_load_ratios.loc[affected_loads]
      load_to_remove = snom_to_load_ratios.index[0]
      affected_nodes = affected_nodes[affected_nodes != self.network.loads.loc[load_to_remove].bus]
      self.network.loads.at[load_to_remove,'p_set'] = 0
      lopf_status = self._call_lopf()
      return lopf_status, affected_nodes
    else:
      load_to_remove = snom_to_load_ratios.index[0]
      self.network.loads.at[load_to_remove,'p_set'] = 0
      lopf_status = self._call_lopf()
      return lopf_status, np.array([])
    
  def _call_lopf(self):
    try:
      lopf_status = self.network.lopf(pyomo=False,solver_name='gurobi',solver_options = {'OutputFlag': 0},solver_logfile=None,store_basis = False,warmstart = False) 
    except Exception as e:
      print(e)
      lopf_status = ('Failure',None)
    return lopf_status

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
    self.network.lines = self.initial_lines.copy()
    self.network.loads = self.initial_loads.copy()
    self.lines = np.ones(self.NUM_LINES,dtype=np.int8)
    self.removed_lines = {None}
    self.current_step = 0
    observation = {'lines':self.lines,'loads':self.network.loads['p_set']}
    return observation

  #TODO add rendering here
  def render(self, mode='human', close=False):
    # # Render the environment to the screen
    # busValue = list(self.network.buses.index)
    # color = self.network.buses_t.p.squeeze()

    # fig = plt.figure(figsize=(6, 3))

    # data = self.network.plot(bus_colors=color, bus_cmap=plt.cm.RdYlGn, line_widths = 5.0, bus_sizes = .1)

    # busTooltip = mpld3.plugins.PointHTMLTooltip(data[0], busValue,0,0,-50)
    # fileName = "outputs/network" + str(self.current_step) + ".html" 

    # mpld3.plugins.connect(fig, busTooltip)

    # html_fig = mpld3.fig_to_html(fig)

    # #Writes the info we want there, then appends the fig html
    # write_file = open(fileName, 'w')
    # append_file = open(fileName, 'a')

    # # TODO
    # # add more detail about visualization here
    # html_text = "<div><h1> This is Step: " + str(self.current_step) + " </h1></div>"

    # write_file.write(html_text)
    # write_file.close()

    # append_file.write(html_fig)
    # append_file.close()

    pass