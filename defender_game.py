import gym
from gym import spaces
from pypsa import Network
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpld3
import re
import copy
import warnings


#If cartopy causing errors delete line below then go to fig code in render.
import cartopy.crs as ccrs

from scipy.special import comb
from itertools import combinations

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

    self.lines_per_turn = env_config['lines_per_turn']
    self.NUM_LINES = self.initial_lines.shape[0]
    self.num_actions = int(comb(self.NUM_LINES,self.lines_per_turn))
    #Status of each line, start active
    self.lines = np.ones(self.NUM_LINES,dtype = np.int8)
    self.removed_lines = {None}
    #Store all actions as list
    self.actions = [set(i) for i in combinations(range(self.NUM_LINES),self.lines_per_turn)]
    # Actions are defend line, each action correspoonds to the index of the line to defend.
    self.action_space = spaces.Discrete(len(self.actions))
    #Observations are just lines whether they are up or down. 

    lines_obs_space = spaces.Box(np.zeros(self.NUM_LINES,dtype = np.int8), np.ones(self.NUM_LINES,dtype = np.int8), dtype=np.int8)
    loads_obs_space = spaces.Box(np.zeros(self.network.loads.shape[0],dtype = np.int8), np.full(self.network.loads.shape[0],self.network.loads['p_set'].max()), dtype=np.int8)
    self.observation_space = spaces.Dict({"lines": lines_obs_space, "loads":loads_obs_space})

    #Load adversary
    agent_config = env_config['agent_config']
    agent_class = env_config['agent_class']
    self.adversary_agent = agent_class(game_env=self,agent_config=agent_config)

  def step(self, action):
    self.removed_lines_this_step = []
    done = False
    #get state and adversary action
    current_state ={'lines':self.lines,'loads':self.network.loads['p_set']}
    attacker_action = self.adversary_agent.compute_action(current_state)
    # If not defended, remove line and update network
    lopf_status = self._apply_attack(action,attacker_action)
    #TODO fix
    # if action != attacker_action:
    #     lopf_status = self._apply_attack(action,attacker_action)
    # else:
    #   lopf_status = ('ok',None)
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

  def _apply_attack(self,defended_lines,attacked_lines):
    attacked_lines = self.actions[attacked_lines] - self.actions[defended_lines]
    if len(attacked_lines) == 0:
      lopf_status = ('ok',None)
      return lopf_status
    affected_nodes = []
    for line in attacked_lines:
      self.lines[line] = 0
      self.removed_lines.add(line)
      line_to_remove = self._attacked_line_to_line_name(line)
      affected_nodes.extend(list(self.network.lines.loc[line_to_remove][['bus0','bus1']].values))
      self.network.remove("Line",line_to_remove)
    affected_nodes = np.array(affected_nodes)
    try:
      lopf_status = self._call_lopf()
      #TODO was this breaking shit?
      # if lopf_status[0] == 'Failure':
      #   raise(Exception)
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
      self.network.loads.at[load_to_remove, 'p_set'] = 0
      lopf_status = self._call_lopf()
      return lopf_status, affected_nodes
    else:
      load_to_remove = snom_to_load_ratios.index[0]
      self.network.loads.at[load_to_remove,'p_set'] = 0
      lopf_status = self._call_lopf()
      return lopf_status, np.array([])

  def _call_lopf(self):
    try:
      lopf_status = self.network.lopf(pyomo=False,solver_name='gurobi',solver_options = {'OutputFlag': 0,'SOLUTION_LIMIT':1},solver_logfile=None,store_basis = False,warmstart = False) 
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
    # If no defends, more users will be harmed
    self.network.lines = self.initial_lines.copy()
    self.network.loads = self.initial_loads.copy()
    self.lines = np.ones(self.NUM_LINES,dtype=np.int8)
    self.removed_lines = {None}
    self.current_step = 0
    observation = {'lines':self.lines,'loads':self.network.loads['p_set']}
    return observation

  # TODO

  # Add Legend - in progress - not easy with mpl x pypsa
  # Add Snom interaction w/ lines - added
  # Normalize/change pSqueeze Values so Viz looks better - ask
  # Give context of what the agents are doing (text of what happened, reward amt etc.) - ask group
  # Make change map where values are based off bus/line value changes such that differences are easier to spot
  # Figure out how to access past states of the network ^^
  # Zheng Notes:
  # Show parts getting attacked, and defended.
  # Visualization of Hurriacane

  def render(self, mode='human', close=False):
    # Render the environment to the screen - ??

    # Renames bus_nums so they don't start very high.
    def rename_bus_num(ls_bus_nums):
      new_ls = []
      for i in range(len(ls_bus_nums)):
        int_bus_num = int(ls_bus_nums[i][4:])
        #new_num = int_bus_num - reduce_num
        new_ls.append("Bus " + str(i+1))
      return new_ls

    # Renames buses
    bus_nums = list(self.network.buses.index)
    new_bus_nums = rename_bus_num(bus_nums)

    # Add more necessary values here
    p_squeeze_tuple = []
    bus_p_squeeze = list(self.network.buses_t.p.squeeze())
    max_length = max(len(new_bus_nums), len(bus_p_squeeze))
    for i in range(max_length):
      #Add new name for units... or delete.
      str_p_squeeze = " Active Power =  " + str(bus_p_squeeze[i]) + " units"
      p_squeeze_tuple.append([new_bus_nums[i], str_p_squeeze])


    #Create colormaps
    cmap = plt.cm.RdYlGn
    cmap2 = plt.cm.cividis_r

    # Colorbar not really working with mpld3 or pypsa mpl
    colorBar = plt.figure()
    ax = colorBar.add_axes([0.05, 0.80, 0.9, 0.1])


    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap)

    #html_colorBar = mpld3.fig_to_html(colorBar, figid="colorBar")

    bus_color = self.network.buses_t.p.squeeze()
    line_color = self.network.lines.s_nom

    # If cartopy giving errors, uncomment line below. Comment out subplots line.
    #fig= plt.figure(figsize=(6, 3))
    #fig, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(2, 1, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(5, 4))

    title = "Texas Grid"
    if(len(new_bus_nums) < 30):
      title = "Lopf Grid"

    #Create network based off the initial network
    new_net = self.network.copy()
    new_net.buses = self.initial_buses.copy()
    new_net.lines = self.initial_lines.copy()
    new_net.loads = self.initial_loads.copy()

    # Using S_nom to represent lines that are up or down.

    new_net.lines['s_nom'] = 1000
    new_net.lines['s_nom'] = new_net.lines['s_nom'] * self.lines
    new_net.lines['s_nom'].replace(0, 10)

    # Trying to fix graph on step 0.
    if(self.current_step == 0):
      new_net.lines['s_nom'] = new_net.lines['s_nom'] * 100000
    # Expand on this detail

    #curr_reward_string = "Current Reward is " + str(self._calculate_reward(self._call_lopf())[0])

    data = self.network.plot(ax=ax1, title=title, bus_colors=bus_color, bus_cmap=cmap, line_colors=line_color, line_cmap=plt.cm.YlGn_r, line_widths = 3.0, bus_sizes = .005)


    data3 = new_net.plot(ax = ax2, title= "Lines Removed", bus_colors= bus_color, bus_cmap=plt.cm.YlGn, line_colors = new_net.lines['s_nom'], line_cmap = plt.cm.Reds_r, line_widths = 4,  bus_sizes = .000005)
    #ax.legend(['Power Outflow'])

    #ax.colorbar(location="bottom")

    busTooltip = mpld3.plugins.PointHTMLTooltip(data[0], p_squeeze_tuple, 0, -50, -100)
    #lineTooltip = mpld3.plugins.LineLabelTooltip(list(data[1][0]), tup, 0, -50, -100)

    fileName = "outputs/network" + str(self.current_step) + ".html"

    mpld3.plugins.connect(fig, busTooltip)
    #mpld3.plugins.connect(fig, lineTooltip)

    html_fig = mpld3.fig_to_html(fig, figid='fig1')

    #Writes the info we want there, then appends the fig html
    write_file = open(fileName, 'w')
    append_file = open(fileName, 'a')

    # TODO
    # add more detail about visualization here
    # Write template html file, so some of these vars can be erased.
    html_text = "<div style=\"text-align: center;\"><h1> This is Step: " + str(self.current_step+1) + " </h1><h3>" #+ str(curr_reward_string)+ "</h3></div>"
    #Inside box stays white, so it doesn't look great.
    #bg_html = "<body style = \"background-image: url(\'https://wallpaperaccess.com/full/187161.jpg\');\" ></<body>"
    #bg_html = "<body style = \"background-color:#E6E6FA;\" ></body>"

    #total_beg_text = html_text + bg_html
    write_file.write(html_text)
    write_file.close()

    append_file.write(html_fig)
    #append_file.write(html_colorBar)

    center_fig_html = f'''<style type="text/css">div#fig1 {{ text-align: center; }}</style>'''

    #lavender bg
    #center_fig_html = f'''<style type="text/css">div#fig1 {{ text-align: center; background-color:#E6E6FA; }}</style>'''

    #legend_html = "<ul class=\"charts-css legend legend-circle\" float: right;>\n<li> Label 1 </li><li> Label 2 </li><li> Label 3 </li></ul>"
    del_axes_css = "<style>g.mpld3-xaxis, g.mpld3-yaxis {display: none;}</style>"


    total_css = center_fig_html + del_axes_css
    append_file.write(total_css)
    append_file.close()
    pass

