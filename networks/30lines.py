import pypsa
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0)
np.random.seed(0)
network = pypsa.Network()
#Generating 15 buses
for i in range(15):
    network.add("Bus","Bus {}".format(i))
#Assinging buses lat and long
for i in range(15):
    network.buses.at[network.buses.index[i], 'x'] = random.randint(0,100)
    network.buses.at[network.buses.index[i], 'y'] = random.randint(0,100)

#Creating a line for each of the edges between buses
edges = [(0, 12),(0, 4),(0, 5),(1, 10),(1, 3),
(1, 6),(1, 9),(10, 3),(11, 3),(11, 4),(11, 5),
(11, 8),(12, 13),(13, 14),(13, 4),(13, 7),
(14, 2),(14, 7),(2, 6),(2, 7),(3, 6),(3, 8),
(4, 6),(4, 7),(5, 8),(6, 7),(6, 9),(4, 3),(9, 2),(14, 6)]
for i in range(len(edges)):
    network.add("Line","Linine {}".format(i),
                bus0="Bus {}".format(edges[i][0]),
                bus1="Bus {}".format(edges[i][1]),
                x=0.0001,
                s_nom=np.random.normal(loc=120,scale=40))

genBus = [2,3,9,10,13,14]
loadBus = [0,1,4,5,6,7,8,11,12]
#Add generators and loads at specific buses
for i in genBus:
    network.add("Generator","Gen {}".format(i),
                bus="Bus {}".format(i),
                p_nom=random.randint(200,600),
                marginal_cost=1)

for i in loadBus:
    network.add("Load",f"Load {i}",
                bus=f"Bus {i}",
                p_set=random.randint(25,125))

initial_lopf_status = network.lopf(pyomo=False,solver_name='gurobi',solver_options = {'OutputFlag': 0,'SOLUTION_LIMIT':1},solver_logfile=None,store_basis = True)
if initial_lopf_status[0] != 'ok':
  print(initial_lopf_status)
  raise ValueError('The original network is not feasible')
#EXPORT
network.export_to_netcdf("30line.nc")
