import pypsa
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0)
np.random.seed(0)
network = pypsa.Network()

for i in range(30):
    network.add("Bus","Bus {}".format(i))

for i in range(30):
    network.buses.at[network.buses.index[i], 'x'] = random.randint(0,100)
    network.buses.at[network.buses.index[i], 'y'] = random.randint(0,100)

edges = [(0, 12),(0, 21),(0, 22),(0, 23),(0, 6),(1, 17),(1, 18),(1, 19),(10, 17),
    (10, 24),(10, 3),(11, 3),(11, 8),(12, 2),(12, 22),(12, 24),(12, 3), (12, 6),(12, 8),
    (13, 14),(13, 21),(13, 5),(14, 16),(14, 2),(14, 21),(14, 4),(14, 5),(15, 18),(15, 19),
    (15, 2),(15, 9),(16, 4),(16, 9),(17, 18),(17, 24),(18, 19),(18, 24),(19, 7),(2, 21),
    (2, 24),(2, 6),(2, 9),(20, 9),(21, 5),(21, 6),(22, 25),(22, 8),(24, 3),(25, 8),(3, 8),
    (4, 5),(7, 9), (7, 26), (2, 27), (1, 28), (15, 29), (0, 29), (28, 4), (27, 22), (27, 23)
]
for i in range(len(edges)):
    network.add("Line","Linine {}".format(i),
                bus0="Bus {}".format(edges[i][0]),
                bus1="Bus {}".format(edges[i][1]),
                x=0.0001,
                s_nom=np.random.normal(loc=130,scale=40))

#generators
for i in range(12):
    network.add("Generator","Gen {}".format(i),
                bus="Bus {}".format(i),
                p_nom=random.randint(600,800),
                marginal_cost=1)

#loads
for i in range(12,30):
    network.add("Load",f"Load {i}",
                bus=f"Bus {i}",
                p_set=random.randint(25,125))
initial_lopf_status = network.lopf(pyomo=False,solver_name='gurobi',solver_options = {'OutputFlag': 0,'SOLUTION_LIMIT':1},solver_logfile=None,store_basis = True)
if initial_lopf_status[0] != 'ok':
  print(initial_lopf_status)
  raise ValueError('The original network is not feasible')
network.export_to_netcdf("60line.nc")