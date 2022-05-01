import pypsa
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(69)
network = pypsa.Network()

for i in range(15):
    network.add("Bus","Bus {}".format(i))

for i in range(15):
    network.buses.at[network.buses.index[i], 'x'] = random.randint(0,100)
    network.buses.at[network.buses.index[i], 'y'] = random.randint(0,100)

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
                s_nom=60)

genBus = [2,3,9,10,13,14]
loadBus = [0,1,4,5,6,7,8,11,12]

for i in genBus:
    network.add("Generator","Gen {}".format(i),
                bus="Bus {}".format(i),
                p_nom=300,
                marginal_cost=random.randint(25,75))

for i in loadBus:
    network.add("Load",f"Load {i}",
                bus=f"Bus {i}",
                p_set=random.randint(25,125))

network.export_to_netcdf("30line.nc")