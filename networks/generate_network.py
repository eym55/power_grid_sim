import pypsa
import numpy as np
import random
import matplotlib.pyplot as plt, mpld3
random.seed(69)
network = pypsa.Network()
for i in range(8):
    network.add("Bus","Bus {}".format(i))
network.buses.at[network.buses.index[0], 'x'] = 0
network.buses.at[network.buses.index[0], 'y'] = 5

network.buses.at[network.buses.index[1], 'x'] = 6
network.buses.at[network.buses.index[1], 'y'] = 5

network.buses.at[network.buses.index[2], 'x'] = 9
network.buses.at[network.buses.index[2], 'y'] = 5

network.buses.at[network.buses.index[3], 'x'] = 3
network.buses.at[network.buses.index[3], 'y'] = 2

network.buses.at[network.buses.index[4], 'x'] = 0
network.buses.at[network.buses.index[4], 'y'] = 0

network.buses.at[network.buses.index[5], 'x'] = 3
network.buses.at[network.buses.index[5], 'y'] = 0

network.buses.at[network.buses.index[6], 'x'] = 6
network.buses.at[network.buses.index[6], 'y'] = 0

network.buses.at[network.buses.index[7], 'x'] = 9
network.buses.at[network.buses.index[7], 'y'] = 0

edges = [(0,1),(0,3),(0,4),(0,5),(1,3),(1,7),(1,2),(2,6),(2,7),(3,4),(3,5),(3,6),(5,6),(6,7)]
for i in range(len(edges)):
    network.add("Line","Linine {}".format(i),
                bus0="Bus {}".format(edges[i][0]),
                bus1="Bus {}".format(edges[i][1]),
                x=0.0001,
                s_nom=60)

#add a generator at bus 2
network.add("Generator","Gen 2",
            bus="Bus 2",
            p_nom=300,
            marginal_cost=random.randint(25,75))
#add a generator at bus 3
network.add("Generator","Gen 3",
            bus="Bus 3",
            p_nom=300,
            marginal_cost=random.randint(25,75))
#add a generator at bus 4
network.add("Generator","Gen 4",
            bus="Bus 4",
            p_nom=300,
            marginal_cost=random.randint(25,75))

for i in [0,1,5,6,7]:
    network.add("Load",f"Load {i}",
                bus=f"Bus {i}",
                p_set=random.randint(25,125))

network.export_to_netcdf("lopf_grid.nc")