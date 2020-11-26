"""
Dans_Diffraction Examples
Calculate exchange path between magnetic atoms
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif

f = cf+'/../Dans_Diffraction/Structures/Sr3LiRuO6_C2\'c\'.mcif' # from isodistort R-3c

xtl = dif.Crystal(f)

exchange_paths, exchange_distances, exchange_str = xtl.Properties.exchange_paths(
    cen_idx=None,                   # Centre on first magnetic ion
    nearest_neighbor_distance=7.0,  # Distance to search for neighbors
    exchange_type='O',              # Only connect to O ions
    search_in_cell=True,            # Only look for neighbors within cell
    group_neighbors=True,           # Group neighbors by bond distance
    disp=True,                     # print calculation details
    return_str=True                 # return exchange_str
)

print(exchange_str)

xtl.Plot.plot_exchange_paths(
    cen_idx=None,                   # Centre on first magnetic ion
    nearest_neighbor_distance=7.0,  # Distance to search for neighbors
    exchange_type='O',              # Only connect to O ions
    search_in_cell=True,            # Only look for neighbors within cell
    group_neighbors=True,           # Group neighbors by bond distance
    disp=False,                     # print calculation details
)
plt.show()