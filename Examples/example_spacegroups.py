"""
Dans_Diffraction Examples
Load space groups and look at the information contained.
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf, '..'))
import Dans_Diffraction as dif

# Print all spacegroups
print(dif.fc.spacegroup_list(range(1,231)))

# Load spacegroup dict
sg = dif.fc.spacegroup(194)
print(sg['space group number'])
print(sg['space group name'])
print(sg['general positions'])
print(sg['positions wyckoff letter'])

# Perfrom symmetry operations
sym = dif.fc.gen_sym_pos(sg['general positions'], 0.1, 0, 0)
print(sym)
