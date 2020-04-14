"""
Dans_Diffraction Examples
Plot powder pattern from a crystal
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'

xtl = dif.Crystal(f)

xtl.Plot.simulate_powder(energy_kev=8, peak_width=0.01, background=0, powder_average=True)
plt.show()