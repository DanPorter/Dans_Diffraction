"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


f1 = cf+'/../Dans_Diffraction/Structures/KCl.cif'
f2 = cf+'/../Dans_Diffraction/Structures/Na0.8CoO2_P63mmc.cif'

kcl = dif.Crystal(f1)
nacoo2 = dif.Crystal(f2)

# kcl.scale = 100.

xtls = nacoo2 + kcl
xtls.setup_scatter(energy_kev=5.0)

# Plot Powder
xtls.Plot.simulate_powder(peak_width=0.001)

# Plot hk0
xtls.Plot.simulate_intensity_cut([[1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]], centre=[0, 0, 0])
plt.show()

print("Reflections")
print(xtls.print_all_reflections(print_symmetric=False))
print("Extinctions")
print(xtls.print_all_reflections(max_intensity=0.01, print_symmetric=False))
