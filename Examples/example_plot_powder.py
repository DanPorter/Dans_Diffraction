"""
Dans_Diffraction Examples
Plot powder pattern from a crystal
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt  # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

f = cf + '/../Dans_Diffraction/Structures/Diamond.cif'

xtl = dif.Crystal(f)

xtl.Plot.simulate_powder(energy_kev=8, peak_width=0.01, background=0, powder_average=True)
#plt.show()

# Manual plotting
xtl.Scatter.setup_scatter(energy_kev=8, min_twotheta=10, max_twotheta=130, scattering_type='xray', powder_units='tth')
tth, inten, reflections = xtl.Scatter.powder(units='tth')

plt.figure()
plt.plot(tth, inten)
for h, k, l, x, y in reflections:
    if y > inten.min() + 1:
        plt.text(x, y, dif.fc.hkl2str((h,k,l)), c='k')
    else:
        plt.text(x, y + 1, dif.fc.hkl2str((h, k, l)), c='r')
plt.xlabel('Two-Theta [Deg]')
plt.ylabel('Intensity')
plt.title('%s 8 keV' % xtl.name)
plt.show()
