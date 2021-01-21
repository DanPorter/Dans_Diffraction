"""
Dans_Diffraction Examples
Calculate reflection intensity with x-ray resonant dispersion corrections
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

f = cf+'/../Dans_Diffraction/Structures/LiCoO2.cif'

xtl = dif.Crystal(f)

energy = np.arange(6, 10, 0.01)

# Select smallest non-zero reflections
all_hkl = xtl.Cell.all_hkl(energy[0])
all_hkl = xtl.Symmetry.remove_symmetric_reflections(all_hkl)
sf2 = xtl.Scatter.x_ray(all_hkl)
sort_idx = np.argsort(sf2)
all_hkl = all_hkl[sort_idx]
sf2 = sf2[sort_idx]
hkl = all_hkl[sf2 > 0.1][:5]
#hkl = [[1, 2, 3], [2, 0, 1], [0, 0, 2]]

# Calculate squared-structure factor using x-ray form factors with dispersion corrections
inten = xtl.Scatter.xray_dispersion(hkl, energy)

dif.fp.newplot(energy, inten[0], label=hkl[0])
for n in range(1, len(hkl)):
    plt.plot(energy, inten[n], lw=2, label=hkl[n])
dif.fp.labels('%s' % xtl.name, 'Energy [keV]', '|SF|$^2$', legend=True)

# Shortcut!
# xtl.Plot.plot_xray_resonance(hkl, xtl.Properties.Co.K)
plt.show()
