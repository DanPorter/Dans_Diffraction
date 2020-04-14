"""
Dans_Diffraction Examples
Turn a reflection list from single crystal experiment to a powder pattern
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'

xtl = dif.Crystal(f)

en = 8.0 # keV

# create random reflection list (as if from a diffractometer)
hkl = np.random.randint(-4, 5, [200, 3])
inten = xtl.Scatter.intensity(hkl)
ninten = [np.random.normal(i, np.sqrt(i+0.1)) for i in inten]

# Average symmetric reflections
chkl, cinten = xtl.Scatter.powder_correction(hkl, ninten)
cq = xtl.Cell.Qmag(chkl)

# Calculate completeness
maxq = dif.fc.calqmag(180, en)
all_hkl = xtl.Cell.all_hkl(en)
sym_hkl = xtl.Symmetry.symmetric_reflections_unique(chkl[cq<maxq, :])
print('Completeness = %d / %d = %6.2f %%' % (len(sym_hkl), len(all_hkl), 100*(len(sym_hkl)/len(all_hkl))))

# Create powder grid
grid_q, grid_i = dif.fg.grid_intensity(cq, cinten, peak_width=0.01)
grid_tth = dif.fc.cal2theta(grid_q, en)

# Simulate powder for comparison
cal_q, cal_i = xtl.Scatter.generate_powder(en, peak_width=0.005)
cal_tth = dif.fc.cal2theta(cal_q, en)

# Create plot
plt.figure(figsize=[14,8], dpi=60)
plt.plot(cal_tth, cal_i, 'k-', lw=1, label='Full Crystal Simulation')
plt.plot(grid_tth, grid_i, 'b-', lw=3, label='From reflections')
dif.fp.labels('%s E = %s keV' % (xtl.name, en), 'Two-Theta [Deg]', 'Intensity [a. u.]', legend=True)
plt.show()
