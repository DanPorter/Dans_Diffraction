"""
Dans_Diffraction Examples
Generate powder spectrum from a cif file
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt  # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

f = cf + '/../Dans_Diffraction/Structures/Diamond.cif'

xtl = dif.Crystal(f)

energy_kev = dif.fc.wave2energy(1.5498)  # 8 keV
max_twotheta = 180

xtl.Scatter.setup_scatter('xray')  # 'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
max_wavevector = dif.fc.calqmag(max_twotheta, energy_kev)
q, intensity = xtl.Scatter.generate_powder(max_wavevector, peak_width=0.01, background=0, powder_average=True)

# convert wavevector, q=2pi/d to two-theta:
twotheta = dif.fc.cal2theta(q, energy_kev)

# save data as csv file
head = '%s\nx-ray powder diffraction energy=%6.3f keV\n two-theta, intensity' % (xtl.name, energy_kev)
np.savetxt('powder.csv', (twotheta, intensity), delimiter=',', header=head)

# plot the spectra
plt.figure()
plt.plot(twotheta, intensity, '-', lw=2)
dif.fp.labels('x-ray powder diffraction E=%6.3f keV\n%s' % (energy_kev, xtl.name), 'two-theta', 'intensity')
plt.show()
