"""
Dans_Diffraction Examples
Generate powder spectrum from a cif file with a custom peak profile

Requires: lmfit
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

from lmfit import lineshapes

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif
dif.fg.nice_print()

#f = cf + '/../Dans_Diffraction/Structures/Diamond.cif'  # small cell - few reflections
f = cf + '/../Dans_Diffraction/Structures/Sr3LiRuO6.cif'  # large cell - many reflections

xtl = dif.Crystal(f)

energy_kev = 8

#tth1, intensity1 = xtl.Scatter.powder(scattering_type='x-ray', units='twotheta', peak_width=0.01, background=0,
#                                      pixels=None, powder_average=None, lorentz_fraction=None, custom_peak=None)
xtl.Scatter.setup_scatter(
    scattering_type='x-ray',
    powder_units='twotheta',
    powder_overlap=5,
    energy_kev=energy_kev
)

# Default Gaussian / Lorentzian peaks
tth1, intensity1, ref1 = xtl.Scatter.powder(lorentz_fraction=0.5)
tth2, intensity2, ref2 = xtl.Scatter.powder(lorentz_fraction=0)
tth3, intensity3, ref3 = xtl.Scatter.powder(lorentz_fraction=1)


# Custom peak
peak_x = np.arange(-10, 10, 0.1)  # must be centred about 0, step size vs sigma determines peak width
peak_y = lineshapes.skewed_voigt(peak_x, amplitude=1, center=0, sigma=0.2, gamma=0.5, skew=0.5)
peak_y = peak_y / np.max(peak_y)  # peak should have height 1
tth4, intensity4, ref4 = xtl.Scatter.powder(custom_peak=peak_y)

# plot the spectra
plt.figure(figsize=[12, 10], dpi=60)
plt.plot(tth1, intensity1, '-', lw=2, label='Lorentz fraction=0.5')
plt.plot(tth2, intensity2, '-', lw=2, label='Lorentz fraction=0.0')
plt.plot(tth3, intensity3, '-', lw=2, label='Lorentz fraction=1.0')
plt.plot(tth4, intensity4, '-', lw=2, label='Custom peak')

for ref in ref1:
    if ref[4] > 100:
        plt.text(ref[3], ref[4], '(%1.0f,%1.0f,%1.0f)' % (ref[0], ref[1], ref[2]))

dif.fp.labels('x-ray powder diffraction E=%6.3f keV\n%s' % (energy_kev, xtl.name), 'two-theta', 'intensity', legend=True)
plt.show()
