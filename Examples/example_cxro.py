"""
Dans_Diffraction Examples
Simulate the functionality of the CXRO website: https://henke.lbl.gov/optical_constants/
Use formulas from:
B.L. Henke, E.M. Gullikson, and J.C. Davis, X-ray interactions: photoabsorption, scattering, transmission,
and reflection at E=50-30000 eV, Z=1-92, Atomic Data and Nuclear Data Tables 54 no.2, 181-342 (July 1993).
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif


formula = 'Ca2RuO4'
density = 4.583  # g/cm^3
energy_range = np.arange(0.03, 15, 0.01)  # keV
grazing_angle = 90  # Deg
slab_thickness = 0.2  # microns

# Attenuation length
dif.fp.plot_xray_attenuation_length(formula, density, energy_range, grazing_angle)

# Transmission
dif.fp.plot_xray_transmission(formula, density, energy_range, slab_thickness)

plt.show()


