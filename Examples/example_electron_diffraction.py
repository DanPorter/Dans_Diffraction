"""
Dans_Diffraction Examples
Calculate diffraction using electron form factors
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif


f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'

xtl = dif.Crystal(f)

electron_energy = 200  # eV
electron_wavelength = dif.fc.electron_wavelenth(electron_energy)


xtl.Scatter.setup_scatter(scattering_type='electron', wavelength_a=electron_wavelength)
print("Reflections")
print(xtl.Scatter.print_all_reflections(print_symmetric=False, min_intensity=0.01, units='tth'))
print("Extinctions")
print(xtl.Scatter.print_all_reflections(print_symmetric=False, min_intensity=None, max_intensity=0.01, units='tth'))

# Plot Electron scattering factor
element = 'C'
q_range = np.arange(0, 5, 0.01)
xsf = dif.fc.xray_scattering_factor(element, q_range)
esf = dif.fc.electron_scattering_factor(element, q_range)

dif.fp.set_plot_defaults()
plt.figure()
plt.plot(q_range, xsf / xsf[0], label='X-Ray')
plt.plot(q_range, esf / esf[0], label='Electron')
plt.legend()
plt.xlabel('Q [A$^-1$]')
plt.ylabel('Relative Scattering factor')
plt.title(element)
plt.show()

