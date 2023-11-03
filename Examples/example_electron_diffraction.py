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
electron_wavelength = dif.fc.electron_wavelength(electron_energy)


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

# Plot Scattering factors
dif.fp.set_plot_defaults()
plt.figure()
plt.plot(q_range, xsf / xsf[0], label='X-Ray')
plt.plot(q_range, esf / esf[0], label='Electron')
plt.legend()
plt.xlabel('Q [A$^-1$]')
plt.ylabel('Relative Scattering factor')
plt.title(element)

# Plot LEED detector
detector_distance = 50  # mm
energy_ev = 100
# Generate detector image
#xtl.Cell.orientation.rotate_6circle(chi=45, mu=57)
xx, yy, mesh, reflist = xtl.Scatter.detector_image(
    detector_distance_mm=detector_distance,
    delta=0,
    gamma=180,
    height_mm=400.,
    width_mm=400.,
    pixel_size_mm=0.1,
    energy_range_ev=1000 * dif.fc.wave2energy(dif.fc.electron_wavelength(200)),
    peak_width_deg=5,
    wavelength_a=dif.fc.electron_wavelength(energy_ev),
    background=0,
    min_intensity=0.001
)
plt.figure()
plt.pcolormesh(xx, yy, mesh, vmin=0, vmax=.001, shading='auto')
plt.xlabel('x-axis [mm]')
plt.ylabel('z-axis [mm]')
plt.title('LEED Image %.0f eV' % energy_ev)
plt.axis('image')
# reflection labels
for n in range(len(reflist['hkl'])):
    plt.text(reflist['detx'][n], reflist['dety'][n], reflist['hkl_str'][n], c='w')
plt.show()

