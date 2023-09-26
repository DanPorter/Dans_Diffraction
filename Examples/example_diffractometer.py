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

dif.fg.nice_print()
dif.fp.set_plot_defaults()


f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'

xtl = dif.Crystal(f)

# Parameters
energy_kev = 8.0
resolution_ev = 200
detector_distance = 565  # mm, I16 Pilatus_100K
det_height = 34
det_width = 67
pixel_size = 172  # um

delta_value = xtl.Cell.tth([0, 0, 4], energy_kev=energy_kev)[0]
chi_value = 90
phi_value = 0
eta_value = delta_value / 2.
phi_scan = np.arange(-180, 180, 1)
eta_scan = np.linspace(eta_value-1, eta_value+1, 101)

# Orient crystal to scattering position
xtl.Cell.orientation.rotate_6circle(eta=eta_value, chi=chi_value, phi=phi_value)

# Generate detector image
xx, yy, mesh, reflist = xtl.Scatter.detector_image(
    detector_distance_mm=detector_distance,
    delta=delta_value,
    gamma=0,
    height_mm=det_height,
    width_mm=det_width,
    pixel_size_mm=pixel_size/1000.,
    energy_range_ev=1.,
    peak_width_deg=0.1,
    wavelength_a=dif.fc.energy2wave(energy_kev),
    background=0
)

# Plot detector image
plt.figure()
plt.pcolormesh(xx, yy, mesh, vmin=0, vmax=1, shading='auto')
plt.xlabel('x-axis [mm]')
plt.ylabel('z-axis [mm]')
ttl = '%s E=%.4g keV' % (xtl.name, energy_kev)
ttl += '\nphi=%.4g, chi=%.4g, eta=%.4g, delta=%.4g' % (phi_value, chi_value, eta_value, delta_value)
plt.title(ttl)
plt.axis('image')
# reflection labels
for n in range(len(reflist['hkl'])):
    plt.text(reflist['detx'][n], reflist['dety'][n], reflist['hkl_str'][n], c='w')


" --- Scan detector image --- "
inten = np.zeros_like(eta_scan)
for n, eta_value in enumerate(eta_scan):
    xtl.Cell.orientation.rotate_6circle(eta=eta_value, chi=chi_value, phi=phi_value)
    xx, yy, mesh, reflist = xtl.Scatter.detector_image(
        detector_distance_mm=detector_distance,
        delta=delta_value,
        gamma=0,
        height_mm=det_height,
        width_mm=det_width,
        pixel_size_mm=pixel_size / 1000.,
        energy_range_ev=1.,
        peak_width_deg=0.1,
        wavelength_a=dif.fc.energy2wave(energy_kev),
        background=0
    )
    inten[n] = np.sum(mesh)

plt.figure()
plt.plot(eta_scan, inten)
plt.xlabel('Eta [Deg]')
plt.ylabel('Intensity')
plt.title('%s [0,0,4]' % xtl.name)

plt.show()
