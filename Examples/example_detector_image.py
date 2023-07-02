"""
Dans_Diffraction Examples
Calcualte a detector image
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt  # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

dif.fg.nice_print()
energy_kev = 8.0
detector_distance = 1000  # mm

# Structure
# xtl = dif.structure_list.NaCl()
xtl = dif.structure_list.Na08CoO2_P63mmc()

# Reflection
tth = xtl.Cell.tth([0, 0, 2], energy_kev=energy_kev)[0]
# Orient crystal to scattering position
xtl.Cell.orientation.rotate_6circle(mu=tth/2)
# Rotate detector to two-theta
delta, gamma = 0, tth

# Generate detector image
xx, yy, mesh, reflist = xtl.Scatter.detector_image(
    detector_distance_mm=detector_distance,
    delta=delta,
    gamma=gamma,
    height_mm=400.,
    width_mm=100.,
    pixel_size_mm=0.1,
    energy_range_ev=1.,
    peak_width_deg=0.5,
    wavelength_a=dif.fc.energy2wave(energy_kev),
    background=0
)

# Plot detector image
plt.figure()
plt.pcolormesh(xx, yy, mesh, vmin=0, vmax=1, shading='auto')
plt.xlabel('x-axis [mm]')
plt.ylabel('z-axis [mm]')
plt.axis('image')
# reflection labels
for n in range(len(reflist['hkl'])):
    plt.text(reflist['detx'][n], reflist['dety'][n], reflist['hkl_str'][n], c='w')

# Scan detector image
mu_range = np.arange((tth/2) - 1, (tth/2) + 1, 0.02)
inten = np.zeros_like(mu_range)
for n, mu in enumerate(mu_range):
    xtl.Cell.orientation.rotate_6circle(mu=mu)
    xx, yy, mesh, reflist = xtl.Scatter.detector_image(
        detector_distance_mm=detector_distance,
        delta=delta,
        gamma=gamma,
        height_mm=400.,
        width_mm=100.,
        pixel_size_mm=0.1,
        energy_range_ev=2.,
        peak_width_deg=0.2,
        wavelength_a=dif.fc.energy2wave(energy_kev),
        background=0
    )
    inten[n] = np.sum(mesh)

plt.figure()
plt.plot(mu_range, inten)
plt.xlabel('Mu [Deg]')
plt.ylabel('Intensity')
plt.title('%s [0,0,2]' % xtl.name)

plt.show()


