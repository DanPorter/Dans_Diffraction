"""
Dans_Diffraction Examples
Simulate a powder diffraction pattern from a large structrue
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt  # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

dif.fg.nice_print()
dif.fp.set_plot_defaults()

xtl = dif.structure_list.KCl()  # BCC

# Generate random lattice with some distortions
lat = xtl.generate_lattice(20, 20, 2)
lat.name = 'KCl Superstructure'
lat.scale = 20 * 20 * 2  # makes intensities comparable to parent
for n in range(len(lat.Structure.u)):
    lat.Structure.w[n] = lat.Structure.w[n] + 0.1 * np.sin(2 * np.pi * lat.Structure.u[n])

# Plot reciprocal space plane
plt.rcParams.update({'lines.linewidth': 0.1})
print('Intensity cut')
lat.Plot.simulate_intensity_cut(centre=[0, 0, 0])  # this is pretty slow
# Alternate - plot envelope cut
# this is faster for very large cells (less accurate though)
print('envelope cut')
lat.Plot.simulate_envelope_cut(centre=[0, 0, 0], pixels=101)

lat.Scatter.setup_scatter(energy_kev=8, min_twotheta=10, max_twotheta=60,
                          scattering_type='xray', output=False)
print('powder')
# run with save='file.npy' the first time
tth, inten, reflections = lat.Scatter.powder(units='tth', peak_width=0.01, save='sf.npy')
# load structure factors from 'sf.npy' to save the lengthy calculation
tth, inten, reflections = lat.Scatter.powder(units='tth', peak_width=0.01, load='sf.npy')

tth_xtl, inten_xtl, reflections_xtl = xtl.Scatter.powder(units='tth', peak_width=0.01, energy_kev=8, max_twotheta=60)

# plot the spectra
plt.figure()
plt.plot(tth_xtl, inten_xtl, '-', lw=1, label='Crystal')
plt.plot(tth, inten, '-', lw=0.5, label='Superstructure')

plt.xlabel('Two-Theta [Deg]')
plt.ylabel('Intensity')
plt.title('Powder pattern')
plt.legend()
plt.show()

