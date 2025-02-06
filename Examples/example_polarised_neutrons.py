"""
Dans_Diffraction Examples
Calculate magnetic neutron scattering using polarised neutrons
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif


#f = cf+'/../Dans_Diffraction/Structures/MnO.mcif'
f = dif.structure_list.MnO.filename

xtl = dif.Crystal(f)

wl = 1.5498
options = {
    'units': 'Q',
    'wavelength_a': wl,
    'peak_width': 0.02,
    'background': 0,
}
xtl.Scatter.setup_scatter(max_twotheta=40, output=False)

q1, intensity1, reflections1 = xtl.Scatter.powder(scattering_type='neutron', **options)
q2, intensity2, reflections2 = xtl.Scatter.powder(scattering_type='neutron magnetic', **options)
xtl.Scatter.setup_scatter(polarisation_vector=[1, 0, 0])
q3, intensity3, reflections3 = xtl.Scatter.powder(scattering_type='neutron polarised', **options)
xtl.Scatter.setup_scatter(polarisation_vector=[0, 0, 1])
q4, intensity4, reflections4 = xtl.Scatter.powder(scattering_type='neutron polarised', **options)


plt.figure(figsize=[14, 8], dpi=60)
plt.plot(q1, intensity1, '-', lw=2, label='neutron')
plt.plot(q2, intensity2, '-', lw=2, label='neutron magnetic')
plt.plot(q3, intensity3, '-', lw=2, label='neutron polarised || a*')
plt.plot(q4, intensity4, '-', lw=2, label='neutron polarised || c*')
dif.fp.labels('neutron powder diffraction MnO', 'Q ($\AA^{-1}$)', 'intensity', legend=True)

plt.show()
