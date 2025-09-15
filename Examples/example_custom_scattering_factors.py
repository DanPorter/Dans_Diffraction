"""
Dans_Diffraction Examples
Example use of custom scattering factors
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

f = cf+'/../Dans_Diffraction/Structures/LiCoO2.cif'
xtl = dif.Crystal(f)

# add custom element
coefs = (12.2841,  4.5791,  7.4409,  0.6784,  4.2034, 12.5359,  2.2488, 72.1692,  1.1118,  0.)
table = np.array([
    [7.7, 7.71, 7.72, 7.73, 7.74, 7.75, 7.76, 7.77, 7.78, 7.79],  # keV
    [-9.6866, -11.0362, -10.1969, -9.3576, -8.5184, -7.6791, -6.8398, -6.0005, -5.1613, -4.322],  # f1
    [-0.4833, -3.7933, -3.7862, -3.7791, -3.772, -3.765, -3.7579, -3.7508, -3.7437, -3.7367]  # f2
])
dif.fc.add_custom_form_factor_coefs('Co3+', *coefs, dispersion_table=table)
xtl.Atoms.changeatom(1, type='Co3+')
xtl.generate_structure()
xtl.Structure.type[3] = 'Co'


# Display scattering factors
print(xtl.Scatter.print_scattering_factor_coefficients())

xtl.Scatter.setup_scatter(scattering_type='custom', output=False)
xtl.Plot.plot_scattering_factors(q_max=6)

# Calculate structure factors
en = np.arange(7.7, 7.8, 0.01)
ref = [0, 0, 3]
inten1 = xtl.Scatter.xray_dispersion(ref, en)
inten2 = xtl.Scatter.intensity(ref, 'custom', energy_kev=en)

plt.figure()
plt.plot(en, inten1, '-', label='original')
plt.plot(en, inten2[0], '-', label='custom')
plt.xlabel('Energy [keV]')
plt.ylabel('intensity')
plt.legend()
plt.show()

energy_kev = np.arange(7, 8, 0.2)
q_mag = np.arange(3, 4, 0.1)
elements = ['Co', 'O']
print(f"Shape should be: ({len(q_mag)}, {len(elements)}, {len(energy_kev)})")
qff = dif.fc.custom_scattering_factor(elements, q_mag, energy_kev)
print(qff.shape)
