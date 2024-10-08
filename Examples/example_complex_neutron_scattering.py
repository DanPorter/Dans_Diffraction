"""
Dans_Diffraction Examples
Calculate neutron scattering using complex neutron scattering lengths for isotopes
"""

import numpy as np
import Dans_Diffraction as dif

print(dif.version_info())

# Neutron scattering length table
nsl = dif.fc.read_neutron_scattering_lengths()

# complex neutron scattering length in fm for isotope
b_Ti = dif.fc.neutron_scattering_length('46-Ti')
print(f"46-Ti: {b_Ti[0]:.3f}")

# Add isotopes to structure
xtl = dif.structure_list.LiCoO2()
xtl.Atoms.changeatom(0, type='7-Li')
xtl.generate_structure()
print(xtl)

isotope_intensity = xtl.Scatter.intensity([0,0,6], scattering_type='neutron')
natural_intensity = xtl.Scatter.neutron([0,0,6])  # the function neutron still uses the old atom_properties

print(f"Natural intensity [0,0,6]: {natural_intensity[0]:.2f}")
print(f"Isotope intensity [0,0,6]: {isotope_intensity[0]:.2f}")

