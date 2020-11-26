"""
Dans_Diffraction Examples
Build a supercell from multiple unit cells and plot the layers
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif


xtl = dif.structure_list.Ca2RuO4()

# Generate the superstructure, repeating the parent/average structure to fill the supercell
size = 20
P = [[size, 0, 0], [0, size, 0], [0, 0, 1]]
sup = xtl.generate_superstructure(P)

# Remove certain atoms
all_ru = np.where(sup.Atoms.type == 'Ru')[0]
remove_percentage = 0.1
rem_ru = np.random.choice(all_ru, int(remove_percentage * len(all_ru)), replace=False)
#sup.Structure.occupancy[rem_ru] = 0  # remove atom
sup.Atoms.type[rem_ru] = 'Mn'
sup.Atoms.label[rem_ru] = 'Mn1'
sup.generate_structure()
#sup.Structure.occupancy
print(sup.Properties.molname('Ca', 2))

# Plot the structure in 2D layers
sup.Plot.plot_layers(layers=[0, 0.5], layer_width=0.01)
plt.show()

# Write cif
sup.write_cif(f'Ca2RuMnO4_supercell_{size}x{size}_Mn{remove_percentage}.cif')
