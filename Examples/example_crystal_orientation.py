"""
Dans_Diffraction Examples
Set and change the orientation of the crystal
Makes use of the new orientation functionality in Dans_Diffraction 2.1.0+

Requires Dans_Diffraction version >2.1.0
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt  # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif
print(dif.version_info())
dif.fg.nice_print()

# ---Settings---
f = cf+'/../Dans_Diffraction/Structures/LiCoO2.cif'

# Create Crystal
xtl = dif.Crystal(f)

# Set orientation
xtl.Cell.orientation.orient(a_axis=[1, 0, 0], c_axis=[0, 0, 1])

# Rotate using 6-circle diffractometer axes
xtl.Cell.orientation.rotate_6circle(chi=90, eta=20)

# Switch to lab frame
xtl.Cell.orientation.set_lab([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])

print('Unit Vectors:')
print(xtl.Cell.UV())
print('Reciprocal vectors:')
print(xtl.Cell.UVstar())
print('Location of (1,1,0)')
print(xtl.Cell.calculateQ([1, 1, 0]))

# Plot oriented crystal
xtl.Plot.plot_crystal()
plt.show()
