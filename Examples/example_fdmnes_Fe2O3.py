"""
Dans_Diffraction Examples
Create FDMNES files, run program and plot results
Generate the FDMNES Test file 'Fe2O3_inp.txt', run this file and plot the results
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

# Activate FDMNES
if not dif.fdmnes_checker():  # checks existence of executable path file
    dif.activate_fdmnes()  # Finds the fdmnes executable, only need to be run once.

# Build Fe2O3 Crystal
xtl = dif.Crystal()
xtl.name = 'Fe2O3'
xtl.Cell.latt(5.4135, 5.4135, 5.4135, 55.283, 55.283, 55.283)
Fe = [
    [0.105, 0.105, 0.105],
    [0.395, 0.395, 0.395],
    [0.605, 0.605, 0.605],
    [0.895, 0.895, 0.895],
]
for u, v, w in Fe:
    xtl.Atoms.addatom(type='Fe', label='Fe', u=u, v=v, w=w)
Ox = [
    [0.292, 0.708, 0.000],
    [0.708, 0.000, 0.292],
    [0.000, 0.292, 0.708],
    [0.208, 0.792, 0.500],
    [0.792, 0.500, 0.208],
    [0.500, 0.208, 0.792],
]
for u, v, w in Ox:
    xtl.Atoms.addatom(type='O', label='O', u=u, v=v, w=w)
xtl.Atoms.removeatom(0)  # remove orinal atom in
xtl.generate_structure()
print(xtl)

fdm = dif.Fdmnes(xtl)

fdm.setup(
    folder_name='Fe2O3',  # this will create the directory /FDMNES/Sim/Test, but if it exists Test_2 will be used etc.
    comment='Calculation in Hematite to see the Finkelstein effect on the (111)sigma-pi reflection.',
    radius=3.0,
    energy_range=' -7.  0.1  7.  0.5  12. 1. 20. 2. 50.',
    edge='K',
    absorber='Fe',
    scf=False,
    quadrupole=True,
    azi_ref=[1, 0, 0],
    hkl_reflections=[[1, 1, 1], [2, 2, 2]]
)

# Create files and run FDMNES
#fdm.create_files()
#fdm.write_fdmfile()
#fdm.run_fdmnes()  # This will take a few mins, output should be printed to the console

# Analysis
ana = fdm.analyse('Fe2O3')
print(ana)
ana.xanes.plot()
ana.density.plot()
for ref in ana:
    ref.plot3D()
plt.show()

