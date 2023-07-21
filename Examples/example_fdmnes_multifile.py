"""
Dans_Diffraction Examples
Create FDMNES files, run program and plot results
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


f = cf+'/../Dans_Diffraction/Structures/Ca2RuO4.cif'

xtl = dif.Crystal(f)


fdm = dif.Fdmnes(xtl)  # this might take a while the first time as the fdmnes_win64.exe file is found

fdm.setup(
    comment='A test run',
    edge='L3',
    absorber='Ru',
    scf=False,
    quadrupole=False,
    azi_ref=[0, 1, 0],
    hkl_reflections=[[1, 0, 0], [0, 1, 3], [0, 0, 3]]
)

run_files = []
radii = np.arange(4, 7, 1)
for radius in radii:
    fdm.setup(folder_name='CRO_Radius_%1.0f' % radius, radius=radius)
    fdm.create_files()
    run_files += [fdm.generate_input_path()]

# Create files and run FDMNES
fdm.write_fdmfile(run_files)
fdm.run_fdmnes()  # This will take a few mins, output should be printed to the console

# Analysis
energy = []
xanes = []
for radius in radii:
    ana = fdm.analyse('CRO_Radius_%1.0f' % radius)
    energy += [ana.xanes.energy]
    xanes += [ana.xanes.intensity]

dif.fp.multiplot(energy, xanes, radii, cmap='jet')
dif.fp.labels(xtl.name, 'E [eV]', 'XANES')
