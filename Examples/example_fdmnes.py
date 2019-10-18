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

# Activate FDMNES
dif.fdmnes_checker() # returns true/false
dif.activate_fdmnes()


xtl = dif.Crystal(f)

dif.classes_gui.RunFDMNESgui(xtl)

fdm = dif.Fdmnes(xtl) # this might take a while the first time as the fdmnes_win64.exe file is found

fdm.setup(
    folder_name='Test', # this will create the directory /FDMNES/Sim/Test, but if it exists Test_2 will be used etc.
    comment='A test run',
    radius=4.0,
    edge='L3',
    absorber='Ru',
    scf=False,
    quadrupole=False,
    azi_ref=[0,1,0],
    hkl_reflections=[[1,0,0],[0,1,3],[0,0,3]]
)

# Create files and run FDMNES
fdm.create_files()
fdm.write_fdmfile()
fdm.run_fdmnes() # This will take a few mins, output should be printed to the console

# Analysis
ana = fdm.analyse()
ana.xanes.plot()
ana.density.plot()
ana.I100sp.plot3D()
