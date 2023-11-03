"""
Dans_Diffraction Examples
Build your very own crystal structure,
using lattice parameters and atomic coordinates
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

xtl = dif.Crystal()

xtl.name = 'Oh! What a Lovely Crystal'
xtl.new_cell([2.8,2.8,6.0,90,90,90])
#xtl.new_atoms(u=[0,0.5], v=[0,0.5], w=[0,0.25], type=['Na','O'], label=['Na1','O1'], occupancy=None, uiso=None, mxmymz=None)
xtl.Atoms.changeatom(0,u=0,   v=0,   w=0,    type='Na',label='Na1',occupancy=None, uiso=None, mxmymz=None) # there is an Fe ion added by default
xtl.Atoms.addatom(u=0.5, v=0.5, w=0.25, type='O', label='O1', occupancy=None, uiso=None, mxmymz=None)
#xtl.Symmetry.addsym('x,y,z+1/2')
xtl.Symmetry.load_spacegroup(7)
xtl.generate_structure() # apply symmetry to atomic positions.

print(xtl.info())
xtl.Plot.plot_crystal()
plt.show()
