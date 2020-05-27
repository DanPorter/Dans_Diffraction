"""
Dans_Diffraction Examples
Read values from a Crystallographic Information File (.cif or .mcif), edit the structure, write a different file
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif

#f = cf+'/../Dans_Diffraction/Structures/Ca2RuO4.cif'
xtl = dif.structure_list.Ca2RuO4()
xtl.Atoms.changeatom(1, mxmymz=[0, 3, 0.3])
xtl.generate_structure()

# write cif file
xtl.write_cif('test.mcif', comments='This is a test!')