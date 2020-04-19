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

# Filepath
ciffile = '../Dans_Diffraction/Structures/LiCoO2.cif'

xtl = dif.Crystal(ciffile)

# Change lattice parameters
xtl.Cell.latt([2.85, 2.85, 10.8, 120])

# Change atomic parameters
xtl.Atoms.changeatom(1, occupancy=0.5)
xtl.generate_structure()

# write cif file
xtl.write_cif('test.cif', comments='This is a test!')