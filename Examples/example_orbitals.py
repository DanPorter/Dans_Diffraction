"""
Dans_Diffraction Examples
Display atomic orbitals
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf,'..'))
import Dans_Diffraction as dif

xtl = dif.structure_list.Ca2RuO4()
orbitals = xtl.Properties.orbitals()
print(xtl.name)
print(orbitals)

s = 'Sr4Ru2.6Mn0.4O10'
compound = dif.classes_orbitals.CompoundString(s)
print('\n%s' % s)
print(compound)
print('Total Charge: %1.3f' % compound.check_charge())

s = 'Li0.8CoO2'
compound = dif.classes_orbitals.CompoundString(s)
print('\n%s' % s)
print(compound)
print('Total Charge: %1.3f' % compound.check_charge())

s = '(Na0.9Ca0.1)1.6Co2O4'
compound = dif.classes_orbitals.CompoundString(s)
print('\n%s' % s)
print(compound)
print('Total Charge: %1.3f' % compound.check_charge())

