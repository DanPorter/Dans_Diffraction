"""
Dans_Diffraction Examples
Load space groups and look at the information contained.
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf, '..'))
import Dans_Diffraction as dif

f = '../Dans_Diffraction/Structures/LaMnO3.mcif'
f = "../Dans_Diffraction/Structures/Sr3LiRuO6_C2'c'.mcif"
cif = dif.readcif(f)

ops = cif['_space_group_symop_magn_operation.xyz']
cen = cif['_space_group_symop_magn_centering.xyz']

print('Symmetry Operations (%d):' % len(ops))
print(ops)
print('Centring Operations (%d):' % len(cen))
print(cen)

# Combine operations with centring
allops = dif.fc.gen_symcen_ops(ops, cen)

# Convert to magnetic symmetry
magops = dif.fc.symmetry_ops2magnetic(allops)

print('\n%35s (%d) | %-40s' % ('Symmetry Operations', len(allops), 'Magnetic operations'))
for n in range(len(allops)):
    print('%40s | %-40s' % (allops[n], magops[n]))


sym, mag, tim = dif.fc.cif_symmetry(cif)
print('\ncif_symmetry')
for n in range(len(sym)):
    print('%40s | %+d | %-40s' % (sym[n], tim[n], mag[n]))