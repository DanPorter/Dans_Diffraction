"""
Dans_Diffraction Examples
Read values from a Crystallographic Information File (.cif or .mcif)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif

# Filepath
ciffile = '../Dans_Diffraction/Structures/LiCoO2.cif'

# Create a cif dictionary
cif = dif.fc.readcif(ciffile)

# Display available dict keys:
cif.keys()

# Name
name = cif['_pd_phase_name']

"""
Values are often given in standard scientific form with errors:
cif['_cell_length_a'] >> '2.824(1)'
use dif.fg.readstfm to return the value and error
"""
# Lattice parameters
a, da = dif.fg.readstfm(cif['_cell_length_a'])
b, db = dif.fg.readstfm(cif['_cell_length_b'])
c, dc = dif.fg.readstfm(cif['_cell_length_c'])
alpha, dalpha = dif.fg.readstfm(cif['_cell_angle_alpha'])
beta, dbeta = dif.fg.readstfm(cif['_cell_angle_beta'])
gamma, dgamma = dif.fg.readstfm(cif['_cell_angle_gamma'])

# Symmetry operators
symms = cif['_symmetry_equiv_pos_as_xyz']

# Atomic Structure parameters
labels = cif['_atom_site_label']
types = cif['_atom_site_type_symbol']
occupancy = [dif.fg.readstfm(val)[0] for val in cif['_atom_site_occupancy']]
d_occupancy = [dif.fg.readstfm(val)[1] for val in cif['_atom_site_occupancy']]
x = [dif.fg.readstfm(val)[0] for val in cif['_atom_site_fract_x']]
dx = [dif.fg.readstfm(val)[1] for val in cif['_atom_site_fract_x']]
y = [dif.fg.readstfm(val)[0] for val in cif['_atom_site_fract_y']]
dy = [dif.fg.readstfm(val)[1] for val in cif['_atom_site_fract_y']]
z = [dif.fg.readstfm(val)[0] for val in cif['_atom_site_fract_z']]
dz = [dif.fg.readstfm(val)[1] for val in cif['_atom_site_fract_z']]


# Print the results
print(cif['Filename'])
print(name)
print('Lattice Parameters:')
print(a,b,c,alpha,beta,gamma)
print('Space Group: %s (%s)' % (cif['_symmetry_space_group_name_H-M'], cif['_symmetry_Int_Tables_number']))
print('No symmetry operations: %d'%len(symms))
print('\nAtomic Positions:')
print('%5s %4s %4s %6s %6s %6s' % ('Label', 'Type', 'Occ', 'x', 'y', 'z'))
for n in range(len(labels)):
    print('%5s %4s %4.2f %6.3f %6.3f %6.3f' % (labels[n], types[n], occupancy[n], x[n], y[n], z[n]))

