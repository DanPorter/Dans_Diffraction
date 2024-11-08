"""
Dans_Diffraction Examples
Example of using different crystal basis options
"""

import sys
import os

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))

import Dans_Diffraction as dif
from Dans_Diffraction import functions_lattice as fl

dif.fg.nice_print()

hkl = [3, 5, 1]
lp = fl.random_lattice('hexagonal')  # 'cubic', 'hexagonal', 'tetragonal', 'rhobohedral', 'monoclinic-a/b/c', 'triclinic'
basis_options = ['materialsproject', 'vesta', 'busingandlevy']
"""
Available basis options:
1. 'MaterialsProject': c || z, b* || y
2. 'Vesta': a || x, c* || z
3. 'BusingandLevy': c || z, a* || x, 'default'
"""

for n in range(3):
    basis_function = fl.choose_basis(n + 1)
    basis_vectors = basis_function(*lp)  # returns [avec, bvec, cvec]
    reciprocal_vectors = fl.reciprocal_basis(basis_vectors)
    volume = fl.lattice_volume(*lp)
    dspace = fl.dspacing(*hkl, *lp)

    print(f"Basis no. {n + 1}: {basis_options[n]}")
    print(f"Lattice parameters: {lp}")
    print('\n'.join(f"vector {v}: {basis_vectors[vn]}" for vn, v in enumerate(['a ', 'b ', 'b '])))
    print('\n'.join(f"vector {v}: {reciprocal_vectors[vn]}" for vn, v in enumerate(['a*', 'b*', 'b*'])))
    print(f"volume = {volume:.2f} A^3")
    print(f"d-spacing {hkl} = {dspace:.2f} A\n")


