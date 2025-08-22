"""
Dans_Diffraction Examples
Calculate magnetic neutron structure factors with magnetic form factor
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import Dans_Diffraction as dif


#f = '../Dans_Diffraction/Structures/MnO.mcif'
# f = dif.structure_list.LaMnO3.filename
# f = r"C:\Users\grp66007\PycharmProjects\Dans_Diffraction\Test\compare_mag_neutron_intensity_Aug25\LaMnO3_300K.cif"
f = r"C:\Users\grp66007\PycharmProjects\Dans_Diffraction\Test\compare_mag_neutron_intensity_Aug25\LaMnO3.cif"

# Magnetic form factor
# mag_ff = dif.fc.load_magnetic_ff_coefs()
q = np.arange(0, 20, 0.05)
ion = 'Mn3+'
j2j0_ratio = dif.fc.magnetic_ff_j2j0_ratio(dif.fc.magnetic_ff_g_factor(ion), ratio_type=1)
mag_ff = dif.fc.magnetic_form_factor(ion, qmag=q, j2j0_ratio=j2j0_ratio)
print(f"Magnetic Form Facors for {ion} with f(Q) = <j0> + {j2j0_ratio:.4g}<j2>")
print(dif.fc.print_magnetic_ff_coefs(ion))

# plt.figure()
# plt.plot(q, mag_ff)
# plt.xlabel('|Q|')
# plt.ylabel(f"<j0> + {j2j0_ratio:.3g}<j2>")
# plt.title(ion)
# plt.grid()
# plt.show()


xtl = dif.Crystal(f)
xtl.Atoms.changeatom(1, type='Mn3+')
# xtl.Atoms.removeatom(3)
# xtl.Atoms.removeatom(2)
# xtl.Atoms.removeatom(0)
xtl.generate_structure()
ub = 3.87
xtl.Structure.changeatom(4, mxmymz=[ub, 0, 0])
xtl.Structure.changeatom(5, mxmymz=[-ub, 0, 0])
xtl.Structure.changeatom(6, mxmymz=[-ub, 0, 0])
xtl.Structure.changeatom(7, mxmymz=[ub, 0, 0])
print(xtl)

xtl.Scatter.setup_scatter(
    scattering_type='neutron',
    magnetic_formfactor=True,
    use_sears=True,
)

# print(xtl.Scatter.print_scattering_factor_coefficients())


# print(xtl.Scatter.print_atomic_contributions([0,0,2]))

# print(xtl.Scatter.print_symmetry_contributions([0, 0, 2]))

# print("Neutron Magnetic Reflections")
# print(xtl.Scatter.print_all_reflections(print_symmetric=True, min_intensity=0.1))

# 40.602
pols = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
ii = 0
for pol in pols:
    xtl.Scatter._polarisation_vector_incident = pol
    f = np.sqrt(xtl.Scatter.intensity([0, 0, 1], scattering_type='neutron magnetic'))
    print(f"pol = {pol}, |SF| = {f[0]:8.5g}")
    ii += f[0]
print(f"Total |SF| = {ii:8.5g}")
"""
pol = [1, 0, 0], |SF| =   15.355
pol = [0, 1, 0], |SF| =   15.355
pol = [0, 0, 1], |SF| =   15.355
Total |SF| =   46.065
"""

# sf_file = r"C:\Users\grp66007\PycharmProjects\Dans_Diffraction\Test\compare_mag_neutron_intensity_Aug25\Fcal_Mag_CM.txt"
sf_file = r"C:\Users\grp66007\PycharmProjects\Dans_Diffraction\Test\compare_mag_neutron_intensity_Aug25\Fcal_Mag_CM_LMO.txt"
data = np.loadtxt(sf_file)

hkl = data[:, :3].astype(int)
nuclear = np.sqrt(xtl.Scatter.intensity(hkl, scattering_type='neutron'))  # |Fcal| = sqrt(I) = sqrt(|Fcal|^2)
magnetic = np.sqrt(xtl.Scatter.intensity(hkl, scattering_type='neutron magnetic'))

for n, (h, k, l) in enumerate(hkl):
    if (abs(h)+abs(k)+abs(l)) <= 4 and data[n, 5] > 0.1 and h >= 0 and k >= 0 and l >= 0:
        print(f"{h:2} {k:2} {l:2}  {data[n, 4]:8.3f}={nuclear[n]:8.3f}  {data[n, 5]:8.3f}={magnetic[n]:8.3f}")

#
"""
 0  0  1     0.000=   0.000    40.602=  15.355
 0  0  3     0.000=   0.000    32.648=  14.270
 0  2  1     8.136=   8.136    33.371=  14.390
 1  1  1    10.026=  10.026    28.146=  11.436
 2  0  1     0.000=   0.000    11.136=   4.845
"""
hkl_inten = [
    ([0, 0, 1], 40.602),
    ([0, 0, 3], 32.648),
    ([0, 2, 1], 33.371),
    ([1, 1, 1], 28.146),
    ([2, 0, 1], 11.136),
]

# LaMnO3.cif, ...LMO.txt
"""
 0  1  0     0.000=   0.000    41.456=  15.354
 0  1  2     0.000=   0.000    38.630=  14.308
 0  3  0     0.000=   0.000    38.502=  14.260
 1  1  1    10.489=  10.489    31.652=  11.723
 2  1  0     7.534=   7.534    13.637=   5.051
 """
hkl_inten = [
    ([0, 1, 0], 41.456),
    ([0, 1, 2], 38.630),
    ([0, 3, 0], 38.502),
    ([1, 1, 1], 31.652),
    ([2, 1, 0], 13.637),
]