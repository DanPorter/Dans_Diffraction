"""
Dans_Diffraction Examples
Run multiple scattering calculations
"""

import sys, os
import matplotlib.pyplot as plt
import numpy as np

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))

import Dans_Diffraction as dif
import Dans_Diffraction.tensor_scattering as ts

np.set_printoptions(precision=3, suppress=True)

print('Stand alone code:')

cif_file='../Dans_Diffraction/Structures/ZnO.cif'
xtl = dif.Crystal(cif_file)

t = ts.TensorScatteringClass(xtl, Site='Zn1', TimeEven=False)
t.PlotIntensityInPolarizationChannels('E1E2', lam=12.4/9.659, hkl=np.array([1,1,5]), hkln=np.array([1,0,0]), K=3, Time=1, Parity=-1, mk=None, sk=None, sigmapi='sigma')
t.print_tensors()
plt.show()


print('Built in code:')

cif_file='../Dans_Diffraction/Structures/Ca2RuO4.cif'
xtl = dif.Crystal(cif_file)

xtl.Plot.tensor_scattering_azimuth('Ru1', [0,0,3], 2.838, [0,1,0])
plt.show()

xtl.Plot.tensor_scattering_stokes('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90)
plt.show()

psi = np.arange(-180, 180, 1)
ss, sp, ps, pp = xtl.Scatter.tensor_scattering('Ru1', [0,0,3], 2.838, [0,1,0], psideg=psi)

stokes = np.arange(-180,181,2)
pol = xtl.Scatter.tensor_scattering_stokes('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90, stokes=stokes)


print(xtl.Scatter.print_tensor_scattering('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90))

print(xtl.Scatter.print_tensor_scattering_refs_max('Ru1', 2.838, [0,1,0]))