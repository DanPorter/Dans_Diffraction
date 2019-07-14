"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


f = cf+'/../Dans_Diffraction/Structures/Ca2RuO4.cif'

xtl = dif.Crystal(f)
xtl.Atoms.changeatom(1,mxmymz=[0,3,0.3])
xtl.generate_structure()

F0,F1,F2 = 0,1,0
HKLs = [[1,0,0],[2,0,0]]
inten=xtl.Scatter.xray_resonant(HKLs, energy_kev=2.967,azim_zero=[0,0,1],PSI=[90],F0=F0,F1=F1,F2=F2)

print('simulating azimuth')
xtl.Plot.simulate_azimuth([1,0,0], energy_kev=2.967, polarisation='sp', azim_zero=[0,1,0])

print('simulating polarisation (resonant)')
xtl.Plot.simulate_polarisation_resonant([1,0,0], energy_kev=2.967, F0=0, F1=1, F2=0, azim_zero=[0, 1, 0], psi=0)

print('simulating polarisation (non-resonant)')
xtl.Plot.simulate_polarisation_nonresonant([1,0,0], energy_kev=2.967, azim_zero=[0, 1, 0], psi=0)

print('finished')
plt.show()