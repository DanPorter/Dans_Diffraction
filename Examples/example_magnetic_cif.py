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


f1 = cf+'/../Dans_Diffraction/Structures/Ca3CoMnO6.mcif'  # from Bilbao magnetic server R-3c
f2 = cf+'/../Dans_Diffraction/Structures/Sr3LiRuO6_C2\'c\'.mcif'  # from isodistort C2'/c'
f3 = cf+'/../Dans_Diffraction/Structures/LaMnO3.mcif'  # from Bilbao magnetic server Pn'ma'

xtl1 = dif.Crystal(f1)
xtl2 = dif.Crystal(f2)
xtl3 = dif.Crystal(f3)

print(xtl1.name)
print('Magnetic Intensity (100): %5.3f' % xtl1.Scatter.xray_resonant([1,0,0], 7.8, 'sp', F0=0, F1=1, F2=0))
print(xtl1.Symmetry.info())

print(xtl2.name)
print('Magnetic Intensity (100): %5.3f' % xtl2.Scatter.xray_resonant([1,0,0], 2.838, 'sp', F0=0, F1=1, F2=0))
print(xtl2.Symmetry.info())

print(xtl3.name)
print('Magnetic Intensity (100): %5.3f' % xtl3.Scatter.xray_resonant([1,0,0], 2.838, 'sp', F0=0, F1=1, F2=0))
print(xtl3.Symmetry.info())