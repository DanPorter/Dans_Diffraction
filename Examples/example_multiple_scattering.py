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
#import Dans_Diffraction.classes_multiple_scattering as ts
#from Dans_Diffraction.classes_multiple_scattering import run_calcms

cif_file='../Dans_Diffraction/Structures/Ca2RuO4.cif'
xtl = dif.Crystal(cif_file)
mslist = xtl.Scatter.multiple_scattering([0,0,3], [0,1,0], [1,0], [2.82, 2.85])
#mslist = run_calcms(xtl, [0,0,3], [0,1,0], [1,0], [2.83, 2.85])

plt.show()
