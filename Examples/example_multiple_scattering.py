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

cif_file='../Dans_Diffraction/Structures/Na0.8CoO2_P63mmc.cif'
xtl = dif.Crystal(cif_file)
#mslist = xtl.Scatter.multiple_scattering([0,0,3], [0,1,0], [1,0], [2.82, 2.85])
#mslist = run_calcms(xtl, [0,0,3], [0,1,0], [1,0], [2.83, 2.85])

# xtl.Plot.plot_multiple_scattering([1,0,0], [0,1,0], energy_range=[2.81, 2.86])
xtl.Plot.plot_ms_azimuth([0, 0, 6], 8, [0, 1, 0], peak_width=0.1)
plt.show()
