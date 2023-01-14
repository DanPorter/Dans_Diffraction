"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'

xtl = dif.Crystal(f)

xtl.Scatter.setup_scatter(scattering_type='x-ray', energy_kev=8.0)
print("Reflections")
print(xtl.Scatter.print_all_reflections(print_symmetric=False, min_intensity=0.01, units='tth'))
print("Extinctions")
print(xtl.Scatter.print_all_reflections(print_symmetric=False, min_intensity=None, max_intensity=0.01, units='tth'))

