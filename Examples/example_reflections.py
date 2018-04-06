"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from Dans_Diffraction.classes_crystal import Crystal

cf=os.path.dirname(__file__)


f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'

xtl = Crystal(f)

xtl.Scatter.setup_scatter(type='x-ray', energy_kev=8.0)
print("Reflections")
print(xtl.Scatter.print_all_reflections(print_symmetric=False, min_intensity=0.01))
print("Extinctions")
print(xtl.Scatter.print_all_reflections(print_symmetric=False, min_intensity=None, max_intensity=0.01))