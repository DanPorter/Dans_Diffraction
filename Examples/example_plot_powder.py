"""
Dans_Diffraction Examples
Plot powder patter from crystal
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from Dans_Diffraction.classes_crystal import Crystal

cf=os.path.dirname(__file__)


f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'

xtl = Crystal(f)

xtl.Plot.simulate_powder(energy_kev=8, peak_width=0.01, background=0)