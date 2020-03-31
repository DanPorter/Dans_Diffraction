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


f = cf+'/../Dans_Diffraction/Structures/Sr3LiRuO6.cif'

xtl = dif.Crystal(f)

print(xtl.Properties.latex_table())