"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
import Dans_Diffraction as dif

cf=os.path.dirname(__file__)


f1 = cf+'/../Dans_Diffraction/Structures/Ca3CoMnO6.mcif' # from Bilbao magnetic server R-3c
f2 = cf+'/../Dans_Diffraction/Structures/Li2IrO3.mcif' # from Bilbao magnetic server C2/m
f3 = cf+'/../Dans_Diffraction/Structures/Sr3LiRuO6_C2\'c\'.mcif' # from isodistort R-3c

xtl1 = dif.Crystal(f1)
xtl2 = dif.Crystal(f2)
xtl3 = dif.Crystal(f3)

xtl1.Symmetry.info()

xtl2.Symmetry.info()

xtl3.Symmetry.info()