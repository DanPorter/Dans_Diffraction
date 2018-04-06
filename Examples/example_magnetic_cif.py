"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from Dans_Diffraction import functions_general as fg
from Dans_Diffraction import functions_plotting as fp
from Dans_Diffraction import Crystal
from Dans_Diffraction import structure_list

cf=os.path.dirname(__file__)


f1 = cf+'/../Dans_Diffraction/Structures/Ca3CoMnO6.mcif' # from Bilbao magnetic server R-3c
f2 = cf+'/../Dans_Diffraction/Structures/Li2IrO3.mcif' # from Bilbao magnetic server C2/m
f3 = cf+'/../Dans_Diffraction/Structures/Sr3LiRuO6_C2\'c\'.mcif' # from isodistort R-3c

xtl1 = Crystal(f1)
xtl2 = Crystal(f2)
xtl3 = Crystal(f3)

xtl1.Symmetry.info()

xtl2.Symmetry.info()

xtl3.Symmetry.info()