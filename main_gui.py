"""
Dans_Diffraction
Start GUI script
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import Dans_Diffraction as dif

print('\nDans_Diffraction version %s, %s\n By Dan Porter, Diamond Light Source Ltd.' % (dif.__version__, dif.__date__))
print('See help(dif.Crystal) for info, or dif.start_gui() to get started!')

dif.start_gui()


