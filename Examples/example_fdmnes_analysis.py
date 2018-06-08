"""
Dans_Diffraction Examples
Analyse FDMNES results
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
import Dans_Diffraction as dif

cf=os.path.dirname(__file__)

output_path = r'C:\Users\grp66007\Documents\FDMNES\Sim\Test'
output_name = r'out'

ana = dif.FdmnesAnalysis(output_path, output_name)

ana.xanes.plot()
plt.show()
ana.I100sp.plot3D()
plt.show()
ana.density.plot()
plt.show()
