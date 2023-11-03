"""
Dans_Diffraction Examples
Analyse FDMNES results
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf, '..'))
import Dans_Diffraction as dif

output_path = r'Test'
output_name = r'out'

ana = dif.FdmnesAnalysis(output_path, output_name)
print(ana)

ana.xanes.plot()
plt.show()
ana.I1000sp.plot3D()
plt.show()
ana.density.plot()
plt.show()
