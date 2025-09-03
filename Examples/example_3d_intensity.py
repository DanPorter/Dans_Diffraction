"""
Dans_Diffraction Examples
Calculate crystal structure factors and display 3D plot of reciprocal space
"""

import numpy as np
import matplotlib.pyplot as plt
import Dans_Diffraction as dif

print(dif.version_info())
dif.fp.set_plot_defaults()

xtl = dif.structure_list.Diamond()

xtl.Scatter.setup_scatter(scattering_type='xray')

# xtl.Plot.plot_intensity_histogram()
xtl.Plot.plot_3Dintensity()

xtl = dif.structure_list.LiCoO2()
xtl.Scatter.setup_scatter(scattering_type='xray')
xtl.Plot.plot_3Dintensity(q_max=1, show_forbidden=True, central_hkl=(0,0,6))

plt.show()