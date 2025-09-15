"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(cf,'..'))
import Dans_Diffraction as dif


#f = cf+'/../Dans_Diffraction/Structures/Ca2RuO4.cif'
f = dif.structure_list.Ca2RuO4.filename

xtl = dif.Crystal(f)
xtl.Atoms.changeatom(1,mxmymz=[0,3,0])
xtl.generate_structure()

print(xtl.Symmetry.info())

xtl.Plot.plot_crystal()
plt.show()

#xtl._scattering_type = 'xray magnetic' # neutron magnetic
xtl.Scatter.setup_scatter(
    scattering_type='xray magnetic',
    energy_kev=2.838,
    specular=[1,0,0],
    min_theta=-20,
    min_twotheta=0,
    max_twotheta=130
    )

print("X-Ray Magnetic Reflections")
print(xtl.Scatter.print_all_reflections(print_symmetric=True, min_intensity=0.1))

#xtl.Scatter.print_ref_reflections(min_intensity=0.1)
