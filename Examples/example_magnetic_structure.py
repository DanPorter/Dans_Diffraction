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


#f = cf+'/../Dans_Diffraction/Structures/Ca2RuO4.cif'
f = structure_list.Ca2RuO4.filename

xtl = Crystal(f)
xtl.Atoms.changeatom(1,mxmymz=[0,3,0])
xtl.generate_structure()

xtl.Plot.plot_crystal()

#xtl._scattering_type = 'xray magnetic' # neutron magnetic
xtl.Scatter.setup_scatter(
    type='xray magnetic',
    energy_kev=2.838,
    specular=[1,0,0],
    min_theta=-20,
    min_twotheta=0,
    max_twotheta=130
    )

print "X-Ray Magnetic Reflections"
print(xtl.Scatter.print_all_reflections(print_symmetric=True, min_intensity=0.1))

#xtl.Scatter.print_ref_reflections(min_intensity=0.1)
