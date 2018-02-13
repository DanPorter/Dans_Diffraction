"""
CrystalProgs Examples
Calculate list of reflections in a pressure cell with 
limited opening window.
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting

from Dans_Diffraction import functions_general as fg
from Dans_Diffraction import functions_crystallography as fc
from Dans_Diffraction.classes_crystal import Crystal

cf=os.path.dirname(__file__)


#f = cf+'/Sr3Ir2O7.cif'
f = cf+'/Diamond.cif'

xtl = Crystal(f)

en = 12

# Reflection Geometry
dac_opening = 100
max_delta = 135
min_eta = 90-dac_opening/2.
max_eta = max_delta-min_eta

xtl.Scatter.setup_scatter(type='xray', 
                  specular=[0,0,1], 
                  theta_offset=0, 
                  min_theta=90-dac_opening/2., 
                  max_theta=max_delta-min_eta, 
                  min_twotheta=5, 
                  max_twotheta=max_delta)

# Transmission Geometry
#dac_opening = 100
#min_eta = -dac_opening/2. 
#max_eta = dac_opening/2.
#max_delta = dac_opening/2.

"""
hkl=xtl.Cell.all_hkl(en, max_delta)
eta=xtl.Cell.theta_reflection(hkl,en,[0,0,1])
tth=xtl.Cell.tth(hkl,en)

p1=eta>min_eta
p2=tth>(eta+min_eta)
hkl1=hkl[p1*p2]
hkl1=xtl.Cell.sort_hkl(hkl1)
xtl.hkl(hkl1,en)
"""
xtl.Scatter.print_ref_reflections(en, min_intensity=None, max_intensity=None)


xtl.Plot.simulate_ewald_coverage(en, [0,0,1], [1,0,0])
angles = np.arange(min_eta,max_eta,0.1)
Q1x,Q1y=fc.diffractometer_Q(angles,max_delta,en)
Q2x,Q2y=fc.diffractometer_Q(angles,angles+min_eta,en)
Q3x,Q3y=fc.diffractometer_Q(min_eta,angles+min_eta,en)
plt.plot(Q1x,Q1y,'r',lw=2)
plt.plot(Q2x,Q2y,'r',lw=2)
plt.plot(Q3x,Q3y,'r',lw=2)
