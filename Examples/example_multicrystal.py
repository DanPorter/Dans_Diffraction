"""
CrystalProgs Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from Dans_Diffraction.classes_crystal import Crystal, Multi_Crystal

cf=os.path.dirname(__file__)


f1 = cf+'/Diamond.cif'
f2 = cf+'/Na0.8CoO2_P63mmc.cif'

diamond = Crystal(f1)
nacoo2 = Crystal(f2)

xtls = Multi_Crystal([nacoo2,diamond])

# Plot Powder
xtls.simulate_powder(energy_kev = 5.0, peak_width=0.001)

print "Reflections"
xtls.print_all_reflections(energy_kev = 5.0, max_angle=130, print_symmetric=False)
print "Extinctions"
xtls.print_all_reflections(energy_kev = 5.0, max_angle=130, print_symmetric=False)