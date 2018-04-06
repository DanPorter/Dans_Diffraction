"""
Dans_Diffraction Examples
Calculate list of reflections
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from Dans_Diffraction.classes_crystal import Crystal, MultiCrystal

cf=os.path.dirname(__file__)


f1 = cf+'/../Dans_Diffraction/Structures/Diamond.cif'
f2 = cf+'/../Dans_Diffraction/Structures/Na0.8CoO2_P63mmc.cif'

diamond = Crystal(f1)
nacoo2 = Crystal(f2)

xtls = MultiCrystal([nacoo2, diamond])

# Plot Powder
xtls.simulate_powder(energy_kev = 5.0, peak_width=0.001)

print "Reflections"
print(xtls.print_all_reflections(energy_kev=5.0, max_angle=130, print_symmetric=False))
#print "Extinctions"
#print(xtls.print_all_reflections(energy_kev=5.0, max_angle=130, print_symmetric=False))
