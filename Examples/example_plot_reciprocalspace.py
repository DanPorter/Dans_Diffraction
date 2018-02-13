"""
CrystalProgs Examples
Generate a plane in reciprocal space
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt # Plotting
from Dans_Diffraction.classes_crystal import Crystal

cf=os.path.dirname(__file__)


f = cf+'/Diamond.cif'
f = cf+'/Na0.8CoO2_P63mmc.cif'

xtl = Crystal(f)

xtl.Plot.simulate_hk0(L=0)
xtl.Plot.simulate_h0l(K=0)
xtl.Plot.simulate_hhl(HmH=0,q_max=6)
xtl.Plot.simulate_0kl(H=0)