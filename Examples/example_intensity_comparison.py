"""
Dans_Diffraction Examples
Compare calcualted reflection intensities for a CIF between different software
"""

import numpy as np
import Dans_Diffraction as dif

print(dif.version_info())

xtl = dif.Crystal('~/Downloads/Rutile.cif')

# The CIF doesn't include isotropic thermal parameters, the default in Dans_Diffraction is 0, in Vesta it is to set B=1
xtl.Atoms.changeatom([0, 1], uiso=0.0126651)  # B=1
xtl.generate_structure()

# Vesta uses scattering factors from "Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431"
# Dans_Diffraction uses scattering factors from Internaltion Tables Vol C., set the Waasmaier coefficients:
xtl.Scatter.setup_scatter(energy_kev=8.048, scattering_factors='waaskirf')

# Vesta intesnity (http://jp-minerals.org/vesta/en/)
#   h    k    l      d (Å)      F(real)      F(imag)          |F|         2θ          I    M
#   1    1    0   3.243499    37.053131     0.000000      37.0531   27.47676  100.00000    4
#   1    0    1   2.483556    23.439516     0.000000      23.4395   36.13751   44.32719    8
#   2    0    0   2.293500    13.943417     0.000000      13.9434   39.24969    6.53618    4
#   1    1    1   2.183984   -17.270726     0.000000      17.2707   41.30531   17.90116    8
hkl = [[1, 1, 0], [0, 1, 1], [0, 2, 0], [1, 1, 1]]
inten_vesta = np.array([37.0531, 23.4395, 13.9434, 17.2707]) ** 2
inten_disp = np.array([37.5527, 23.9331, 14.5057, 17.4241]) ** 2  # with x-ray dispersion effects
inten_neut = np.array([3.97176, 14.0359, 23.4644, 19.4905]) ** 2  # Neutron scattering lengths
inten_dift = np.array([44.4472885365421, 20.3801391390013, 3.04824911713064, 8.42516396817784])**2  # From CrystalDiffract
intensity = xtl.Scatter.x_ray(hkl)  # Dans_Diffraction
imax = intensity[0]
print('\n                       Dans_Diffraction     Vesta                  CrystalDiffract')
print('             (h,k,l)          I   I/I110           I   I/I110           I   I/I110')
for n in range(len(hkl)):
    print('%20s   %8.2f %8.2f    %8.2f %8.2f    %8.2f %8.2f' % (
        hkl[n], intensity[n], 100 * intensity[n] / imax,
        inten_vesta[n], 100 * inten_vesta[n] / inten_vesta[0],
        inten_dift[n], 100 * inten_dift[n] / inten_dift[0]
    ))
