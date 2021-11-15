"""
Dans_Diffraction Examples
Set and change the orientation of the crystal
Makes use of the new orientation functionality in Dans_Diffraction 2.1.0+

Requires Dans_Diffraction version >2.1.0
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt  # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif
print(dif.version_info())
dif.fg.nice_print()

# ---Settings---
f = cf+'/../Dans_Diffraction/Structures/Ca2RuO4.cif'

# Create Crystal
xtl = dif.Crystal(f)

# Set orientation
xtl.Cell.orientation.orient(a_axis=np.random.rand(3), c_axis=[0, 0, 1])


def diffractometer(eta, chi, phi, delta, energy_kev=8, fwhm=0.01):
    hkl, factor = xtl.Cell.diff6circle_match(phi, chi, eta, 0, delta, 0, energy_kev, fwhm=fwhm)
    inten = xtl.Scatter.diff6circle_intensity(phi, chi, eta, 0, delta, 0, energy_kev, fwhm=fwhm)
    s = '%6.3f keV: eta=%5.2f, chi=%5.2f, phi=%5.2f, delta=%5.2f   hkl=%s  I=%5.2f'
    print(s % (energy_kev, eta, chi, phi, delta, hkl, inten))


def scan_eta(eta_range, chi, phi, delta, energy_kev=8, fwhm=0.01):
    return [xtl.Scatter.diff6circle_intensity(phi, chi, eta, 0, delta, 0, energy_kev, fwhm=fwhm) for eta in eta_range]


diffractometer(14.5, 90, 0, 30)
diffractometer(15, 90, 0, 30)
diffractometer(15.5, 90, 0, 30)

eta_range = np.arange(14, 16, 0.01)
inten = scan_eta(eta_range, 90, 0, 30)

plt.figure()
plt.plot(eta_range, inten, '+-')
plt.xlabel('eta')
plt.ylabel('Intensity')
plt.title(xtl.name)
plt.show()
