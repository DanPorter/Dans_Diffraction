"""
Dans_Diffraction Examples
Calculate list of reflections using different methods
Makes use of the new scattering functionality in Dans_Diffraction 2.0.0+

Requires Dans_Diffraction version >2.0.0
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt  # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif
print(dif.version_info())

# ---Settings---
f = cf+'/../Dans_Diffraction/Structures/Ca2RuO4.cif'
en = 2.838  # keV
aziref = [1, 1, 0]

# Create Crystal, assign magnetic moment
xtl = dif.Crystal(f)
xtl.Atoms.changeatom(1, mxmymz=[0, 3, 0.3])
xtl.generate_structure()

xtl.Scatter.setup_scatter(
    energy_kev=en,
    scattering_factors='WaasKirf',  # 'WaasKirf' or 'itc'
    magnetic_formfactor=False,
    polarisation='sp',
    polarisation_vector=(1,0,0),
    azimuthal_reference=aziref,
    azimuth=0,
    flm=(0,1,0)
)

hkl = xtl.Scatter.get_hkl(remove_symmetric=True)
i_xray = xtl.Scatter.intensity(hkl, 'xray')
i_neutron = xtl.Scatter.intensity(hkl, 'neutron')
i_mag_xray = xtl.Scatter.intensity(hkl, 'xray magnetic')
i_mag_neutron = xtl.Scatter.intensity(hkl, 'neutron magnetic')
i_xray_resonant = xtl.Scatter.intensity(hkl, 'xray resonant')

print('%20s %10s %10s %10s %10s %10s' % ('(h, k, l)', 'xray', 'neutron', 'mag. xr', 'mag. n', 'res. xr'))
for ref, i1, i2, i3, i4, i5 in zip(hkl, i_xray, i_neutron, i_mag_xray, i_mag_neutron, i_xray_resonant):
    print('%20s %10.2f %10.2f %10.2f %10.2f %10.2f' % (ref, i1, i2, i4, i4, i5))


# Calcualte energy and psi ranges
hkl = [0, 1, 3]
envals = np.arange(en-0.5, en+0.5, 0.01)
psivals = np.arange(-180, 180)


energy_refs = xtl.Scatter.intensity([0,0,2], 'xray dispersion', energy_kev=envals)
psi_refs = xtl.Scatter.intensity(hkl, 'xray resonant', psi=psivals)

plt.figure()
plt.plot(envals, energy_refs[0])
plt.title(hkl)
plt.xlabel('Energy [keV]')

plt.figure()
plt.plot(psivals, psi_refs[0])
plt.title(hkl)
plt.xlabel('Psi [deg]')

plt.show()

# Large cell calculation
import time
print('Superstructure calculation')
sup = xtl.generate_superstructure([[5, 0, 0], [0, 5, 0], [0, 0, 1]])
shkl = sup.Scatter.get_hkl()
t0 = time.process_time()
s_xray = sup.Scatter.intensity(shkl, 'xray')
t1 = time.process_time()
print('%d reflections, %d atoms, max_intensity = %s' % (len(shkl), len(sup.Structure.u), np.max(s_xray)))
print('Time taken: %5.2f s' % (t1 - t0))
