"""
SuperDuper Set of Tests of Dans_Diffraction
"""

import sys, os

cf = os.path.dirname(os.path.abspath(__file__))
# Location of Dans_Diffraction
ddf = os.path.split(cf)[0]

# 0) Check python version
print('Dans_Diffraction Unit Tests')
print('Python Version:')
print(sys.version)
print('File Location:')
print('%s'%(cf))
print('Location of Dans_Diffraction')
print('%s'%(ddf))

# 1) Import python modules
import numpy as np
print('\nNumpy version: %s'%np.__version__)
import matplotlib.pyplot as plt
print('Matplotlib version: %s'%plt.matplotlib.__version__)
import scipy
print('Scipy version: %s'%scipy.__version__)

# 2) Import Dans_Diffraction
print('\nImporting Dans_Diffraction...')
try:
    import Dans_Diffraction as dif
    dif.__version__
except ImportError:
    print('Import Dans_Diffraction was not possible, adding path...')
    if ddf not in sys.path:
        sys.path.insert(0,ddf)
    import Dans_Diffraction as dif
except AttributeError:
    print('Import Dans_Diffraction was incorrect, retrying...')
    from Dans_Diffraction import Dans_Diffraction as dif
print('Dans_Diffraction version: %s'%dif.__version__)
print('  classes_crystal: %s'%dif.classes_crystal.__version__)
print('  classes_plotting: %s'%dif.classes_plotting.__version__)
print('  classes_scattering: %s'%dif.classes_scattering.__version__)
print('  classes_properties: %s'%dif.classes_properties.__version__)
print('  classes_structures: %s'%dif.classes_structures.__version__)
print('  classes_orbitals: %s' % dif.classes_orbitals.__version__)
print('  classes_fdmnes: %s'%dif.classes_fdmnes.__version__)
print('  functions_crystallography: %s'%dif.fc.__version__)
print('  functions_plotting: %s'%dif.fp.__version__)
print('  functions_general: %s'%dif.fg.__version__)

# 2.1) Import GUI
print('Graphical front end version:')
try:
    from Dans_Diffraction import tkgui
    print('  tkgui: %s'%tkgui.__version__)
except ImportError:
    print('--- No tkinter available ---')

# 3) Read atom properties
weight = dif.fc.atom_properties('Co','Weight')
print('\nCobalt Weight = %6.2f g' % weight[0])
xsf = dif.fc.xray_scattering_factor('Co', 3)
print('Cobalt x-ray scattering factor @ 3A = %6.2f' % xsf[0, 0])
mff = dif.fc.magnetic_form_factor('Co', 0.5)
print('Cobalt magnetic form factor @ 0.5A = %6.2f' % mff[0, 0])
att = dif.fc.attenuation(27, 8)
print('Cobalt attenuation @ 8.00keV = %6.2f cm^2/g' % att)
asf1, asf2 = dif.fc.atomic_scattering_factor('Co', 2.838)
print('Cobalt atomic scattering factor @ 8.000keV: f1+if2 = %s' % np.complex(asf1, asf2))

# 4) Test crystal building
print('\nBuilding a crystal manually...')
xtl1 = dif.Crystal()
xtl1.name = 'Oh! What a Lovely Crystal'
xtl1.new_cell([2.8,2.8,6.0,90,90,90])
#xtl.new_atoms(u=[0,0.5], v=[0,0.5], w=[0,0.25], type=['Na','O'], label=['Na1','O1'], occupancy=None, uiso=None, mxmymz=None)
xtl1.Atoms.changeatom(0,u=0,   v=0,   w=0,    type='Na',label='Na1',occupancy=None, uiso=None, mxmymz=None) # there is an Fe ion added by default
xtl1.Atoms.addatom(u=0.5, v=0.5, w=0.25, type='O', label='O1', occupancy=None, uiso=None, mxmymz=None)
xtl1.Symmetry.addsym('x,y,z+1/2')
xtl1.generate_structure() # apply symmetry to atomic positions.
xtl1.info()

# 5) Spacegroups
print('Loading spacegroup 194')
spg = dif.fc.spacegroup(194)
print('Spacegroup %s: %s, nsyms=%d' % (spg['space group number'], spg['space group name'], len(spg['general positions'])))
mspg = dif.fc.spacegroup_magnetic(spg['magnetic space groups'][1])
print('Magnetic Spacegroup %s: %s, nsyms=%d' % (mspg['space group number'], mspg['space group name'], len(mspg['operators magnetic'])))
xtl1.Symmetry.load_spacegroup(61)
xtl1.Symmetry.load_magnetic_spacegroup(spg['magnetic space groups'][1])

# 6) CIF Reading
ciffile = dif.structure_list.Na08CoO2_P63mmc.filename
print('\nBuilding a crystal from a cif file: %s'%ciffile)
files = list(dif.classes_structures.cif_list())
print('NB %d cif files are included'%len(files))
xtl2 = dif.Crystal(ciffile)
xtl2.info()

# 7) mCIF Reading
ciffile = dif.structure_list.Sr3LiRuO6_C2c.filename
print('\nBuilding a crystal from an mcif file: %s'%ciffile)
files = list(dif.classes_structures.cif_list())
xtlm = dif.Crystal(ciffile)
xtlm.info()

# 8) CIF Writing
xtl2.name = 'Testing Testing'
xtl2.write_cif('Test.cif', comments='This is a test\n it is not interesting')
print('\nWriting CIF to: test.cif')
xtl2 = dif.Crystal('Test.cif')
print('Test.cif loaded succesfully')

# 9) X-ray scattering
print('\nTest x-ray scattering:')
xtl2.Scatter.hkl([[1,0,0],[2,0,0],[1,1,0],[0,0,1]],8.0)

# 10) Magnetic scattering
print('\nTest Magnetic scattering:')
xtlm.Scatter.print_intensity([0,0,3])

# 11) Resonant scattering
resi=xtlm.Scatter.xray_resonant([0,0,3], energy_kev=2.967, polarisation='sp', azim_zero=[0,1,0], PSI=[90], F0=1, F1=1, F2=1)
print('\nResonant scattering RuL2 psi=90 sp = %6.2f'%resi)

# 12) Multiple scattering
mslist = xtlm.Scatter.ms_azimuth([0,0,2], 2.967, [1,0,0])

# 13) Tensor scattering
ss, sp, ps, pp = xtlm.Scatter.tensor_scattering('Ru1_1', [0,0,3], 2.838, [0,1,0], psideg=np.arange(-180, 180, 1))

# 14) Generate superstructure
P = [[3,0,0],[4,5,0],[0,0,1]] # Stripe Supercell
xtl2.Atoms.occupancy[2]=0
xtl2.Atoms.occupancy[3]=1
xtl2.generate_structure()
sup = xtl2.generate_superstructure(P)
sup.Structure.occupancy[[6 ,16,26,77,87,107]] = 1 # Na1
sup.Structure.occupancy[[8 ,18,38,28,48,58, 139, 119, 149, 109, 89,79]] = 0 # Na2

# 15) Multicrystal
xtls = xtl2 + dif.structure_list.Diamond() + dif.structure_list.Aluminium()

# 16) Plotting
print('\nStarting Plotting Tests...')
print('  Plotting Powder')
xtl2.Plot.simulate_powder(energy_kev=8, peak_width=0.01)
plt.show()
print('  Plotting hk0 plane')
xtl2.Plot.simulate_hk0()
plt.show()
print('  Plotting h0l plane')
xtl2.Plot.simulate_h0l()
plt.show()
print('  Plotting Crystal structure')
xtl2.Plot.plot_crystal()
plt.show()
print('  Plotting azimuthal scan')
xtlm.Plot.simulate_azimuth([0,0,3])
plt.show()
print('  Plotting multiple scattering')
xtlm.Plot.plot_multiple_scattering([0,0,2], [1,0,0], energy_range=[2.95, 2.98], numsteps=61)
plt.show()
print('  Plotting tensor scattering')
xtlm.Plot.tensor_scattering_azimuth('Ru1_1', [0,0,3], 2.838, [0,1,0])
plt.show()
print('  Plotting Superstructure hk0 plane')
sup.Plot.simulate_hk0()
plt.clim([0,10])
plt.show()
print('  Plotting multicrystal powder')
xtls.Plot.simulate_powder(energy_kev = 5.0, peak_width=0.001)
plt.show()

# 17) FDMNES
stop
print('\nTest FDMNES code')
fdm = dif.Fdmnes(xtl2)  # this might take a while the first time as the fdmnes_win64.exe file is found
fdm.setup(
    folder_name='Test',  # this will create the directory /FDMNES/Sim/Test, but if it exists Test_2 will be used etc.
    comment='A test run',
    radius=4.0,
    edge='K',
    absorber='Cu',
    scf=False,
    quadrupole=False,
    azi_ref=[0,1,0],
    hkl_reflections=[[1,0,0],[0,1,3],[0,0,3]]
)
print('Create Files')
fdm.create_files()
fdm.write_fdmfile()
fdm.run_fdmnes()  # This will take a few mins, output should be printed to the console

# 18) Analysis FDMNES
print('\nAnalyse FDMNES Results')
ana = fdm.analyse()
ana.xanes.plot()
plt.show()
ana.I100sp.plot3D()
plt.show()
ana.density.plot()
plt.show()

