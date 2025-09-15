"""
SuperDuper Set of Tests of Dans_Diffraction
"""

import os
import pytest
import matplotlib.pyplot as plt
import Dans_Diffraction as dif


@pytest.fixture
def example_crystal():
    return dif.structure_list.Na08CoO2_P63mmc()


@pytest.fixture
def magnetic_crystal():
    return dif.structure_list.Sr3LiRuO6_C2c()


@pytest.fixture
def superstructure(example_crystal):
    # 15) Generate superstructure
    P = [[3, 0, 0], [4, 5, 0], [0, 0, 1]]  # Stripe Supercell
    example_crystal.Atoms.occupancy[2] = 0
    example_crystal.Atoms.occupancy[3] = 1
    example_crystal.generate_structure()
    sup = example_crystal.generate_superstructure(P)
    sup.Structure.occupancy[[6, 16, 26, 77, 87, 107]] = 1  # Na1
    sup.Structure.occupancy[[8, 18, 38, 28, 48, 58, 139, 119, 149, 109, 89, 79]] = 0  # Na2
    return sup


def test_properties():
    weight = dif.fc.atom_properties('Co','Weight')
    print('\nCobalt Weight = %6.2f g' % weight[0])
    xsf = dif.fc.xray_scattering_factor('Co', 3)
    print('Cobalt x-ray scattering factor @ 3A = %6.2f' % xsf[0, 0])
    mff = dif.fc.magnetic_form_factor('Co', qmag=0.5)
    print('Cobalt magnetic form factor @ 0.5A = %6.2f' % mff[0, 0])
    att = dif.fc.attenuation(27, 8)
    print('Cobalt attenuation @ 8.00keV = %6.2f cm^2/g' % att)
    asf1, asf2 = dif.fc.atomic_scattering_factor('Co', 2.838)
    print('Cobalt atomic scattering factor @ 8.000keV: f1+if2 = %s' % complex(asf1, asf2))
    assert True


def test_build_crystal():
    print('\nBuilding a crystal manually...')
    xtl1 = dif.Crystal()
    xtl1.name = 'Oh! What a Lovely Crystal'
    xtl1.new_cell([2.8, 2.8, 6.0, 90, 90, 90])
    #xtl.new_atoms(u=[0,0.5], v=[0,0.5], w=[0,0.25], type=['Na','O'], label=['Na1','O1'], occupancy=None, uiso=None, mxmymz=None)
    xtl1.Atoms.changeatom(0, u=0,   v=0,   w=0,    type='Na', label='Na1',occupancy=None, uiso=None, mxmymz=None) # there is an Fe ion added by default
    xtl1.Atoms.addatom(u=0.5, v=0.5, w=0.25, type='O', label='O1', occupancy=None, uiso=None, mxmymz=None)
    xtl1.Symmetry.addsym('x,y,z+1/2')
    xtl1.generate_structure()  # apply symmetry to atomic positions.
    print(xtl1.info())
    assert True

    # Test spacegroups
    print('Loading spacegroup 194')
    spg = dif.fc.spacegroup(194)
    print('Spacegroup %s: %s, nsyms=%d' % (
    spg['space group number'], spg['space group name'], len(spg['general positions'])))
    mspg = dif.fc.spacegroup_magnetic(spg['magnetic space groups'][1])
    print('Magnetic Spacegroup %s: %s, nsyms=%d' % (
    mspg['space group number'], mspg['space group name'], len(mspg['operators magnetic'])))
    xtl1.Symmetry.load_spacegroup(61)
    xtl1.Symmetry.load_magnetic_spacegroup(spg['magnetic space groups'][1])
    assert True


def test_cif_reading():
    # 6) CIF Reading
    ciffile = dif.structure_list.Na08CoO2_P63mmc.filename
    print('\nBuilding a crystal from a cif file: %s' % ciffile)
    files = list(dif.classes_structures.cif_list())
    print('NB %d cif files are included' % len(files))
    xtl2 = dif.Crystal(ciffile)
    print(xtl2.info())
    assert True


def test_mcif_reading():
    # 7) mCIF Reading
    ciffile = dif.structure_list.Sr3LiRuO6_C2c.filename
    print('\nBuilding a crystal from an mcif file: %s'%ciffile)
    files = list(dif.classes_structures.cif_list())
    xtlm = dif.Crystal(ciffile)
    print(xtlm.info())
    assert True


def test_cif_writing(example_crystal):
    # 8) CIF Writing
    example_crystal.name = 'Testing Testing'
    example_crystal.write_cif('Test.cif', comments='This is a test\n it is not interesting')
    print('\nWriting CIF to: test.cif')
    xtl3 = dif.Crystal('Test.cif')
    print('Test.cif loaded succesfully')
    os.remove('Test.cif')
    assert True


def test_xray_scattering(example_crystal):
    # 9) X-ray scattering
    print('\nTest x-ray scattering:')
    example_crystal.Scatter.hkl([[1, 0, 0], [2, 0, 0], [1, 1, 0], [0, 0, 1]], 8.0)

    # 10) Powder spectrum
    print('\nTest calculation of powder pattern:')
    tth_mesh, i_mesh, refs = example_crystal.Scatter.powder(units='twotheta', energy_kev=8)
    for ref in refs:
        print('(%3.0f,%3.0f,%3.0f)  tth=%5.2f  inten=%8.2f' % (ref[0], ref[1], ref[2], ref[3], ref[4]))
    assert True


def test_magnetic_scattering(magnetic_crystal):
    # 11) Magnetic scattering
    print('\nTest Magnetic scattering:')
    magnetic_crystal.Scatter.print_intensity([0, 0, 3])
    assert True

    # 11) Resonant scattering
    resi = magnetic_crystal.Scatter.xray_resonant(
        HKL=[0, 0, 3],
        energy_kev=2.967,
        polarisation='sp',
        azim_zero=(0, 1, 0),
        PSI=[90],
        F0=1, F1=1, F2=1
    )
    print('\nResonant scattering RuL2 psi=90 sp = %6.2f' % resi)
    assert True


def test_xray_dispersion(example_crystal):
    # 12) X-ray scattering with dispersion corrections
    en = example_crystal.Properties.Co.K
    inten = example_crystal.Scatter.xray_dispersion([0, 0, 2], en)
    print('\nX-ray scattering with dispersion correction (002) at %s keV: %s' % (en, inten))
    assert True


def test_mutliple_scattering(magnetic_crystal):
    # 13) Multiple scattering
    psi, ms_intensity = magnetic_crystal.Scatter.ms_azimuth([0,0,2], 2.967, [1,0,0])
    print('\nMultiple scattering (0,0,2) RuL2, max(intensity) = %s' % ms_intensity.max())
    assert True


# def test_tensor_scattering(magnetic_crystal):
#     # 14) Tensor scattering
#     ss, sp, ps, pp = magnetic_crystal.Scatter.tensor_scattering('Ru1_1', [0,0,3], 2.838, [0,1,0], psideg=np.arange(-180, 180, 1))
#     print('\nTensor scattering from Ru1_1 at (003), RuL3, max sp = %s' % sp.max())
#     assert True


def test_superstructure(superstructure):
    # 15) Generate superstructure
    print('\nSuperstructure:\n%s' % superstructure)
    assert True


def test_multi_crystal(example_crystal):
    # 16) Multicrystal
    xtls = example_crystal + dif.structure_list.Diamond() + dif.structure_list.Aluminium()
    print('\nMulticrystal:\n%s' % xtls)
    assert True


def test_plotting(example_crystal, magnetic_crystal, superstructure):
    plt.ion()

    # 17) Plotting
    print('\nStarting Plotting Tests...')
    print('  Plotting Powder')
    example_crystal.Plot.simulate_powder(energy_kev=8, peak_width=0.01)
    assert True

    print(' Manual Powder Plot')
    tth_mesh, i_mesh, refs = example_crystal.Scatter.powder(units='twotheta', energy_kev=8)

    plt.figure(figsize=[14, 6], dpi=60)
    plt.plot(tth_mesh, i_mesh)
    dif.fp.labels('Scatter.powder()', 'Two-Theta [deg]', 'Intensity')
    for ref in refs:
        if ref[-1] > 10:
            plt.text(ref[-2], ref[-1], '(%1.0f,%1.0f,%1.0f)' % (ref[0], ref[1], ref[2]))
    plt.show()
    assert True

    print('  Plotting hk0 plane')
    example_crystal.Plot.simulate_hk0()
    plt.show()
    print('  Plotting h0l plane')
    example_crystal.Plot.simulate_h0l()
    plt.show()
    print('  Plotting Crystal structure')
    example_crystal.Plot.plot_crystal()
    plt.show()
    print('  Plotting azimuthal scan')
    magnetic_crystal.Plot.simulate_azimuth([0,0,3])
    plt.show()
    assert True

    print('  Plotting multiple scattering')
    magnetic_crystal.Plot.plot_multiple_scattering([0,0,2], [1,0,0], energy_range=[2.95, 2.98], numsteps=61)
    plt.show()
    #print('  Plotting tensor scattering')
    #xtlm.Plot.tensor_scattering_azimuth('Ru1_1', [0,0,3], 2.838, [0,1,0])
    #plt.show()
    assert True

    print('  Plotting Superstructure hk0 plane')
    superstructure.Plot.simulate_hk0()
    plt.clim([0, 10])
    plt.show()
    assert True

    print('  Plotting multicrystal powder')
    xtls = example_crystal + dif.structure_list.Diamond() + dif.structure_list.Aluminium()
    xtls.Plot.simulate_powder(energy_kev=5.0, peak_width=0.001)
    plt.show()
    assert True


# def test_fdmnes(example_crystal):
#     # 18) FDMNES
#     print('\nTest FDMNES code')
#     fdm = dif.Fdmnes(example_crystal)  # this might take a while the first time as the fdmnes_win64.exe file is found
#     fdm.setup(
#         folder_name='Test',  # this will create the directory /FDMNES/Sim/Test, but if it exists Test_2 will be used etc.
#         comment='A test run',
#         radius=4.0,
#         edge='K',
#         absorber='Cu',
#         scf=False,
#         quadrupole=False,
#         azi_ref=[0, 1, 0],
#         hkl_reflections=[[1, 0, 0], [0, 1, 3], [0, 0, 3]]
#     )
#     print('Create Files')
#     fdm.create_files()
#     fdm.write_fdmfile()
#     fdm.run_fdmnes()  # This will take a few mins, output should be printed to the console
#
#     # 18) Analysis FDMNES
#     print('\nAnalyse FDMNES Results')
#     ana = fdm.analyse()
#     ana.xanes.plot()
#     plt.show()
#     ana.I100sp.plot3D()
#     plt.show()
#     ana.density.plot()
#     plt.show()
#     assert True

