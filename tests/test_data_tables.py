"""
Dans_Diffraction test of data loading from tables
"""

import pytest
import numpy as np
import Dans_Diffraction as dif


def test_xray_scattering_factor_resonant():
    elements = ['Co', 'O', 'Ru']
    q_mag = [2, 3, 4, 5]
    energy_kev = np.arange(7.5, 7.8, 0.05)
    data = dif.fc.xray_scattering_factor_resonant(elements, q_mag, energy_kev)
    assert data.shape == (len(q_mag), len(elements), len(energy_kev))
    assert abs(data.sum() - complex(1178.5990511, 111.8445339)) < 0.0001
    data = dif.fc.xray_scattering_factor_resonant(elements, q_mag, energy_kev, use_waaskirf=True)
    assert data.shape == (len(q_mag), len(elements), len(energy_kev))
    assert abs(data.sum() - complex(1178.6375568, 111.8445333)) < 0.0001


def test_old_new_functions():
    """tests old form factor functions and new ones are consistent"""

    elements = ['Co', 'O']
    q_mag = [0, 1, 2, 3]
    energy_kev = np.arange(7.5, 7.8, 0.05)

    # Neutron
    tables = ['ndb', 'sears']
    for table in tables:
        old_ff = dif.fc.neutron_scattering_length(elements, table)
        new_ff = dif.fc.custom_scattering_factor(elements, 0, None, default_table=table)
        new_ff = new_ff.squeeze()
        diff = np.sum(abs(old_ff - new_ff))
        print(table, old_ff.shape, new_ff.shape, diff)
        # TODO: check why this isn't smaller
        assert diff < 0.02, f"{table} is incorrect"

    # x-ray
    tables = {
        'itc': dif.fc.xray_scattering_factor,
        'waaskirf': dif.fc.xray_scattering_factor_WaasKirf,
        'peng': dif.fc.electron_scattering_factor,
    }
    for table, fun in tables.items():
        old_ff = fun(elements, q_mag)
        new_ff = dif.fc.custom_scattering_factor(elements, q_mag, None, default_table=table)
        new_ff = new_ff.squeeze()
        diff = np.sum(abs(old_ff - new_ff))
        print(table, old_ff.shape, new_ff.shape, diff)
        assert diff < 0.01, f"{table} is incorrect"

    # resonant x-ray
    old_ff = dif.fc.xray_scattering_factor_resonant(elements, q_mag, energy_kev)
    new_ff = dif.fc.custom_scattering_factor(elements, q_mag, energy_kev, default_table='itc')
    diff = np.sum(abs(old_ff - new_ff))
    print('resonant', old_ff.shape, new_ff.shape, diff)  # diff~6.5, due to interpolation difference in f2
    assert diff < 10, f"resonant scattering factors are incorrect"


def test_magnetic_form_factor():
    mff = dif.fc.load_magnetic_ff_coefs()
    elements = [
        # mmf label, element
        ('Co0', 'Co'),
        ('Co1', 'Co1+'),
        ('Co3', 'Co3+'),
        ('O', 'O'),
        ('Ru1', 'Ru1+'),
        ('U4', 'U4+')
    ]
    coefs = dif.fc.magnetic_ff_coefs(*(el for lab, el in elements))
    for n, (label, element) in enumerate(elements):
        if label in mff:
            assert all(abs(coefs[n] - list(mff[label].values())[:21]) < 0.001), f"form factor coefficients for {element} wrong"

    glande = dif.fc.magnetic_ff_g_factor(*(el for lab, el in elements))
    check = [1.3333, 1.5, 2, 0, 2, 1.5]
    assert all(abs(glande - check) < 0.01), 'incrorrect Lande g-factor'

    ratio = [dif.fc.magnetic_ff_j2j0_ratio(g) for g in glande]
    check = [0.5, 0.333, 0, 0, 0, 0.333]
    assert all(abs(r - c) < 0.01 for r, c in zip(ratio, check)), 'incorrect j2j0 ratio'

    qmag = [0, 4, 12]
    ff = dif.fc.magnetic_form_factor(*(el for lab, el in elements), qmag=qmag)
    check  = [
        [ 0.9979,  0.9996,  0.9998,  0.    ,  1.    ,  1.0001],
        [ 0.5198,  0.4582,  0.4902,  0.    ,  0.1202,  0.2987],
        [ 0.036 ,  0.0208, -0.0073,  0.    ,  0.0158,  0.0129]
    ]
    assert (abs(ff - check) < 0.001).all(), "incorrect form factors"


def test_custom_scattering_factor():
    coefs = (12.2841, 4.5791, 7.4409, 0.6784, 4.2034, 12.5359, 2.2488, 72.1692, 1.1118, 0.)
    table = np.array([
        [7.7, 7.71, 7.72, 7.73, 7.74, 7.75, 7.76, 7.77, 7.78, 7.79],  # keV
        [-9.6866, -11.0362, -10.1969, -9.3576, -8.5184, -7.6791, -6.8398, -6.0005, -5.1613, -4.322],  # f1
        [-0.4833, -3.7933, -3.7862, -3.7791, -3.772, -3.765, -3.7579, -3.7508, -3.7437, -3.7367]  # f2
    ])
    dif.fc.add_custom_form_factor_coefs('Co3+', *coefs, dispersion_table=table)

    elements = ['Co', 'Co3+']
    q_mag = [0, 1, 2, 3]
    energy_kev = np.arange(7.7, 7.8, 0.05)
    ff = dif.fc.custom_scattering_factor(elements, q_mag, energy_kev)
    diff = np.sum(np.diff(ff, axis=1))
    print(f"Difference between {elements[0]} and {elements[1]} = {diff}")
    assert abs(diff) > 6


def test_spacegroup_find():
    spg = dif.fc.find_spacegroup('P63/mmc')
    assert spg['space group number'] == 194
    spg = dif.fc.find_spacegroup('m-3m')  # Pm-3m
    assert len(spg['general positions']) == 48
    spg = dif.fc.find_spacegroup('m\'')
    assert spg['space group name'] == "P2/m'"
    spg = dif.fc.find_spacegroup('218.83')  # P-4'3n'
    assert len(spg['operators magnetic']) == 24
