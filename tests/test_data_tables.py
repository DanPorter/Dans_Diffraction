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
    # TODO: add tests
    assert True


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
