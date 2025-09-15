import pytest

import Dans_Diffraction as dif
from Dans_Diffraction import functions_lattice as fl


from .load_data import LATTICE, HKL, VestaData


@pytest.fixture
def vesta_data():
    return {latt: VestaData(latt) for latt in LATTICE}


def test_lattice(vesta_data):
    for latt, latpar in LATTICE.items():
        xtl = dif.structure_list.triclinic()
        xtl.Cell.latt(**latpar)
        xtl.Scatter.setup_scatter(  # values match vesta
            wavelength_a=1.54059,
            scattering_factors='waaskirf',
            scattering_lengths='Sears',
            output=False
        )
        sf = xtl.Scatter.structure_factor(HKL)
        for hkl, sf_calc in zip(HKL, sf):
            sf_vesta = vesta_data[latt].get_sf(hkl)
            print(f"{latt}: {hkl}: vesta={sf_vesta}, DD={sf_calc}")
            assert abs(sf_calc - sf_vesta) < 0.01, f"SF wrong for {hkl}, {latt}"


def test_rutile():
    vesta_hkl = [[1, 1, 0], [0, 1, 1], [0, 2, 0], [1, 1, 1]]
    vesta_xray = [37.0531, 23.4395, 13.9434, 17.2707]  # structure factor
    vesta_neutron = [3.97176, 14.0359, 23.4644, 19.4905]

    xtl = dif.structure_list.Rutile()

    xtl.Atoms.changeatom([0, 1], uiso=0.0126651)  # B=1
    xtl.generate_structure()

    xtl.Scatter.setup_scatter(  # values match vesta
        wavelength_a=1.54059,
        scattering_factors='waaskirf',
        scattering_lengths='Sears',
        output=False
    )

    for hkl, sf_xray, sf_neutron in zip(vesta_hkl, vesta_xray, vesta_neutron):
        calc_xray = xtl.Scatter.intensity(hkl, scattering_type='xray')[0]
        calc_neutron = xtl.Scatter.intensity(hkl, scattering_type='neutron')[0]
        # calc_xray = xtl.Scatter.x_ray(hkl)[0]
        # calc_neutron = xtl.Scatter.neutron(hkl)[0]
        print(f"{hkl}:    xray: {sf_xray ** 2:.2f} - {calc_xray:.2f}")
        print(f"{hkl}: neutron: {sf_neutron ** 2:.2f} - {calc_neutron:.2f}")
        assert abs(calc_xray - sf_xray**2) < 0.01, f"x-ray intensity wrong for hkl"
        assert abs(calc_neutron - sf_neutron ** 2) < 0.01, f"neutron intensity wrong for hkl"


def test_rutile_xray():
    vesta = VestaData('rutile_xray')
    hkl = vesta.get_hkl()
    vesta_sf2 = vesta.data[:, 6] ** 2  # |F|**2

    xtl = dif.structure_list.Rutile()

    xtl.Scatter.setup_scatter(wavelength_a=1.54059, scattering_factors='waaskirf', output=False)
    calc_xray = xtl.Scatter.intensity(hkl, scattering_type='xray')

    for hh, vs, dd in zip(hkl, vesta_sf2, calc_xray):
        print(f"{hh}: xray: {vs:.2f} - {dd:.2f}")
    assert sum((vesta_sf2 - calc_xray) ** 2) < 1.0, 'difference in rutile x-ray intensities'


def test_rutile_neutron():
    # Note, Vesta uses neutron scattering lengths from Sears [ITC Vol.C, 1995]
    vesta = VestaData('rutile_neutron')
    hkl = vesta.get_hkl()
    vesta_sf2 = vesta.data[:, 6] ** 2  # |F|**2

    xtl = dif.structure_list.Rutile()

    xtl.Scatter.setup_scatter(wavelength_a=1.54059, scattering_lengths='Sears', output=False)
    calc_neutron = xtl.Scatter.intensity(hkl, scattering_type='neutron')

    for hh, vs, dd in zip(hkl, vesta_sf2, calc_neutron):
        print(f"{hh}: neutron: {vs:.2f} - {dd:.2f}")
    assert sum((vesta_sf2 - calc_neutron) ** 2) < 0.01, 'difference in rutile neutron intensities'


def test_magnetic_mno():
    xtl = dif.structure_list.MnO()
    xtl.Scatter.setup_scatter(
        scattering_type='neutron polarised',
        polarisation_vector=[1, 0, 0]
    )
    assert abs(xtl.Scatter.intensity([1, 1, 1]) - 4332.39) < 0.01, 'incorrect polarised neutron intensity'

    assert abs(
        xtl.Scatter.intensity([1, 1, 1], scattering_type='xray magnetic')[0] - 25994.32
    ) < 0.01, 'incorrect magnetic xray intensity'
    assert abs(
        xtl.Scatter.intensity([1, 1, 1], scattering_type='neutron magnetic')[0] - 25994.32
    ) < 0.01, 'incorrect magnetic neutron intensity'


def test_magnetic_lmo():
    xtl = dif.structure_list.LaMnO3()
    hkl = [[0,0,1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]]
    exp = [0, 0, 215.507397, 0, 148.85200365]
    cal = xtl.Scatter.intensity(hkl, scattering_type='xray magnetic')
    assert all(abs(cal - exp) < 0.001), 'incorrect magnetic x-ray intensity'
    exp = [0, 0, 215.507397, 0, 92.48078345]
    cal = xtl.Scatter.intensity(hkl, scattering_type='neutron magnetic')
    assert all(abs(cal - exp) < 0.001), 'incorrect magnetic neutron intensity'

    # Comparison with https://taro-nakajima.github.io/fcal-n/#m-form_symbols
    # LaMnO3.cif with Mn3 <j2>/<j0>=2
    hkl_inten = [
        ([0, 0, 2], 0),
        ([0, 1, 0], 41.456),
        ([0, 1, 2], 38.630),
        ([0, 3, 0], 38.502),
        ([1, 1, 1], 31.652),
        ([2, 1, 0], 13.637),
    ]
    hkl = [h for h, ii in hkl_inten]
    cal = xtl.Scatter.intensity(hkl, scattering_type='neutron magnetic')
    # TODO: work out why this doesn't work
    # assert all(abs(cal - [ii**2 for h, ii in hkl_inten]) < 0.1), "neutron magnetic intensity doesnt match fcal-n"

