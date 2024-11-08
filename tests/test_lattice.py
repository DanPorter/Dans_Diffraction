import pytest

import Dans_Diffraction as dif
from Dans_Diffraction import functions_lattice as fl

from .load_data import LATTICE, HKL, VOLUMES, VestaData


@pytest.fixture
def vesta_data():
    return {latt: VestaData(latt) for latt in LATTICE}


def test_dspace(vesta_data):
    for latt, latpar in LATTICE.items():
        for hkl in HKL:
            calc_dspace = fl.dspacing(*hkl, **latpar)
            check_dspace = vesta_data[latt].get_dspace(hkl)
            assert abs(calc_dspace - check_dspace) < 0.01, f"d-space wrong for {hkl}, {latt}"


def test_volume():
    for latt, latpar in LATTICE.items():
        calc_volume = fl.lattice_volume(**latpar)
        check_volume = VOLUMES[latt]
        assert abs(calc_volume - check_volume) < 0.01, f"volume wrong for {latt}"


def test_basis_options(vesta_data):
    for n in range(3):
        basis_fun = fl.choose_basis(n + 1)
        for latt, latpar in LATTICE.items():
            lp = fl.gen_lattice_parameters(**latpar)
            basis = basis_fun(**latpar)
            basis_lp = fl.basis2latpar(basis)
            assert sum(abs(a-b) for a, b in zip(lp, basis_lp)) < 0.01, f"lattice parameters wrong for basis {n+1}, {latt}"


def test_crystal(vesta_data):
    basis_options = ['materialsproject', 'vesta', 'busingandlevy']
    for latt, latpar in LATTICE.items():
        xtl = dif.structure_list.triclinic()
        xtl.Cell.latt(**latpar)
        for opt in basis_options:
            xtl.Cell.choose_basis(opt)
            for hkl in HKL:
                calc_dspace = xtl.Cell.dspace(hkl)
                check_dspace = vesta_data[latt].get_dspace(hkl)
                assert abs(calc_dspace - check_dspace) < 0.01, f"d-space wrong for {hkl}, {latt}"

