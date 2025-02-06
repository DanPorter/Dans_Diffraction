import pytest

import Dans_Diffraction as dif

from .load_data import STRUCTURES


def test_structures():
    for xtl in dif.structure_list:
        assert xtl.name in STRUCTURES, f"{xtl.name} not generated in cif output"
        assert str(xtl) == STRUCTURES[xtl.name], f"{xtl.name} output changed"
