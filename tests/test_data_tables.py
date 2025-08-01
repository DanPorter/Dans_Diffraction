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

