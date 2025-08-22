# -*- coding: utf-8 -*-
"""
Module: functions_scattering.py

Functions:
intensity(structurefactor)
    Returns the squared structure factor
phase_factor(hkl, uvw)
    Return the complex phase factor:
phase_factor_qr(q, r)
    Return the complex phase factor:
scatteringbasis(q, azi_ref_q=(1, 0, 0), psi=0)
    Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
scatteringcomponents(moment, q, azi_ref_q=(1, 0, 0), psi=0)
    Transform magnetic vector into components within the scattering plane
scatteringvectors(q, energy_kev, azi_ref_q=(1, 0, 0), psi=0, polarisation='s-p')
    Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
sf_magnetic_neutron(q, r, occ, moment, magnetic_formfactor=None, debyewaller=False)
    Calculate the magnetic structure factor for the given HKL, using neutron magnetic form factor
sf_magnetic_neutron_polarised(q, r, occ, moment, incident_polarisation_vector=(1, 0, 0), magnetic_formfactor=None, debyewaller=False)
    Calculate the magnetic structure factor for the given HKL, using neutron magnetic form factor
sf_magnetic_xray(q, r, occ, moment, magnetic_formfactor=None, debyewaller=False)
    Calculate the non-resonant magnetic component of the structure factor
sf_magnetic_xray_beamline(q, r, occ, moment, energy_kev, magnetic_formfactor=None, debyewaller=False, azi_ref_q=(1, 0, 0), psi=0, polarisation='s-p')
    Calculate the non-resonant magnetic component of the structure factor
sf_magnetic_xray_polarised(q, r, occ, moment, incident_polarisation_vector=(1, 0, 0), magnetic_formfactor=None, debyewaller=False)
    Calculate the non-resonant magnetic component of the structure factor
sf_magnetic_xray_resonant(q, r, occ, moment, energy_kev, debyewaller=False, azi_ref_q=(1, 0, 0), psi=0, polarisation='s-p', f0=0, f1=1, f2=0)
    Calculate the non-resonant magnetic component of the structure factor
sf_magnetic_xray_resonant_alternate(q, r, occ, moment, energy_kev, debyewaller=False, polarisation='sp', azi_ref_q=(1, 0, 0), psi=0, f0=1, f1=1, f2=1)
    Calculate structure factors using resonant scattering factors in the dipolar approximation
structure_factor(scattering_factor, occupancy, debyewaller, phase)
    Return the complex structure factor:
xray_resonant_scattering_factor(q, moment, energy_kev, polarisation='sp', flm=(1, 1, 1), psi=0, azi_ref_q=(1, 0, 0))
    Calcualte fxres, the resonant x-ray scattering factor


By Dan Porter, PhD
Diamond
2018

Version 1.1
Last updated: 05/08/25

Version History:
11/11/18 0.1    Version History started.
13/07/21 0.9    Functions re-written and tested
06/02/25 1.0    Removed refrences to unpolarised magnetic scattering due to incorrect averaging
05/08/25 1.1    Added custom scattering option

@author: DGPorter
"""

import numpy as np
import datetime

from . import functions_general as fg
from . import functions_crystallography as fc

__version__ = '1.1'

MAX_QR_ARRAY = 1.0e7
TIME_REPORT = True
DEBUG_MODE = False
SCATTERING_TYPES = {
    'xray': ['xray', 'x', 'x-ray', 'thomson', 'charge'],
    'xray fast': ['xray fast', 'xfast'],
    'neutron': ['neutron', 'n', 'nuclear'],
    'xray magnetic': ['xray magnetic', 'magnetic xray', 'spin xray', 'xray spin'],
    'neutron magnetic': ['neutron magnetic', 'magnetic neutron', 'magnetic'],
    'neutron polarised': ['neutron polarised', 'neutron polarized'],
    'xray polarised': ['xray polarised', 'xray polarized'],
    'xray resonant': ['xray resonant', 'resonant', 'resonant xray', 'rxs'],
    'xray dispersion': ['dispersion', 'xray dispersion'],
    'electron': ['electron', 'ele', 'e'],
    'custom': ['custom'],
}


def _debug(message):
    if DEBUG_MODE:
        print(message)


def phase_factor(hkl, uvw):
    """
    Return the complex phase factor:
        phase_factor = exp(i.2.pi.HKL.UVW')
    :param hkl: array [n,3] integer reflections
    :param uvw: array [m,3] atomic positions in atomic basis units
    :return: complex array [n,m]
    """

    hkl = np.asarray(np.rint(hkl), dtype=float).reshape([-1, 3])
    uvw = np.asarray(uvw, dtype=float).reshape([-1, 3])

    dotprod = np.dot(hkl, uvw.T)
    return np.exp(1j * 2 * np.pi * dotprod)


def phase_factor_qr(q, r):
    """
    Return the complex phase factor:
        phase_factor = exp(i.Q.R')
    :param q: array [n,3] reflection positions in A^-1
    :param r: array [m,3] atomic positions in A
    :return: complex array [n,m]
    """

    q = np.asarray(q, dtype=float).reshape([-1, 3])
    r = np.asarray(r, dtype=float).reshape([-1, 3])

    dotprod = np.dot(q, r.T)
    return np.exp(1j * dotprod)


def structure_factor(scattering_factor, occupancy, debyewaller, phase):
    """
    Return the complex structure factor:
        structure_factor = sum_i( f.occ.dw.phase )
    :param scattering_factor: array [n,m] or [n]: radiation dependent scattering factor/ form factor,/ scattering length
    :param occupancy: array [m]: occupancy of each atom
    :param debyewaller: array [n,m]: thermal vibration factor of each atom and reflection
    :param phase: complex array [n,m]: complex phase factor exp(-i.Q.R)
    :return: complex array [n]
    """
    #nrefs, natoms = phase.shape
    #scattering_factor = np.asarray(scattering_factor).reshape([-1, natoms])
    #occupancy = np.asarray(occupancy, dtype=float).reshape([1, natoms])
    return np.sum(scattering_factor * occupancy * debyewaller * phase, axis=1)


def intensity(structurefactor):
    """
    Returns the squared structure factor
    :param structurefactor: complex array [n] structure factor
    :return: array [n]
    """
    return np.real(structurefactor * np.conj(structurefactor))


########################################################################################################################
# ----------------------------------  NONMAGNETIC STRUCTURE FACTORS  ------------------------------------------------- #
########################################################################################################################


def sf_atom(q, r, scattering_factor=None, occ=None, debyewaller=None, **kwargs):
    """

    :param q: array [n,3] reflection positions in A^-1
    :param r: array [m,3] atomic positions in A
    :param scattering_factor: array [n,m] or [n]: radiation dependent scattering factor/ form factor,/ scattering length
    :param occ: array [m]: occupancy of each atom
    :param debyewaller: array [n,m]: thermal vibration factor of each atom and reflection
    :param kwargs: additional options[*unused]
    :return: complex array [n]
    """
    phase = phase_factor_qr(q, r)
    if scattering_factor is None:
        scattering_factor = np.ones(phase.shape)
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    _debug('sf_atom(phase.shape=%s)' % (phase.shape,))
    return structure_factor(scattering_factor, occ, debyewaller, phase)


def sf_xray_dispersion(q, r, scattering_factor, occ=None, debyewaller=None, **kwargs):
    """
    Calculate the resonant x-ray structure factor
    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param scattering_factor: array [n,m,e]: energy dependent complex atomic form factor
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param kwargs: additional options[*unused]
    :return sf: [n, e] complex array of structure factors
    """
    phase = phase_factor_qr(q, r)
    scattering_factor = np.asarray(scattering_factor, dtype=complex).reshape(*phase.shape, -1)
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)

    neng = scattering_factor.shape[2]
    _debug('sf_xray_dispersion(phase.shape=%s, energies=%s)' % (phase.shape, neng))
    sf = np.zeros([len(q), neng], dtype=complex)
    for engval in range(neng):
        sf[:, engval] = structure_factor(scattering_factor[:, :, engval], occ, debyewaller, phase)
    if neng == 1:
        return sf[:, 0]
    return sf


def sf_custom(q, r, scattering_factor, occ=None, debyewaller=None, **kwargs):
    """
    Calculate the structure factor using customised atomic scattering factors
    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param scattering_factor: array [n,m,e]: energy dependent complex atomic form factor
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param kwargs: additional options[*unused]
    :return sf: [n, e] complex array of structure factors
    """
    phase = phase_factor_qr(q, r)
    scattering_factor = np.asarray(scattering_factor, dtype=complex).reshape(*phase.shape, -1)
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)

    neng = scattering_factor.shape[2]
    _debug('sf_custom(phase.shape=%s, energies=%s)' % (phase.shape, neng))
    sf = np.zeros([len(q), neng], dtype=complex)
    for engval in range(neng):
        sf[:, engval] = structure_factor(scattering_factor[:, :, engval], occ, debyewaller, phase)
    if neng == 1:
        return sf[:, 0]
    return sf


########################################################################################################################
# -----------------------------------  MAGNETIC STRUCTURE FACTORS  --------------------------------------------------- #
########################################################################################################################


def sf_magnetic_neutron(q, r, moment, magnetic_formfactor=None, occ=None,  debyewaller=None, **kwargs):
    """
    ***Not currently used because of incorrect method of averaging polarisations***
    Calculate the magnetic structure factor for the given HKL, using neutron magnetic form factor
    Assumes an unpolarised incident beam.
        Reference: G. L. Squires, Introduction to the Theory of Thermal Neutron Scattering (Cambridge University Press, 1997).
    :param q: [n,3] array of reflections in cartesian coordinate system in units of inverse-A
    :param r: [m,3] array of atomic positions in A
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param magnetic_formfactor: [n,m] array of magnetic form factors for each atom and relection
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection, or False to omit
    :param kwargs: additional options[*unused]
    :return sf: [n] complex array of structure factors
    """

    phase = phase_factor_qr(q, r)
    moment = np.asarray(moment, dtype=float).reshape((-1, 3))
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if magnetic_formfactor is None:
        magnetic_formfactor = np.ones(phase.shape)

    # direction of q
    qhat = fg.norm(q).reshape([-1, 3])

    # Calculate structure factor
    _debug('sf_magnetic_neutron(phase.shape=%s)' % (phase.shape,))
    sf = np.zeros(len(q), dtype=complex)
    for n, qh in enumerate(qhat):
        sfm = np.array([0., 0., 0.])
        for m, mom in enumerate(moment):
            # Calculate Magnetic part
            qm = mom - np.dot(qh, mom) * qh  # [mx, my, mz]

            # Calculate structure factor
            sfm = sfm + (magnetic_formfactor[n, m] * debyewaller[n, m] * occ[m] * phase[n, m] * qm)

        # Calculate polarisation with incident neutron
        # sf[n] = np.dot(sfm, incident_polarisation_vector)
        # sf[n] = np.dot(sfm, sfm)  # maximum possible
        # average polarisation  # 6/2/25: this method is incorrect for averaging polarisations, should combine intensity
        # sf[n] = (np.dot(sfm, [1, 0, 0]) + np.dot(sfm, [0, 1, 0]) + np.dot(sfm, [0, 0, 1])) / 3
        # magnitude of moment structure factor
        sf[n] = np.sqrt(np.dot(sfm, sfm))
    return sf


def sf_magnetic_neutron_polarised(q, r, moment, incident_polarisation_vector=(1, 0, 0),
                                  magnetic_formfactor=None, occ=None, debyewaller=None, **kwargs):
    """
    Calculate the magnetic structure factor for the given HKL, using neutron magnetic form factor
        Reference: G. L. Squires, Introduction to the Theory of Thermal Neutron Scattering (Cambridge University Press, 1997).
    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param incident_polarisation_vector: [1,3] direction of incident polarisation
    :param magnetic_formfactor: [n,m] array of magnetic form factors for each atom and relection
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection, or False to omit
    :param kwargs: additional options[*unused]
    :return sf: [n] complex array of structure factors
    """

    phase = phase_factor_qr(q, r)
    moment = np.asarray(moment, dtype=float).reshape((-1, 3))
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if magnetic_formfactor is None:
        magnetic_formfactor = np.ones(phase.shape)

    # direction of q
    qhat = fg.norm(q).reshape([-1, 3])

    # Calculate structure factor
    _debug('sf_magnetic_neutron_polarised(phase.shape=%s)' % (phase.shape,))
    sf = np.zeros(len(q), dtype=complex)
    for n, qh in enumerate(qhat):
        sfm = np.array([0., 0., 0.])
        for m, mom in enumerate(moment):
            # Calculate Magnetic part
            qm = mom - np.dot(qh, mom) * qh

            # Calculate structure factor
            sfm = sfm + (magnetic_formfactor[n, m] * debyewaller[n, m] * occ[m] * phase[n, m] * qm)

        # Calculate polarisation with incident neutron
        sf[n] = np.dot(sfm, incident_polarisation_vector)
    return sf


def sf_magnetic_xray(q, r, moment, magnetic_formfactor=None, occ=None, debyewaller=None, **kwargs):
    """
    ***Not currently used because of incorrect method of averaging polarisations***
    Calculate the non-resonant magnetic component of the structure factor
    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param magnetic_formfactor: [n,m] array of magnetic form factors for each atom and relection
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param kwargs: additional options[*unused]
    :return sf: [n] complex array of structure factors

        f_non-res_mag = i.r0.(hw/mc^2).fD.[.5.L.A + S.B]
        B = e_o X e_i + (k_o X e_o) * k_o.e_i - (k_i X e_i) * k_i.e_o - (k_o X e_o) X (k_i X e_i)
    - ignore orbital moment L
    - fD = magnetic form factor
    - S = spin moment
    - k_i, k_o = wavevector in, out
    - e_i, e_o = polarisation in, out
    From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
    Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
    No orbital component assumed
    magnetic moments assumed to be in the same reference frame as the polarisation
    """

    phase = phase_factor_qr(q, r)
    moment = np.asarray(moment, dtype=float).reshape((-1, 3))
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if magnetic_formfactor is None:
        magnetic_formfactor = np.ones(phase.shape)

    # Calculate structure factor
    _debug('sf_magnetic_xray(phase.shape=%s)' % (phase.shape,))
    sf = np.zeros(len(q), dtype=complex)
    for n in range(len(q)):
        # Calculate vector structure factor
        sfm = np.array([0., 0., 0.])
        for m, mom in enumerate(moment):
            sfm = sfm + magnetic_formfactor[n, m] * debyewaller[n, m] * occ[m] * phase[n, m] * mom

        # average polarisation # 6/2/25: this method is incorrect for averaging polarisations, should combine intensity
        # sf[n] = (np.dot(sfm, [1, 0, 0]) + np.dot(sfm, [0, 1, 0]) + np.dot(sfm, [0, 0, 1])) / 3
        # magnitude of moment structure factor
        sf[n] = np.sqrt(np.dot(sfm, sfm))
    return sf


def sf_magnetic_xray_polarised(q, r, moment, incident_polarisation_vector=(1, 0, 0),
                               magnetic_formfactor=None, occ=None, debyewaller=None, **kwargs):
    """
    Calculate the non-resonant magnetic component of the structure factor

        f_non-res_mag = i.r0.(hw/mc^2).fD.[.5.L.A + S.B]
        B = e_o X e_i + (k_o X e_o) * k_o.e_i - (k_i X e_i) * k_i.e_o - (k_o X e_o) X (k_i X e_i)

    - ignore orbital moment L
    - fD = magnetic form factor
    - S = spin moment
    - k_i, k_o = wavevector in, out
    - e_i, e_o = polarisation in, out

    From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
    Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
    No orbital component assumed
    magnetic moments assumed to be in the same reference frame as the polarisation

    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param incident_polarisation_vector: [1,3] direction of incident polarisation
    :param magnetic_formfactor: [n,m] array of magnetic form factors for each atom and relection
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param kwargs: additional options[*unused]
    :return sf: [n] complex array of structure factors
    """

    phase = phase_factor_qr(q, r)
    moment = np.asarray(moment, dtype=float).reshape((-1, 3))
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if magnetic_formfactor is None:
        magnetic_formfactor = np.ones(phase.shape)

    # Calculate structure factor
    _debug('sf_magnetic_xray_polarised(phase.shape=%s)' % (phase.shape,))
    sf = np.zeros(len(q), dtype=complex)
    for n in range(len(q)):
        # Calculate vector structure factor
        sfm = np.array([0., 0., 0.])
        for m, mom in enumerate(moment):
            sfm = sfm + magnetic_formfactor[n, m] * debyewaller[n, m] * occ[m] * phase[n, m] * mom

        # Calculate polarisation with incident x-ray
        # The reference frame of the x-ray and the crystal are assumed to be the same
        # i.e. pol=[1,0,0] || mom=[1,0,0] || (1,0,0)
        sf[n] = np.dot(sfm, incident_polarisation_vector)
    return sf


def sf_magnetic_xray_beamline(q, r, moment, energy_kev, magnetic_formfactor=None, occ=None, debyewaller=None,
                              azi_ref_q=(1, 0, 0), psi=0, polarisation='s-p', **kwargs):
    """
    Calculate the non-resonant magnetic component of the structure factor, using standard beamline polarisation

        f_non-res_mag = i.r0.(hw/mc^2).fD.[.5.L.A + S.B]
        B = e_o X e_i + (k_o X e_o) * k_o.e_i - (k_i X e_i) * k_i.e_o - (k_o X e_o) X (k_i X e_i)

    - ignore orbital moment L
    - fD = magnetic form factor
    - S = spin moment
    - k_i, k_o = wavevector in, out
    - e_i, e_o = polarisation in, out

    From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
    Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
    No orbital component assumed
    magnetic moments assumed to be in the same reference frame as the polarisation

    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param energy_kev: float value of incident x-ray energy in keV
    :param magnetic_formfactor: [n,m] array of magnetic form factors for each atom and relection
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param azi_ref_q: [1,3] azimuthal refence, in cartesian basis (Q)
    :param psi: [p] array of azimthal angles - the rotation out of the scattering plane.
    :param polarisation: str definition of the polarisation can be: ['ss','sp','ps','pp'] with 's'=sigma, 'p'=pi
    :param kwargs: additional options[*unused]
    :return sf: [n, p] complex array of structure factors for different reflections and azimuths
    """

    phase = phase_factor_qr(q, r)
    moment = np.asarray(moment, dtype=float).reshape((-1, 3))
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if magnetic_formfactor is None:
        magnetic_formfactor = np.ones(phase.shape)
    psi_array = np.asarray(psi, dtype=float).reshape([-1])
    npsi = len(psi_array)

    _debug('sf_magnetic_xray_beamline(phase.shape=%s, npsi=%d)' % (phase.shape, npsi))
    sf = np.zeros([len(q), npsi], dtype=complex)
    for p, psival in enumerate(psi_array):
        kin, kout, ein, eout = scatteringvectors(q, energy_kev, azi_ref_q, psival, polarisation)

        # Magnetic form factor
        # f_non-res_mag = i.r0.(hw/mc^2).fD.[.5.L.A + S.B] #equ 2 Hill+McMorrow 1996
        # ignore orbital moment L
        fspin = np.zeros([len(q), len(r)], dtype=complex)
        for n in range(len(q)):
            B = np.cross(eout[n], ein[n]) + \
                np.cross(kout[n], eout[n]) * np.dot(kout[n], ein[n]) - \
                np.cross(kin[n], ein[n]) * np.dot(kin[n], eout[n]) - \
                np.cross(np.cross(kout[n], eout[n]), np.cross(kin[n], ein[n]))
            fspin[n, :] = 1j * magnetic_formfactor[n, :] * np.dot(moment, B)
        sf[:, p] = np.sum(fspin * occ * debyewaller * phase, axis=1)
    if npsi == 1:
        return sf[:, 0]
    return sf


def sf_magnetic_xray_resonant(q, r, moment, energy_kev, occ=None, debyewaller=None, azi_ref_q=(1, 0, 0), psi=0,
                              polarisation='sp', f0=0, f1=1, f2=0, **kwargs):
    """
    Calculate the non-resonant magnetic component of the structure factor

    f_res_mag = [(e'.e)F0 - i(e'xe).Z*F1 + (e'.Z)*(e.Z)*F2]

    From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
    Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
    No orbital component assumed
    magnetic moments assumed to be in the same reference frame as the polarisation

    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param energy_kev: float value of incident x-ray energy in keV
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param azi_ref_q: [1,3] azimuthal refence, in cartesian basis (Q)
    :param psi: [p] array of azimthal angles - the rotation out of the scattering plane.
    :param polarisation: str definition of the polarisation can be: ['ss','sp','ps','pp'] with 's'=sigma, 'p'=pi
    :param f0: float Flm value 0 (charge)
    :param f1: float Flm value 1
    :param f2: float Flm value 2
    :param kwargs: additional options[*unused]
    :return sf: [n, p] complex array of structure factors for different reflections and azimuths
    """

    phase = phase_factor_qr(q, r)
    moment = fg.norm(moment).reshape((-1, 3))
    z = fg.norm(moment)  # z^n is a unit vector in the direction of the magnetic moment of the nth ion.
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    psi_array = np.asarray(psi, dtype=float).reshape([-1])
    npsi = len(psi_array)

    _debug('sf_magnetic_xray_resonant(phase.shape=%s, npsi=%d)' % (phase.shape, npsi))
    sf = np.zeros([len(q), npsi], dtype=complex)
    for p, psival in enumerate(psi_array):
        kin, kout, ein, eout = scatteringvectors(q, energy_kev, azi_ref_q, psival, polarisation)

        fe1e1 = np.zeros([len(q), len(r)], dtype=complex)
        flm0, flm1, flm2 = 0, 0, 0
        for ref in range(len(q)):
            # z = scatteringcomponents(moment, q[ref], azi_ref_q, psi)
            # Magnetic form factor
            # f_res_mag = [(e'.e)F0 - i(e'xe).Z*F1 + (e'.Z)*(e.Z)*F2] #equ 7 Hill+McMorrow 1996
            if f0 != 0:
                flm0 = np.dot(eout[ref], ein[ref])
            if f1 != 0:
                flm1 = np.dot(np.cross(eout[ref], ein[ref]), z.T)
            if f2 != 0:
                flm2 = np.dot(eout[ref], z.T) * np.dot(ein[ref], z.T)
            fe1e1[ref, :] = flm0 * f0 - 1j * flm1 * f1 + flm2 * f2

        # flm0 = np.array([np.dot(i_eout, i_ein).repeat(len(z)) for i_eout, i_ein in zip(eout, ein)])
        # flm1 = np.array([np.dot(np.cross(i_eout, i_ein), z.T) for i_eout, i_ein in zip(eout, ein)])
        # flm2 = np.array([np.dot(i_eout, z.T) * np.dot(i_ein, z.T) for i_eout, i_ein in zip(eout, ein)])
        # fe1e1 = flm0 * f0 - 1j * flm1 * f1 + flm2 * f2

        # flm0 = np.tile(np.dot(eout, ein.T).diagonal(), (len(z), 1)).T
        # flm1 = np.array([np.dot(np.cross(i_eout, i_ein), z.T) for i_eout, i_ein in zip(eout, ein)])
        # flm2 = np.dot(eout, z.T).diagonal() * np.dot(ein, z.T).diagonal()
        # fe1e1 = flm0 * f0 - 1j * flm1 * f1 + flm2 * f2

        # Calculate structure factor
        sf[:, p] = np.sum(fe1e1 * debyewaller * occ * phase, axis=1)
    if npsi == 1:
        return sf[:, 0]
    return sf


def sf_magnetic_xray_resonant_alternate(q, r, moment, energy_kev, occ=None, debyewaller=None, polarisation='sp',
                                        azi_ref_q=(1, 0, 0), psi=0, f0=0, f1=1, f2=0, **kwargs):
    """
    Calculate structure factors using resonant scattering factors in the dipolar approximation

      I = Scattering.xray_resonant(HKL,energy_kev,polarisation,F0,F1,F2)
    Returns an array with the same length as HKL, giving the real intensity at each reflection.
        energy_kev = x-ray energy in keV
        polarisation = x-ray polarisation: 'ss',{'sp'},'ps','pp'
        f0/1/2 = Resonance factor Flm
        azim_zero = [h,k,l] vector parallel to the origin of the azimuth
        psi = azimuthal angle defining the scattering plane

    Uses the E1E1 resonant x-ray scattering amplitude:
        fxr_n = (ef.ei)*f0 -i(ef X ei).z_n*f1 + (ef.z_n)(ei.z_n)f2

    Where ei and ef are the initial and final polarisation states, respectively,
    and z_n is a unit vector in the direction of the magnetic moment of the nth ion.
    The polarisation states are determined to be one of the natural synchrotron
    states, where sigma (s) is perpendicular to the scattering plane and pi (p) is
    parallel to it.
            ( s-s  s-p )
            ( p-s  p-p )

    From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (15)

    :param q: [n,3] array of hkl reflections
    :param r: [m,3] array of atomic positions in r.l.u.
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param energy_kev: float value of incident x-ray energy in keV
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param azi_ref_q: [1,3] azimuthal refence, in cartesian basis (Q)
    :param psi: [p] array of azimthal angles - the rotation out of the scattering plane.
    :param polarisation: str definition of the polarisation can be: ['ss','sp','ps','pp'] with 's'=sigma, 'p'=pi
    :param f0: float Flm value 0 (charge)
    :param f1: float Flm value 1
    :param f2: float Flm value 2
    :param kwargs: additional options[*unused]
    :return sf: [n, p] complex array of structure factors for different reflections and azimuths
    """

    phase = phase_factor_qr(q, r)
    # z^n is a unit vector in the direction of the magnetic moment of the nth ion.
    moment = fg.norm(moment).reshape((-1, 3))
    if occ is None:
        occ = np.ones(phase.shape[1])
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    if debyewaller is None:
        debyewaller = np.ones(phase.shape)
    psi_array = np.asarray(psi, dtype=float).reshape([-1])
    npsi = len(psi_array)

    _debug('sf_magnetic_xray_resonant_alternate(phase.shape=%s, npsi=%d)' % (phase.shape, npsi))
    sf = np.zeros([len(q), npsi], dtype=complex)
    for p, psival in enumerate(psi_array):
        # Get resonant form factor
        fxres = xray_resonant_scattering_factor(q, moment, energy_kev, polarisation,
                                                (f0, f1, f2), psival, azi_ref_q)
        # Calculate structure factor
        # Broadcasting used on 2D fxres
        sf[:, p] = np.sum(fxres * debyewaller * occ * phase, axis=1)
    if npsi == 1:
        return sf[:, 0]
    return sf


########################################################################################################################
# -----------------------------------------  MAGNETIC PROJECTIONS  --------------------------------------------------- #
########################################################################################################################


def xray_resonant_scattering_factor(q, moment, energy_kev, polarisation='sp', flm=(1, 1, 1), psi=0,
                                    azi_ref_q=(1, 0, 0)):
    """
    Calcualte fxres, the resonant x-ray scattering factor
      fxres = Scattering.xray_resonant_scattering_factor(HKL,energy_kev,polarisation,flm,azim_zero,psi)
    energy_kev = x-ray energy in keV
        polarisation = x-ray polarisation: 'ss',{'sp'},'ps','pp'
        flm = (f0, f1, f2) Resonance factor Flm, f0/1/2 should be 0 or 1 each
        azim_zero = [h,k,l] vector parallel to the origin of the azimuth {[1,0,0]}
        psi = azimuthal angle defining the scattering plane {0}

    :param q: [n*3] array of reflection coordinates in cartesian basis (Q)
    :param moment: [mx3] array of magnetic moments in cartesian basis
    :param energy_kev: float energy in keV
    :param polarisation: polarisation condition: 'sp', 'ss', 'ps', 'pp'. s=sigma, p=pi
    :param flm: (f0, f1, f2) Resonance factor Flm, f0/1/2 should be 0 or 1 each
    :param azi_ref_q: azimuthal refence, in cartesian basis (Q)
    :param psi: float, azimuthal angle
    :return: fxres [n*1] array of resonant x-ray scattering factors

    Uses the E1E1 resonant x-ray scattering amplitude:
        fxr_n = (ef.ei)*F0 -i(ef X ei).z_n*F1 + (ef.z_n)(ei.z_n)F2

    Where ei and ef are the initial and final polarisation states, respectively,
    and z_n is a unit vector in the direction of the magnetic moment of the nth ion.
    The polarisation states are determined to be one of the natural synchrotron
    states, where sigma (s) is perpendicular to the scattering plane and pi (p) is
    parallel to it.
            ( s-s  s-p )
            ( p-s  p-p )

    From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (15)
    """

    q = np.asarray(q, dtype=float).reshape((-1, 3))
    moment = np.asarray(moment, dtype=float).reshape((-1, 3))
    polarisation = polarisation.lower().replace('-', '').replace(' ', '')
    nref = len(q)
    nat = len(moment)

    qmag = fg.mag(q)
    bragg = fc.cal2theta(qmag, energy_kev) / 2

    fxres = np.zeros([nref, nat], dtype=complex)
    for ref in range(nref):
        # Resonant scattering factor
        # Electric Dipole transition at 3d L edge
        z1, z2, z3 = scatteringcomponents(moment, q[ref], azi_ref_q, psi).T
        bragg_r = np.deg2rad(bragg[ref])

        if polarisation in ['sigmasigma', 'sigsig', 'ss']:  # Sigma-Sigma
            f0 = 1 * np.ones(nat)
            f1 = 0 * np.ones(nat)
            f2 = z2 ** 2
        elif polarisation in ['sigmapi', 'sigpi', 'sp']:  # Sigma-Pi
            f0 = 0 * np.ones(nat)
            f1 = z1 * np.cos(bragg_r) + z3 * np.sin(bragg_r)
            f2 = -z2 * (z1 * np.sin(bragg_r) - z3 * np.cos(bragg_r))
        elif polarisation in ['pisigma', 'pisig', 'ps']:  # Pi-Sigma
            f0 = 0 * np.ones(nat)
            f1 = z3 * np.sin(bragg_r) - z1 * np.cos(bragg_r)
            f2 = z2 * (z1 * np.sin(bragg_r) + z3 * np.cos(bragg_r))
        elif polarisation in ['pipi', 'pp']:  # Pi-Pi
            f0 = np.cos(2 * bragg_r) * np.ones(nat)
            f1 = -z2 * np.sin(2 * bragg_r)
            f2 = -(np.cos(bragg_r) ** 2) * (z1 ** 2 * np.tan(bragg_r) ** 2 + z3 ** 2)
        else:
            raise ValueError('Incorrect polarisation. pol should be e.g. ''ss'' or ''sp''')
        fxres[ref, :] = flm[0] * f0 - 1j * flm[1] * f1 + flm[2] * f2
    return fxres


def scatteringbasis(q, azi_ref_q=(1, 0, 0), psi=0):
    """
    Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
    :param q: [1*3] reflection vector in a cartesian basis
    :param azi_ref_q: [1,3] direction along which the azimuthal zero angle is determind
    :param psi: float azimuthal angle about U3 in degrees
    :return: U1, U2, U3
    The basis is chosen such that Q defines the scattering plane, the sigma direction is normal to this plane,
    the pi direction is always within this plane.
    The azimuthal angle defines a rotation about the Q axis in a clockwise mannor, matching I16.
    At an azimuth of 0degrees, U1 is perpendicular to Q, along the direction of azim_zero.
    """

    # Define coordinate system I,J,Q (U1,U2,U3)
    # See FDMNES User's Guide p20 'II-11) Anomalous or resonant diffraction'
    # U1 || projection of azim_zero
    # U2 _|_ U1,U3
    # U3 || Q = kf-ki
    Qhat = fg.norm(q)  # || Q
    AxQ = fg.norm(np.cross(azi_ref_q, Qhat))
    Ihat = fg.norm(np.cross(Qhat, AxQ))  # || to azim_zero
    Jhat = fg.norm(np.cross(Qhat, Ihat))  # -| to I and Q

    # Rotate psi about Qhat
    rpsi = np.deg2rad(psi)
    # -ve sin makes clockwise rotation
    # This was checked on 21/1/19 vs CRO paper + sergio's calculations and seems to agree with experiment,
    # however we never did an azimuthal scan of the (103) which would have distinguished this completely.
    Ihat_psi = fg.norm(np.cos(rpsi) * Ihat - np.sin(rpsi) * Jhat)
    Jhat_psi = fg.norm(np.cross(Qhat, Ihat_psi))
    return np.vstack([Ihat_psi, Jhat_psi, Qhat])


def scatteringcomponents(moment, q, azi_ref_q=(1, 0, 0), psi=0):
    """
    Transform magnetic vector into components within the scattering plane
    :param moment: [n*3] array of magnetic moments in a cartesian basis
    :param q: [1*3] reflection vector in a cartesian basis
    :param azi_ref_q: [1*3] azimuthal reference in a cartesian basis
    :param psi: float azimuthal angle
    :return: (z1, z2, z3) components of the magnetic moment along the reflection direction
    """
    U = scatteringbasis(q, azi_ref_q, psi)
    # Determine components of the magnetic vector
    z1z2z3 = np.dot(moment, U.T)  # [mxmymz.I, mxmymz.J, mxmymz.Q]
    return fg.norm(z1z2z3)


def scatteringvectors(q, energy_kev, azi_ref_q=(1, 0, 0), psi=0, polarisation='s-p'):
    """
    Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
    :param q: [n,3] reflection vector in a cartesian basis
    :param energy_kev: x-ray scattering energy in keV
    :param azi_ref_q: [1,3] direction along which the azimuthal zero angle is determind
    :param psi: float angle in degrees about the azimuth
    :param polarisation: polarisation with respect to the scattering plane, options:
                'ss' : sigma-sigma polarisation
                'sp' : sigma-pi polarisation
                'ps' : pi-sigma polarisation
                'pp' : pi-pi polarisation
    :return: kin, kout, ein, eout
    Returned values are [n,3] arrays
        kin : [n,3] array of incident wavevectors
        kout: [n,3] array of scattered wavevectors
        ein : [n,3] array of incident polarisation
        eout: [n,3] array of scattered polarisation

    The basis is chosen such that Q defines the scattering plane, sigma and pi directions are normal to this plane.
    """

    q = np.asarray(q, dtype=float).reshape([-1, 3])
    azi_ref_q = np.asarray(azi_ref_q, dtype=float).reshape(3)
    polarisation = polarisation.replace('-', '').replace(' ', '')

    out_kin = np.zeros([len(q), 3])
    out_kout = np.zeros([len(q), 3])
    out_ein = np.zeros([len(q), 3])
    out_eout = np.zeros([len(q), 3])
    for n in range(len(q)):
        # Define coordinate system I,J,Q (U1,U2,U3)
        # See FDMNES User's Guide p20 'II-11) Anomalous or resonant diffraction'
        Qhat = fg.norm(q[n, :])  # || Q
        AxQ = fg.norm(np.cross(azi_ref_q, Qhat))
        Ihat = fg.norm(np.cross(Qhat, AxQ))  # || to azim_zero
        Jhat = fg.norm(np.cross(Qhat, Ihat))  # -| to I and Q

        # Determine wavevectors
        bragg = fc.cal2theta(fg.mag(q[n, :]), energy_kev) / 2
        if np.isnan(bragg):
            raise Exception('Bragg > 180deg at this energy: q(%s) @ E=%s' % (q[n, :], energy_kev))
        rb = np.deg2rad(bragg)
        rp = np.deg2rad(psi)
        kin = np.cos(rb) * np.cos(rp) * Ihat - np.cos(rb) * np.sin(rp) * Jhat - np.sin(rb) * Qhat
        kout = np.cos(rb) * np.cos(rp) * Ihat - np.cos(rb) * np.sin(rp) * Jhat + np.sin(rb) * Qhat
        esig = np.sin(rp) * Ihat + np.cos(rp) * Jhat  # sigma polarisation (in or out)
        piin = np.cross(kin, esig)  # pi polarisation in
        piout = np.cross(kout, esig)  # pi polarisation out

        # Polarisations
        if polarisation in ['sigmasigma', 'sigsig', 'ss']:
            ein = 1.0 * esig
            eout = 1.0 * esig
        elif polarisation in ['sigmapi', 'sigpi', 'sp']:
            ein = 1.0 * esig
            eout = 1.0 * piout
        elif polarisation in ['pisigma', 'pisig', 'ps']:
            ein = 1.0 * piin
            eout = 1.0 * esig
        elif polarisation in ['pipi', 'pp']:
            ein = 1.0 * piin
            eout = 1.0 * piout
        else:
            raise ValueError('Incorrect polarisation. pol should be e.g. ''ss'' or ''sp''')
        out_kin[n, :] = kin
        out_kout[n, :] = kout
        out_ein[n, :] = ein
        out_eout[n, :] = eout
    return out_kin, out_kout, out_ein, out_eout


########################################################################################################################
# ----------------------------------------  ScatteringTypes  --------------------------------------------------------- #
########################################################################################################################


def get_scattering_type(scattering_type):
    """Return correct label for scattering type"""
    scattering_type = scattering_type.lower()
    for name, alt_names in SCATTERING_TYPES.items():
        if scattering_type in alt_names:
            return name
    raise Exception(f"Scattering type {scattering_type} not recognized")


def get_scattering_function(scattering_type):
    """
    Return function for given scattering type
    Function will return structure factor function
    :param scattering_type: str : scattering name as defined in SCATTERING_NAMES
    :return: function
    """
    scattering_type = scattering_type.lower()
    if scattering_type in SCATTERING_TYPES['xray']:
        return sf_atom
    if scattering_type in SCATTERING_TYPES['xray fast']:
        return sf_atom
    if scattering_type in SCATTERING_TYPES['xray dispersion']:
        return sf_xray_dispersion
    if scattering_type in SCATTERING_TYPES['neutron']:
        return sf_atom
    if scattering_type in SCATTERING_TYPES['electron']:
        return sf_atom
    if scattering_type in SCATTERING_TYPES['xray magnetic']:
        return sf_magnetic_xray
    if scattering_type in SCATTERING_TYPES['neutron magnetic']:
        return sf_magnetic_neutron
    if scattering_type in SCATTERING_TYPES['neutron polarised']:
        return sf_magnetic_neutron_polarised
    if scattering_type in SCATTERING_TYPES['xray polarised']:
        return sf_magnetic_xray_polarised
    if scattering_type in SCATTERING_TYPES['xray resonant']:
        return sf_magnetic_xray_resonant
    if scattering_type in SCATTERING_TYPES['custom']:
        return sf_custom  # sf_xray_dispersion is the most general as it allows energy
    raise Exception('Scattering name %s not recognised' % scattering_type)


def options(occ=None, debyewaller=None, scattering_factor=None,
            moment=None, incident_polarisation_vector=(1, 0, 0), magnetic_formfactor=None,
            energy_kev=8, polarisation='sp', azi_ref_q=(1, 0, 0), psi=0, f0=0, f1=1, f2=0):
    """
    Create an input dict that will work with all structure factor (sf_) functions
    :param occ: [m,1] array of atomic occupancies
    :param debyewaller: [n,m] array of thermal factors for each atom and reflection
    :param scattering_factor: array [n,m] or [n]: radiation dependent scattering factor/ form factor,/ scattering length
    :param moment: [m,3] array of magnetic moment direction in orthogonal basis
    :param incident_polarisation_vector: [1,3] direction of incident polarisation
    :param magnetic_formfactor: [n,m] array of magnetic form factors for each atom and relection
    :param energy_kev: float value of incident x-ray energy in keV
    :param azi_ref_q: [1,3] azimuthal refence, in cartesian basis (Q)
    :param psi: float value of the azimthal angle - the rotation out of the scattering plane.
    :param polarisation: str definition of the polarisation can be: ['ss','sp','ps','pp'] with 's'=sigma, 'p'=pi
    :param f0: float Flm value 0 (charge)
    :param f1: float Flm value 1
    :param f2: float Flm value 2
    :return: dict
    """
    return locals()


def scattering_factors(scattering_type, atom_type, qmag, enval,
                       use_sears=False, use_wasskirf=False):
    """
    Return an array of scattering factors based on the radiation
    :param scattering_type: str radiation, see "get_scattering_function()"
    :param atom_type: [nx1] str array of element symbols
    :param qmag: [mx1] or None, float array of wavevector magnitudes for reflections
    :param enval: [ox1] or None, float array of energies in keV
    :param use_sears: if True, use neutron scattering lengths from ITC Vol. C, By V. F. Sears
    :param use_wasskirf: if True, use x-ray scattering factors from Waasmaier and Kirfel
    :return: [nxmxo] array of scattering factors
    """
    if scattering_type in SCATTERING_TYPES['neutron']:
        if use_sears:
            return fc.neutron_scattering_length(atom_type, 'Sears')
        else:
            return fc.neutron_scattering_length(atom_type)
    elif scattering_type in SCATTERING_TYPES['electron']:
        return fc.electron_scattering_factor(atom_type, qmag)
    elif scattering_type in SCATTERING_TYPES['xray fast']:
        return fc.atom_properties(atom_type, 'Z')
    elif scattering_type in SCATTERING_TYPES['xray dispersion']:
        return fc.xray_scattering_factor_resonant(atom_type, qmag, enval, use_waaskirf=use_wasskirf)
    elif scattering_type in SCATTERING_TYPES['custom']:
        return fc.custom_scattering_factor(atom_type, qmag, enval)
    elif use_wasskirf:
        return fc.xray_scattering_factor_WaasKirf(atom_type, qmag)
    else:
        return fc.xray_scattering_factor(atom_type, qmag)


def autostructurefactor(scattering_type, q, r, *args, **kwargs):
    """
    Choose a scattering type can calcuate the structure factor
    :param scattering_type: str radiation, see "get_scattering_function()"
    :param q: array [n,3] reflection positions in A^-1
    :param r: array [m,3] atomic positions in A
    :param args: additional arguments to pass to choosen scattering function
    :param kwargs: named arguments to pass to choosen scattering function
    :return: complex array [n]
    """
    scatter_fun = get_scattering_function(scattering_type)
    opt = options(*args, **kwargs)

    q = np.asarray(q, dtype=float).reshape([-1, 3])
    r = np.asarray(r, dtype=float).reshape([-1, 3])
    energy_kev = np.asarray(opt['energy_kev'], dtype=float).reshape(-1)
    psi = np.asarray(opt['psi'], dtype=float).reshape(-1)

    nref = q.shape[0]
    natom = r.shape[0]
    nenergy = energy_kev.size
    npsi = psi.size

    scattering_factor = opt['scattering_factor']
    scattering_factor = np.asarray(scattering_factor) if scattering_factor is not None else np.ones((nref, len(r)))
    if scattering_factor.ndim < 2 or scattering_factor.shape[1] < 2:
        scattering_factor = np.tile(scattering_factor.reshape((-1, len(r))), (nref, 1))
    scattering_factor = scattering_factor.reshape((nref, len(r), -1))

    debyewaller = opt['debyewaller']
    debyewaller = np.asarray(debyewaller) if debyewaller is not None else np.ones((nref, len(r)))
    if debyewaller.ndim < 2 or debyewaller.shape[1] < 2:
        debyewaller = np.tile(debyewaller.reshape((-1, len(r))), (nref, 1))

    magff = opt['magnetic_formfactor']
    magff = np.asarray(magff) if magff is not None else np.ones((nref, len(r)))
    if magff.ndim < 2 or magff.shape[1] < 2:
        magff = np.tile(magff.reshape((-1, len(r))), (nref, 1))

    # Break up long lists of HKLs
    n_arrays = np.ceil(nref * natom / MAX_QR_ARRAY)
    if n_arrays > 1:
        print('Splitting %d reflections (%d atoms) into %1.0f parts' % (nref, natom, n_arrays))
    q_array = np.array_split(q, n_arrays)
    scattering_factor = np.array_split(scattering_factor, n_arrays)
    debyewaller = np.array_split(debyewaller, n_arrays)
    magff = np.array_split(magff, n_arrays)

    sf = np.zeros([nref, nenergy, npsi], dtype=complex)
    start_time = datetime.datetime.now()
    for e, enval in enumerate(energy_kev):
        opt['energy_kev'] = enval
        for p, psival in enumerate(psi):
            opt['psi'] = psival
            ls = 0
            for n, _q in enumerate(q_array):
                if n_arrays > 1:
                    print(' Starting %2.0f/%2.0f: %d:%d' % (n+1, n_arrays, ls, ls+len(_q)))
                opt['scattering_factor'] = scattering_factor[n][:, :, e]
                opt['debyewaller'] = debyewaller[n]
                opt['magnetic_formfactor'] = magff[n]
                sf[ls: ls+len(_q), e, p] = scatter_fun(_q, r, **opt)
                ls = ls+len(_q)
    end_time = datetime.datetime.now()
    time_difference = end_time - start_time
    if TIME_REPORT and time_difference.total_seconds() > 10:
        print('Calculated %d structure factors in %s' % (nref, time_difference))
    if nenergy == 1 and npsi == 1:
        return sf[:, 0, 0]  # shape(nref)
    if nenergy == 1:
        return sf[:, 0, :]  # shape(nref, nenergy)
    if npsi == 1:
        return sf[:, :, 0]  # shape(nref, nspi)
    return sf


def autointensity(scattering_type, q, r, *args, **kwargs):
    """
    Choose a scattering type can calcuate the scattered intensity
    :param scattering_type: named scattering function, see "get_scattering_function()"
    :param q: array [n,3] reflection positions in A^-1
    :param r: array [m,3] atomic positions in A
    :param args: additional arguments to pass to choosen scattering function
    :param kwargs: named arguments to pass to choosen scattering function
    :return: float array [n]
    """
    sf = autostructurefactor(scattering_type, q, r, *args, **kwargs)
    return intensity(sf)





