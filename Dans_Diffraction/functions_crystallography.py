# -*- coding: utf-8 -*-
"""
Module: functions_crystallography.py

By Dan Porter, PhD
Diamond
2018

Usage: 
    - Run this file in an interactive console
    OR
    - from Dans_Diffraction import functions_crystallography as fc

Version 4.1.0

Version History:
09/07/15 0.1    Version History started.
30/10/17 1.0    Many updates
06/01/18 2.0    Renamed functions_crystallography.py
02/03/18 2.1    Removed call to tkinter
05/04/18 2.2    Added invert_sym
02/05/18 2.3    Added comparison of lower case elements and symmetry values
04/06/18 2.4    Added new checks to readcif, corrected atomic_properties for python3/numpy1.14
08/06/18 2.5    Corrected a few errors and added some more comments
06/03/19 2.6    Added print_atom_properties
14/08/19 2.7    Added new Dans Element Properties file with extra comment line, new functions added
03/04/20 2.8    Updated attenuation to work with arrays of elements
19/04/20 2.9    Added write_cif, made alterations to readcif for speed and readability, added spacegroup
01/05/20 3.0    Updated atom_properties, now have atomic properties above 92 with warning. Some changes to readcif.
05/05/20 3.0.1  Further changes to readcif. Changed method of symmetry_ops2magnetic. Added str2element
12/05/20 3.1.0  More readcif changes, added atomic_scattering_factor, element_charge_string, split_compound
26/05/20 3.1.1  Updated magnetic space groups, added magnetic positions (was only generators), added write_mcif
09/06/20 3.2    Updated gen_sym_mat, symmetry_ops2magnetic, added sym_op_det
03/09/20 3.2.1  Updated cif_symmetry to allow for missing magnetic centring
21/01/21 3.3    Added xray_scattering_factor_resonant and xray_dispersion_corrections functions added
26/01/21 3.4    Added xray attenuation, transmission and refractive index
31/03/21 3.5    Added point groups, gen_sym_unique
10/06/21 3.6    Corrected mistake in DebyeWaller function. Added x-ray scattering factors from Waasmaier and Kirfel
15/11/21 3.7    Added diffractometer orientation commands from Busing & Levy, H. You
12/01/22 3.7.1  Added gen_sym_axial_vector
23/04/22 3.7.2  Corrected magnitude of Q in magnetic_structure_factors
07/05/23 3.8.0  Added electron_scattering_factors and electron wavelength formula
22/05/23 3.8.1  Added wyckoff_label, find_spacegroup
02/07/23 3.8.2  Added wavelength options to several functions, plut DeBroglie wavelength function
26/09/24 3.9.0  Added complex neutron scattering lengths for isotopes from package periodictable
06/11/24 4.0.0  Fixed error with triclinic bases, added function_lattice.
20/11/24 4.0.1  Added alternative neutron scattering length table
28/06/25 4.1.0  Added new magnetic form factor calculation, plus new formulas

Acknoledgements:
    April 2020  Thanks to ChunHai Wang for helpful suggestions in readcif!
    May 2023    Thanks to Carmelo Prestipino for adding electron scattering factors
    Sep 2024    Thanks to thamnos for suggestion to add complex neutron scattering lengths
    Oct 2024    Thanks to Lee Richter for pointing out the error in triclinic basis definition

@author: DGPorter
"""

import sys, os, re
import json
import numpy as np
from warnings import warn

from . import functions_general as fg
from . import functions_lattice as fl

__version__ = '5.0.0'

# File directory - location of "Dans Element Properties.txt"
datadir = os.path.abspath(os.path.dirname(__file__))  # same directory as this file
datadir = os.path.join(datadir, 'data')
ATOMFILE = os.path.join(datadir, 'Dans Element Properties.txt')
PENGFILE = os.path.join(datadir, 'peng.dat')
WAASKIRF_FILE = os.path.join(datadir, 'f0_WaasKirf.dat')
NSLFILE = os.path.join(datadir, 'neutron_isotope_scattering_lengths.dat')
NSLFILE_SEARS = os.path.join(datadir, 'neutron_isotope_scattering_lengths_sears.dat')
ASFFILE = os.path.join(datadir, 'atomic_scattering_factors.npy')
XMAFILE = os.path.join(datadir, 'XRayMassAtten_mup.dat')
MAGFF_FILE = os.path.join(datadir, 'McPhase_Mag_FormFactors.txt')

# List of Elements in order sorted by length of name
ELEMENT_LIST = [
    'Zr', 'Mo', 'Es', 'Eu', 'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge',
    'He', 'Hf', 'Hg', 'Ho', 'Hs', 'In', 'Ir', 'Kr', 'La', 'Li', 'Lr',
    'Lu', 'Lv', 'Mc', 'Zn', 'Mg', 'Er', 'Dy', 'Ds', 'Bk', 'Ag', 'Al',
    'Am', 'Ar', 'As', 'At', 'Au', 'Ba', 'Be', 'Bh', 'Bi', 'Br', 'Db',
    'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs', 'Cu',
    'Mn', 'Md', 'Mt', 'Rb', 'Rf', 'Rh', 'Rn', 'Ru', 'Sb', 'Sc', 'Se',
    'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti',
    'Tl', 'Tm', 'Ts', 'Xe', 'Yb', 'Re', 'Rg', 'Ac', 'Pb', 'Nh', 'Ni',
    'No', 'Np', 'Nd', 'Ra', 'Og', 'Os', 'Pa', 'Pd', 'Nb', 'Pm', 'Po',
    'Pr', 'Pt', 'Na', 'Pu', 'Ne', 'B', 'W', 'V', 'Y', 'U', 'F', 'K',
    'C', 'I', 'P', 'H', 'S', 'N', 'O',
    'zr', 'mo', 'es', 'eu', 'fe', 'fl', 'fm', 'fr', 'ga', 'gd', 'ge',
    'he', 'hf', 'hg', 'ho', 'hs', 'in', 'ir', 'kr', 'la', 'li', 'lr',
    'lu', 'lv', 'mc', 'zn', 'mg', 'er', 'dy', 'ds', 'bk', 'ag', 'al',
    'am', 'ar', 'as', 'at', 'au', 'ba', 'be', 'bh', 'bi', 'br', 'db',
    'ca', 'cd', 'ce', 'cf', 'cl', 'cm', 'cn', 'co', 'cr', 'cs', 'cu',
    'mn', 'md', 'mt', 'rb', 'rf', 'rh', 'rn', 'ru', 'sb', 'sc', 'se',
    'sg', 'si', 'sm', 'sn', 'sr', 'ta', 'tb', 'tc', 'te', 'th', 'ti',
    'tl', 'tm', 'ts', 'xe', 'yb', 're', 'rg', 'ac', 'pb', 'nh', 'ni',
    'no', 'np', 'nd', 'ra', 'og', 'os', 'pa', 'pd', 'nb', 'pm', 'po',
    'pr', 'pt', 'na', 'pu', 'ne', 'b', 'w', 'v', 'y', 'u', 'f', 'k',
    'c', 'i', 'p', 'h', 's', 'n', 'o',
    'D', 'd',  # add Deuterium
]
element_regex = re.compile('|'.join(ELEMENT_LIST))
regex_sub_element = re.compile('[^a-zA-Z]')  # 'Li' = regex_sub_element.sub('', '7-Li2+')

# Custom atomic form factors
CUSTOM_FORM_FACTOR_COEFS = {}  # {'el': (a1, b1, a2, b2, ...)}
CUSTOM_DISPERSION_TABLE = {}  # {'el': np.array(en, f1, f2)}

# Required CIF keys, must be available and not '?'
CIF_REQUIRES = [
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
    "_atom_site_label",
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z",
]


def getenergy():
    return 8.048  # Cu Kalpha energy, keV


'--------------Functions to Read & Write CIF files----------------------'


def readcif(filename=None, debug=False):
    """
    Open a Crystallographic Information File (*.cif) file and store all entries in a key:value dictionary
     Looped values are stored as lists under a single key entry
     All values are stored as strings
    E.G.
      crys=readcif('somefile.cif')
      crys['_cell_length_a'] = '2.835(2)'

    crys[key] = value
    available keys are give by crys.keys()

    To debug the file with outputted messages, use:
      cif = readcif(file, debug=True)

    Some useful standard CIF keywords:
        _cell_length_a
        _cell_length_b
        _cell_length_c
        _cell_angle_alpha
        _cell_angle_beta
        _cell_angle_gamma
        _space_group_symop_operation_xyz
        _atom_site_label
        _atom_site_type_symbol
        _atom_site_occupancy
        _atom_site_U_iso_or_equiv
        _atom_site_fract_x
        _atom_site_fract_y
        _atom_site_fract_z
    """

    # Get file name
    filename = os.path.abspath(os.path.expanduser(filename))
    (dirName, filetitle) = os.path.split(filename)
    (fname, Ext) = os.path.splitext(filetitle)

    # Open file
    file = open(filename)
    text = file.read()
    file.close()

    # Remove blank lines
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    lines = text.splitlines()

    cifvals = {'Filename': filename, 'Directory': dirName, 'FileTitle': fname}

    # Read file line by line, converting the cif file values to a python dict
    n = 0
    while n < len(lines):
        # Convert line to columns
        vals = lines[n].strip().split()

        # skip empty lines
        if len(vals) == 0:
            n += 1
            continue

        # Search for stored value lines
        if vals[0][0] == '_':
            if len(vals) == 1:
                # Record next lines that are not keys as string
                if lines[n + 1][0] == ';': n += 1
                strarg = []
                while n + 1 < len(lines) and (len(lines[n + 1]) == 0 or lines[n + 1][0].strip() not in ['_', ';']):
                    strarg += [lines[n + 1].strip('\'"')]
                    n += 1
                cifvals[vals[0]] = '\n'.join(strarg)
                chk = 'a'
            else:
                cifvals[vals[0]] = ' '.join(vals[1:]).strip(' \'"\n')
                chk = 'b'
            n += 1
            if debug:
                print('%5d %s %s = %s' % (n, chk, vals[0], cifvals[vals[0]]))
            continue

        # Search for loops
        elif vals[0] == 'loop_':
            n += 1
            loopvals = []
            # Step 1: Assign loop columns
            # looped columns are given by "_column_name"
            while n < len(lines) and len(lines[n].strip()) > 0 and lines[n].strip()[0] == '_':
                loopvals += [lines[n].split()[0]]
                cifvals[loopvals[-1]] = []
                n += 1

            # Step 2: Assign data to columns
            # loops until line has less segments than columns
            while n < len(lines):
                # cols = lines[n].split()
                # this fixes error on symmetry arguments having spaces
                # this will only work if the last argument in the loop is split by spaces (in quotes)
                # cols = cols[:len(loopvals) - 1] + [''.join(cols[len(loopvals) - 1:])]
                cols = [col for col in re.split("( |\\\".*?\\\"|'.*?')", lines[n]) if col.strip()]
                if len(cols) != len(loopvals): break
                if cols[0][0] == '_' or cols[0] == 'loop_': break  # catches error if loop is only 1 iteration
                if cols[0][0] == '#': n += 1; continue  # catches comented out lines
                if len(loopvals) == 1:
                    cifvals[loopvals[0]] += [lines[n].strip(' \"\'\n')]
                else:
                    for c, ll in enumerate(loopvals):
                        cifvals[ll] += [cols[c]]
                n += 1

            if debug:
                for ll in loopvals:
                    print('%5d L %s = %s' % (n, ll, str(cifvals[ll])))
            continue

        else:
            # Skip anything else
            if debug:
                print('%5d SKIPPED: %s' % (n, lines[n]))
            n += 1

    # Replace '.' in keys - fix bug from isodistort cif files
    # e.g. '_space_group_symop_magn_operation.xyz'
    current_keys = list(cifvals.keys())
    for key in current_keys:
        if '.' in key:
            newkey = key.replace('.', '_')
            cifvals[newkey] = cifvals[key]
    return cifvals


def cif_check(cifvals, required_keys=None, bad_value='?'):
    """
    Returns True if all basic required cif parameter are available and real.
    E.G.:
        cifvals = readcif(file.cif)
        if cif_check(cifvals):
            print('File OK')
    :param cifvals: dict of cif keys form readcif
    :param required_keys: list of key strings, or None for default
    :param bad_value: if this value is in the cif key item, return False
    :return: bool
    """
    if required_keys is None:
        required_keys = CIF_REQUIRES
    keys = cifvals.keys()
    for key in required_keys:
        if key not in keys:
            return False
        if bad_value in cifvals[key]:
            return False
    return True


def cif_symmetry(cifvals):
    """
    Read symmetries from a cif dict
    :param cifvals:
    :return: list(symmetry_operations), list(magnetic_operations), list(time operations)
    """
    keys = cifvals.keys()

    if '_symmetry_equiv_pos_as_xyz' in keys:
        symops = cifvals['_symmetry_equiv_pos_as_xyz']
        symmetry_operations = gen_symcen_ops(symops, ['x,y,z'])
        symmetry_operations_magnetic = symmetry_ops2magnetic(symops)
        symmetry_operations_time = sym_op_time(symops)
    elif '_space_group_symop_operation_xyz' in keys:
        symops = cifvals['_space_group_symop_operation_xyz']
        symmetry_operations = gen_symcen_ops(symops, ['x,y,z'])
        symmetry_operations_magnetic = symmetry_ops2magnetic(symops)
        symmetry_operations_time = sym_op_time(symops)
    elif '_space_group_symop_magn_operation_xyz' in keys:
        symops = cifvals['_space_group_symop_magn_operation_xyz']
        if '_space_group_symop_magn_centering_xyz' in keys:
            symcen = cifvals['_space_group_symop_magn_centering_xyz']
        else:
            symcen = ['x,y,z']
        symmetry_operations = gen_symcen_ops(symops, symcen)
        symmetry_operations_time = sym_op_time(symmetry_operations)

        if '_space_group_symop_magn_operation_mxmymz' in keys:
            magops = cifvals['_space_group_symop_magn_operation_mxmymz']  # mx,my,mz
            magcen = cifvals['_space_group_symop_magn_centering_mxmymz']
            magops = [op.replace('m', '') for op in magops]
            magcen = [op.replace('m', '') for op in magcen]
            symmetry_operations_magnetic = gen_symcen_ops(magops, magcen)
        else:
            symmetry_operations_magnetic = symmetry_ops2magnetic(symmetry_operations)
    else:
        symmetry_operations = ['x,y,z']
        symmetry_operations_magnetic = ['x,y,z']
        symmetry_operations_time = [1]

    symmetry_operations_magnetic = sym_op_mx(symmetry_operations_magnetic)
    return symmetry_operations, symmetry_operations_magnetic, symmetry_operations_time


def cif2dict(cifvals):
    """
    From a dict of key:value pairs generated from a *.cif file (using readcif),
    read standard crystallographic infromation into a crystal dictionary, with keys:
        'filename' - cif filename
        'name' - name of structure
        'unit vector'   - [3x3] array of basis vectors [a,b,c]
        'parent cell'   - [3x3] array of parent basis vectors [a,b,c]
        'symmetry'      - list of string symmetry operations
        'space group'   - Space Group string
        'space group number' - Space group number
        'cif'           - cif dict
        The following are specific to atomis within the unit cell:
        'atom type'     - list of elements e.g. ['Co','O', 'O']
        'atom label'    - list of atom site names e.g. ['Co1, 'O1', 'O2']
        'atom position' - [nx3] array of atomic positions
        'atom occupancy'- [nx1] array of atom site occupancies
        'atom uiso'     - [nx1] array of atom site isotropic thermal parameters
        'atom uaniso'   - [nx6] array of atom site anisotropic thermal parameters
        'mag moment'    - [nx3] array of atom site magnetic moment vectors
        'mag time'      - [nx3] array of atom site magnetic time symmetry
        'misc'] = [element, label, cif_pos, occ, Uiso]
    :param cifvals: dict from readcif
    :return: dict
    """

    keys = cifvals.keys()

    # Generate unit vectors
    a, da = fg.readstfm(cifvals['_cell_length_a'])
    b, db = fg.readstfm(cifvals['_cell_length_b'])
    c, dc = fg.readstfm(cifvals['_cell_length_c'])
    alpha, dalpha = fg.readstfm(cifvals['_cell_angle_alpha'])
    beta, dbeta = fg.readstfm(cifvals['_cell_angle_beta'])
    gamma, dgamma = fg.readstfm(cifvals['_cell_angle_gamma'])
    UV = fl.basis_3(a, b, c, alpha, beta, gamma)

    # Get atom names & labels
    label = cifvals['_atom_site_label']

    if '_atom_site_type_symbol' in keys:
        element = [x.strip('+-0123456789') for x in cifvals['_atom_site_type_symbol']]
    else:
        element = [x.strip('+-0123456789') for x in cifvals['_atom_site_label']]

    # Get other properties
    if '_atom_site_U_iso_or_equiv' in keys:
        Uiso = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_U_iso_or_equiv']])
    elif '_atom_site_B_iso_or_equiv' in keys:
        biso = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_B_iso_or_equiv']])
        Uiso = biso2uiso(biso)
    else:
        Uiso = np.zeros(len(element))
    if '_atom_site_occupancy' in keys:
        occ = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_occupancy']])
    else:
        occ = np.ones(len(element))

    # Get coordinates
    u = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_x']])
    v = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_y']])
    w = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_z']])
    cif_pos = np.array([u, v, w]).T

    # Get magnetic vectors
    cif_vec = np.zeros(cif_pos.shape)
    cif_mag = np.zeros(len(u))
    if '_atom_site_moment_label' in keys:
        mag_atoms = cifvals['_atom_site_moment_label']
        mx = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_x']])
        my = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_y']])
        mz = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_z']])
        mag_pos = np.array([mx, my, mz]).T
        for n, ma in enumerate(mag_atoms):
            mag_idx = label.index(ma)
            cif_vec[mag_idx, :] = mag_pos[n, :]
            cif_mag[mag_idx] = fg.mag(mag_pos[n, :])

    # Get symmetry operations
    if '_symmetry_equiv_pos_as_xyz' in keys:
        symops = cifvals['_symmetry_equiv_pos_as_xyz']
        symtim = [1] * len(symops)
        symmag = ['mx,my,mz'] * len(symops)
        symcen = ['x,y,z']
        symcentim = [1]
        symcenmag = ['mx,my,mz']
    elif '_space_group_symop_operation_xyz' in keys:
        symops = cifvals['_space_group_symop_operation_xyz']
        symtim = [1] * len(symops)
        symmag = ['mx,my,mz'] * len(symops)
        symcen = ['x,y,z']
        symcentim = [1]
        symcenmag = ['mx,my,mz']
    elif '_space_group_symop.magn_operation_xyz' in keys:
        symops_tim = cifvals['_space_group_symop.magn_operation_xyz']
        # Each symop given with time value: x,y,z,+1, separate them:
        symops = [','.join(s.split(',')[:3]) for s in symops_tim]  # x,y,z
        symtim = [int(s.split(',')[-1]) for s in symops_tim]  # +1
        symmag = cifvals['_space_group_symop.magn_operation_mxmymz']

        # Centring vectors also given in this case
        symcen_tim = cifvals['_space_group_symop.magn_centering_xyz']
        symcen = [','.join(s.split(',')[:3]) for s in symcen_tim]  # x,y,z
        symcentim = [int(s.split(',')[-1]) for s in symcen_tim]  # +1
        symcenmag = cifvals['_space_group_symop.magn_centering_mxmymz']
    else:
        symops = ['x,y,z']
        symtim = [1]
        symmag = ['mx,my,mz']
        symcen = ['x,y,z']
        symcentim = [1]
        symcenmag = ['mx,my,mz']
    symops = [re.sub('\'', '', x) for x in symops]
    symmag = [sm.replace('m', '') for sm in symmag]  # remove m
    symcenmag = [sm.replace('m', '') for sm in symcenmag]  # remove m

    # Get magnetic symmetry constraints
    if '_magnetic_atom_site_moment_symmetry_constraints_label' in keys:
        const_labels = cifvals['_magnetic_atom_site_moment_symmetry_constraints_label']
        constraints = cifvals['_atom_site_magnetic_moment_symmetry_constraints_mxmymz']
    else:
        const_labels = []
        constraints = []

    # apply symmetry operations to each position, generating P1 structure
    # Also repeat each element property
    NsymOps = len(symops)
    NcenOps = len(symcen)
    Nops = NsymOps * NcenOps
    p1pos = np.empty([0, 3])
    p1vec = np.empty([0, 3])
    p1mag = []
    p1typ = []
    p1lbl = []
    p1occ = []
    p1Uiso = []
    p1tim = []
    for n, (x, y, z) in enumerate(cif_pos):
        uvw = gen_sym_pos(symops, x, y, z)
        uvw = fitincell(uvw)

        # Apply symmetry constraints
        if label[n] in const_labels:
            # Assumes single constraint per atom
            idx = const_labels.index(label[n])
            C = constraints[idx]
            C = C.replace('m', '')
            # print('Constraint: {:3s} {:s}'.format(label[n],C))
            # convert '2x' to 'x'
            old = re.findall(r'\d[xyz]', C)
            new = [s.replace('x', '*x').replace('y', '*y').replace('z', '*z') for s in old]
            for s in range(len(old)):
                C = C.replace(old[s], new[s])
            C = C.split(',')
            # Apply constraint to symmetry arguments
            S = [s.replace('x', 'a').replace('y', 'b').replace('z', 'c') for s in symmag]
            S = [s.replace('a', C[0]).replace('b', C[1]).replace('c', C[2]) for s in S]
        else:
            S = symmag

        mx, my, mz = cif_vec[n]
        mag_uvw = gen_sym_pos(S, mx, my, mz)
        # print('---{:3.0f}. {:2s} ({:5.2f},{:5.2f},{:5.2f})---'.format(n,label[n],mx,my,mz))

        # Loop over centring operations
        sympos = np.zeros([Nops, 3])
        symmagpos = np.zeros([Nops, 3])
        symtimpos = np.zeros(Nops)
        for m, (cx, cy, cz) in enumerate(uvw):
            Ni, Nj = m * NcenOps, (m + 1) * NcenOps
            cen_uvw = gen_sym_pos(symcen, cx, cy, cz)
            cen_uvw = fitincell(cen_uvw)
            sympos[Ni:Nj, :] = cen_uvw

            cmx, cmy, cmz = mag_uvw[m]
            mag_cen = gen_sym_pos(symcenmag, cmx, cmy, cmz)
            # print('  Sym{:2.0f}: {:10s} ({:5.2f},{:5.2f},{:5.2f})'.format(m,S[m],cmx,cmy,cmz))
            # for o in range(len(mag_cen)):
            # print('    Cen{:1.0f}: {:10s} ({:5.2f},{:5.2f},{:5.2f})'.format(o,symcenmag[o],*mag_cen[o]))
            symmagpos[Ni:Nj, :] = mag_cen
            symtimpos[Ni:Nj] = symtim[m] * np.array(symcentim)

        # Remove duplicates
        newarray, uniqueidx, matchidx = fg.unique_vector(sympos, 0.01)
        cen_uvw = sympos[uniqueidx, :]
        mag_cen = symmagpos[uniqueidx, :]
        symtimpos = symtimpos[uniqueidx]

        # Append to P1 atoms arrays
        p1pos = np.append(p1pos, cen_uvw, axis=0)
        p1vec = np.append(p1vec, mag_cen, axis=0)
        p1mag = np.append(p1mag, np.repeat(cif_mag[n], len(cen_uvw)))
        p1tim = np.append(p1tim, symtimpos)
        p1typ = np.append(p1typ, np.repeat(element[n], len(cen_uvw)))
        p1lbl = np.append(p1lbl, np.repeat(label[n], len(cen_uvw)))
        p1occ = np.append(p1occ, np.repeat(occ[n], len(cen_uvw)))
        p1Uiso = np.append(p1Uiso, np.repeat(Uiso[n], len(cen_uvw)))

    # Get space group
    if '_symmetry_space_group_name_H-M' in keys:
        spacegroup = cifvals['_symmetry_space_group_name_H-M']
    elif '_space_group_name_H-M_alt' in keys:
        spacegroup = cifvals['_space_group_name_H-M_alt']
    elif len(symops) == 1:
        spacegroup = 'P1'
    else:
        spacegroup = ''
    if '_symmetry_Int_Tables_number' in keys:
        sgn = float(cifvals['_symmetry_Int_Tables_number'])
    elif '_space_group_IT_number' in keys:
        sgn = float(cifvals['_space_group_IT_number'])
    elif spacegroup == 'P1':
        sgn = 1
    else:
        sgn = 0

    # Add values to the dict
    crys = {
        'filename': cifvals['Filename'],
        'name': cifvals['FileTitle'],
        'unit vector': UV,
        'parent cell': UV,
        'origin': np.array([[0., 0., 0.]]),
        'symmetry': symops,
        'atom type': p1typ,
        'atom label': p1lbl,
        'atom position': p1pos,
        'atom occupancy': p1occ,
        'atom uiso': p1Uiso,
        'atom uaniso': np.zeros([len(p1pos), 6]),
        'mag moment': p1vec, 'mag time': p1tim,
        'normalise': 1.,
        'misc': [element, label, cif_pos, occ, Uiso],
        'space group': spacegroup,
        'space group number': sgn,
        'cif': cifvals
    }
    return crys


def write_cif(cifvals, filename=None, comments=None):
    """
    Write .cif file with values from a cif dict,
    Only basic items are saved, rather than returning the orginal file
    :param cifvals: dict from readcif
    :param filename: filename to write (use None to return string)
    :param comments: str comments to write to the file top matter
    :return: None
    """

    keys = cifvals.keys()

    def cif_value(name):
        if name in keys:
            return '%-40s %-12s\n' % (name, cifvals[name])
        else:
            print('%s not in cif dict' % name)
            return '%-40s %-12s\n' % (name, '?')

    def cif_loop(names):
        names = [name for name in names if name in keys]
        if len(names) == 0:
            print('Loop Items not in cif dict')
            return ''
        out = 'loop_\n'
        out += ''.join(['%s\n' % name for name in names])
        vals = [cifvals[name] for name in names]
        for val_line in zip(*vals):
            out += ' '.join(['%-12s' % val for val in val_line])
            out += '\n'
        return out

    # Top Matter
    c = '#----------------------------------------------------------------------\n'
    c += '#   Crystal Structure: %s\n' % (cifvals['FileTitle'] if 'FileTitle' in keys else '')
    c += '#----------------------------------------------------------------------\n'
    c += '# CIF created in Dans_Diffraction\n'
    c += '# Original cif:\n# %s\n' % (cifvals['Filename'] if 'Filename' in keys else 'None')

    # Comments
    c += '# Comments:\n'
    if comments:
        comments = comments.split('\n')
        c += ''.join(['# %s\n' % comment for comment in comments])

    # Crystal Data
    c += '\ndata_WRITECIF\n'
    c += cif_value('_chemical_name_mineral')
    c += cif_value('_chemical_name_common')
    c += cif_value('_pd_phase_name')
    c += cif_value('_chemical_formula_sum')
    c += cif_value('_chemical_formula_weight')
    c += cif_value('_cell_formula_units_Z')

    # Cell info
    c += '\n# Cell info\n'
    c += cif_value('_cell_length_a')
    c += cif_value('_cell_length_b')
    c += cif_value('_cell_length_c')
    c += cif_value('_cell_angle_alpha')
    c += cif_value('_cell_angle_beta')
    c += cif_value('_cell_angle_gamma')
    c += cif_value('_cell_volume')

    # Symmetry info
    c += '\n# Symmetry info\n'
    c += cif_value('_symmetry_cell_setting')
    c += cif_value('_symmetry_space_group_name_H-M')
    c += cif_value('_symmetry_space_group_name_Hall')
    c += cif_value('_symmetry_Int_Tables_number')
    c += '\n'
    c += cif_loop(['_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz'])

    # Atom info
    c += '\n# Atom info\n'
    c += cif_loop([
        '_atom_site_label',
        '_atom_site_type_symbol',
        # '_atom_site_symmetry_multiplicity',
        # '_atom_site_Wyckoff_symbol',
        '_atom_site_fract_x',
        '_atom_site_fract_y',
        '_atom_site_fract_z',
        '_atom_site_U_iso_or_equiv',
        '_atom_site_occupancy',
    ])

    if filename is None:
        return c

    filename, extension = os.path.splitext(filename)
    filename = filename + '.cif'
    with open(filename, 'wt') as f:
        f.write(c)
    print('CIF written to: %s' % filename)


def write_mcif(cifvals, filename=None, comments=None):
    """
    Write magnetic .mcif file with values from a cif dict,
    Only basic items are saved, rather than returning the original file
    :param cifvals: dict from readcif
    :param filename: filename to write (use None to return string)
    :param comments: str comments to write to the file top matter
    :return: None
    """

    keys = cifvals.keys()

    def cif_value(name):
        if name in keys:
            return '%-40s %-12s\n' % (name, cifvals[name])
        else:
            print('%s not in cif dict' % name)
            return '%-40s %-12s\n' % (name, '?')

    def cif_loop(names):
        cnames = [name for name in names if name in keys]
        if len(cnames) < len(names):
            inames = [name for name in names if name not in keys]
            print('Loop Items not in cif dict: %s' % inames)
        if len(cnames) == 0:
            return ''
        out = 'loop_\n'
        out += ''.join(['%s\n' % name for name in cnames])
        vals = [cifvals[name] for name in cnames]
        for val_line in zip(*vals):
            out += ' '.join(['%-12s' % val for val in val_line])
            out += '\n'
        return out

    # Top Matter
    c = '#----------------------------------------------------------------------\n'
    c += '#   Crystal Structure: %s\n' % (cifvals['FileTitle'] if 'FileTitle' in keys else '')
    c += '#----------------------------------------------------------------------\n'
    c += '# MCIF created in Dans_Diffraction\n'
    c += '# Original cif:\n# %s\n' % (cifvals['Filename'] if 'Filename' in keys else 'None')

    # Comments
    c += '# Comments:\n'
    if comments:
        comments = comments.split('\n')
        c += ''.join(['# %s\n' % comment for comment in comments])

    # Crystal Data
    c += '\ndata_WRITECIF\n'
    c += cif_value('_chemical_name_mineral')
    c += cif_value('_chemical_name_common')
    c += cif_value('_pd_phase_name')
    c += cif_value('_chemical_formula_sum')
    c += cif_value('_chemical_formula_weight')
    c += cif_value('_cell_formula_units_Z')

    # Cell info
    c += '\n# Cell info\n'
    c += cif_value('_cell_length_a')
    c += cif_value('_cell_length_b')
    c += cif_value('_cell_length_c')
    c += cif_value('_cell_angle_alpha')
    c += cif_value('_cell_angle_beta')
    c += cif_value('_cell_angle_gamma')
    c += cif_value('_cell_volume')

    # Symmetry info
    c += '\n# Symmetry info\n'
    c += cif_value('_space_group_magn.number_BNS')
    c += cif_value('_space_group_magn.name_BNS')
    c += '\n'
    c += cif_loop([
        '_space_group_symop_magn_operation.id',
        '_space_group_symop_magn_operation.xyz',
        '_space_group_symop_magn_operation.mxmymz',
    ])
    c += '\n'
    c += cif_loop([
        '_space_group_symop_magn_centering.id',
        '_space_group_symop_magn_centering.xyz',
        '_space_group_symop_magn_centering.mxmymz',
    ])

    # Atom info
    c += '\n# Atom info\n'
    c += cif_loop([
        '_atom_site_label',
        '_atom_site_type_symbol',
        # '_atom_site_symmetry_multiplicity',
        # '_atom_site_Wyckoff_symbol',
        '_atom_site_fract_x',
        '_atom_site_fract_y',
        '_atom_site_fract_z',
        '_atom_site_U_iso_or_equiv',
        '_atom_site_occupancy',
    ])

    # Moment info
    c += '\n# Atom info\n'
    c += cif_loop([
        '_atom_site_moment.label',
        '_atom_site_moment.crystalaxis_x',
        '_atom_site_moment.crystalaxis_y',
        '_atom_site_moment.crystalaxis_z',
        # '_atom_site_moment.symmform',
    ])

    if filename is None:
        return c

    filename, extension = os.path.splitext(filename)
    filename = filename + '.mcif'
    with open(filename, 'wt') as f:
        f.write(c)
    print('MCIF written to: %s' % filename)


'--------------Functions to Read Database files----------------------'


def read_atom_properties_file(filedir):
    """
    Reads the text file "Dans Element Properties.txt"
    Returns a list of dicts containing atomic properites from multiple sources
      data = read_atom_properties_file(filedir)
      data[22]['Element']
    :param filedir: location of "Dans Element Properties.txt"
    :return: [dict, ...]
    """
    with open(filedir, 'rt') as f:
        lines = f.readlines()

    head = None
    data = []
    for line in lines:
        # Header
        if '#' in line: continue
        if head is None: head = line.split(); continue
        # Data
        vals = line.split()
        element = {}
        for n in range(len(vals)):
            try:
                value = int(vals[n])
            except ValueError:
                try:
                    value = float(vals[n])
                except ValueError:
                    value = vals[n]
            element[head[n]] = value
        data += [element]
    return data


def atom_properties(elements=None, fields=None):
    """
    Loads the atomic properties of a particular atom from a database
    Atomic properties, scattering lengths, form factors and absorption edges for elements upto and including Uranium.
    Values are taken from various online sources, see data/README.md for more details.

    Usage:
            A = atom_properties() >> returns structured array of all properties for all atoms A[0]['Element']='H'
            A = atom_properties('Co') >> returns structured array of all properties for 1 element
            B = atom_properties('Co','Weight') >> returns regular 1x1 array
            B = atom_properties(['Co','O'],'Weight') >> retruns regular 2x1 array
            A = atom_properties('Co',['a1','b1','a2','b2','a3','b3','a4','b4','c']) >> returns structured array of requested properties
            A = atom_properties(['Co','O'],['Z','Weight']) >> returns 2x2 structured array

    Available properties:
        A = atom_properties()
        print(A.dtype.names)
        print(a[25]['Element'])
        >> 'Fe'

    Available information includes:
          Z             = Element number
          Element       = Element symbol
          Name          = Element name
          Group         = Element Group on periodic table
          Period        = Element period on periodic table
          Block         = Element electronic block (s,p,d,f)
          ValenceE      = Number of valence electrons
          Config        = Element electronic orbital configuration
          Radii         = Atomic Radii (pm)
          Weight        = Standard atomic weight (g)
          Coh_b         = Bound coherent neutron scattering length
          Inc_b         = Bound incoherent neutron scattering length
          Nabs          = Neutron absorption coefficient
          Nscat         = Neutron scattering coefficient
          a1            = Electron form factor
          b1            = Electron form factor
          a2            = Electron form factor
          b2            = Electron form factor
          a3            = Electron form factor
          b3            = Electron form factor
          a4            = Electron form factor
          b4            = Electron form factor
          c             = Electron form factor
          j0_A          = Magnetic Form factor
          j0_a          = Magnetic Form factor
          j0_B          = Magnetic Form factor
          j0_b          = Magnetic Form factor
          j0_C          = Magnetic Form factor
          j0_c          = Magnetic Form factor
          j0_D          = Magnetic Form factor
          K             = x-ray absorption edge
          L1            = x-ray absorption edge
          L2            = x-ray absorption edge
          L3            = x-ray absorption edge
          M1...         = x-ray absorption edge
    """

    try:
        data = np.genfromtxt(ATOMFILE, skip_header=6, dtype=None, names=True, encoding='ascii')
    except TypeError:
        # Numpy version < 1.14
        data = np.genfromtxt(ATOMFILE, skip_header=6, dtype=None, names=True)

    if elements is not None:
        # elements must be a list e.g. ['Co','O']
        elements = [str2element(el) for el in np.asarray(elements).reshape(-1)]
        all_elements = data['Element'].tolist()
        # This will error if the required element doesn't exist
        try:
            # regex to remove additional characters
            index = [all_elements.index(el) for el in elements]
        except ValueError as ve:
            raise Exception('Element not available: %s' % ve)
        data = data[index]
        if np.any(data['Z'] > 92):
            msg = 'Element %s does not have complete atomic properties, scattering calculations will be inaccurate.'
            for el in data[data['Z'] > 92]:
                warn(msg % el['Element'])

    if fields is None:
        return data

    return data[fields]


def print_atom_properties(elements=None):
    """
    Outputs string of stored atomic properties
    :param elements: str or list or None
            str: 'Co'
            list: ['Co', 'O']
    :return: str
    """

    prop = atom_properties(elements)
    keys = prop.dtype.names
    elename = ' '.join(['%10s' % ele for ele in prop['Element']])
    out = '%8s : %s\n' % ('', elename)
    for key in keys:
        propval = ' '.join(['%10s' % ele for ele in prop[key]])
        out += '%8s : %s\n' % (key, propval)
    return out


def read_waaskirf_scattering_factor_coefs():
    """
    Read X-ray scattering factor table
    Uses the coefficients for analytical approximation to the scattering factors from:
       "Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431"
    File from https://github.com/diffpy/libdiffpy/blob/master/src/runtime/f0_WaasKirf.dat
    :return: {'element': array([a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5])}
    """
    data = np.loadtxt(WAASKIRF_FILE)
    with open(WAASKIRF_FILE) as f:
        lines = re.findall(r'#S\s+\d+\s+[A-Z].*?\n', f.read())
        table_names = [line[7:].strip() for line in lines]
    return {element: coefs for element, coefs in zip(table_names, data)}


def read_neutron_scattering_lengths(table='neutron data booklet'):
    """
    Read neutron scattering length of element or isotope
    Returns table of complex bound coherent neutron scattering length, in fm for elements and isotopes

        Natural average for each element given by 'element', e.g. 'Ti'
        Isotope value given by 'weight-element', e.g. '46-Ti'

    Default values are extracted from Periodic Table https://github.com/pkienzle/periodictable
     - Values originally from Neutron Data Booklet, by A-J Dianoux, G. Lander (2003), with additions and corrections upto v1.7.0 (2023)

    Alternative values are also available:
     - ITC Vol. C, Section 4.4.4., By V. F. Sears Table 4.4.4.1 (Jan 1995)

    :param table: name of data table to use, options are 'neutron data booklet' or 'sears'
    :return: {'isotope': complex, ...}
    """
    if table.lower() in ['sears', 'itc']:
        table_file = NSLFILE_SEARS
    else:
        table_file = NSLFILE

    nsl = {}
    with open(table_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            isotope, real, imag = line.split(',')
            nsl[isotope.lower()] = complex(float(real), float(imag))
    return nsl


def neutron_scattering_length(elements, table='neutron data booklet'):
    """
    Return neutron scattering length, b, in fm

        b = neutron_scattering_length('Co')

    Now includes complex neutron scattering lengths for isotopes from package periodictable
        b = neutron_scattering_length('7-Li')

    Lists of elements also allowed
        [bLi, bCo, bO] = neutron_scattering_length(['7-Li', 'Co', 'O'])

    Default values are extracted from Periodic Table https://github.com/pkienzle/periodictable
     - Values originally from Neutron Data Booklet, by A-J Dianoux, G. Lander (2003), with additions and corrections upto v1.7.0 (2023)

    Alternative values are also available:
     - ITC Vol. C, Section 4.4.4., By V. F. Sears Table 4.4.4.1 (Jan 1995)

    :param elements: [n*str] list or array of elements
    :param table: name of data table to use, options are 'neutron data booklet' or 'sears'
    :return: [n] array of scattering lengths
    """
    # b = atom_properties(element, ['Coh_b'])
    nsl = read_neutron_scattering_lengths(table)
    b_lengths = np.array([
        nsl[element] if element in nsl else nsl.get(split_element_symbol(element)[0], 0)
        for element in np.char.lower(np.asarray(elements).reshape(-1))
    ])
    return b_lengths


def xray_scattering_factor(element, Qmag=0):
    """
    Read X-ray scattering factor table, calculate f(|Q|)
    Uses the coefficients for analytical approximation to the scattering factors - ITC, p578 Table 6.1.1.4
     Qff = xray_scattering_factor(element, Qmag=[0])
    :param element: [n*str] list or array of elements
    :param Qmag: [m] array of wavevector distance, in A^-1
    :return: [m*n] array of scattering factors
    """

    # Qmag should be a 1D array
    Qmag = np.asarray(Qmag).reshape(-1)

    coef = atom_properties(element, ['a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'a4', 'b4', 'c'])

    Qff = np.zeros([len(Qmag), len(coef)])

    # Loop over elements
    for n in range(len(coef)):
        a1 = coef['a1'][n]
        b1 = coef['b1'][n]
        a2 = coef['a2'][n]
        b2 = coef['b2'][n]
        a3 = coef['a3'][n]
        b3 = coef['b3'][n]
        a4 = coef['a4'][n]
        b4 = coef['b4'][n]
        c = coef['c'][n]

        # Array multiplication over Qmags
        f = a1 * np.exp(-b1 * (Qmag / (4 * np.pi)) ** 2) + \
            a2 * np.exp(-b2 * (Qmag / (4 * np.pi)) ** 2) + \
            a3 * np.exp(-b3 * (Qmag / (4 * np.pi)) ** 2) + \
            a4 * np.exp(-b4 * (Qmag / (4 * np.pi)) ** 2) + c
        Qff[:, n] = f
    return Qff


def xray_scattering_factor_WaasKirf(element, Qmag=0):
    """
    Read X-ray scattering factor table, calculate f(|Q|)
    Uses the coefficients for analytical approximation to the scattering factors from:
       "Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431"
    File from https://github.com/diffpy/libdiffpy/blob/master/src/runtime/f0_WaasKirf.dat
     Qff = xray_scattering_factor_WaasKirf(element, Qmag=[0])
    :param element: [n*str] list or array of elements
    :param Qmag: [m] array of wavevector distance, in A^-1
    :return: [m*n] array of scattering factors
    """

    data = np.loadtxt(WAASKIRF_FILE)
    # get names
    with open(WAASKIRF_FILE) as f:
        lines = re.findall(r'#S\s+\d+\s+[A-Z].*?\n', f.read())
        table_names = [line[7:].strip() for line in lines]

    # Qmag should be a 1D array
    Qmag = np.asarray(Qmag).reshape(-1)
    element = np.asarray(element, dtype=str).reshape(-1)

    # data table: a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5
    idx = [table_names.index(el) for el in element]
    coef = data[idx, :]

    Qff = np.zeros([len(Qmag), len(element)])

    # Loop over elements
    for n in range(len(element)):
        a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5 = coef[n, :]

        # Array multiplication over Qmags
        # f0[k] = c + [SUM a_i * EXP(-b_i * (k ^ 2))]  i=1,5
        f = a1 * np.exp(-b1 * (Qmag / (4 * np.pi)) ** 2) + \
            a2 * np.exp(-b2 * (Qmag / (4 * np.pi)) ** 2) + \
            a3 * np.exp(-b3 * (Qmag / (4 * np.pi)) ** 2) + \
            a4 * np.exp(-b4 * (Qmag / (4 * np.pi)) ** 2) + \
            a5 * np.exp(-b5 * (Qmag / (4 * np.pi)) ** 2) + c
        Qff[:, n] = f
    return Qff


def xray_scattering_factor_resonant(elements, Qmag, energy_kev, use_waaskirf=False):
    """
    Read X-ray scattering factor table, calculate f(|Q|)
    Uses the coefficients for analytical approximation to the scattering factors - ITC, p578
     Qff = xray_scattering_factor_resonant(element, energy_kev, qmag)

    if resonant_energy = float(energy in keV), the total atomic scattering amplitude will be returned:
        f(|Q|, E) = f0(|Q|) + f'(E) - if''(E)
    See:
    :param elements: [n*str] list or array of elements
    :param Qmag: [m] array wavevector distance |Q|, in A^-1
    :param energy_kev: [o] array energy in keV
    :param use_waaskirf: if True, use f0 scattering factor coefficients from the table of Waasmaier and Kirfel
    :return: [m*n*o] complex array of scattering factors
    """

    # Qmag and energy_kev should be a 1D array
    Qmag = np.asarray(Qmag).reshape(-1)
    energy_kev = np.asarray(energy_kev).reshape(-1)

    f1, f2 = xray_dispersion_corrections(elements, energy_kev)  # shape (len(energy), len(element))
    Qff = np.zeros([len(Qmag), len(elements), len(energy_kev)], dtype=complex)
    # Broadcast dispersion corrections
    Qff[:, :, :] = f1.T - 1j * f2.T  # change from + to - on 2/July/2023

    if use_waaskirf:
        f = xray_scattering_factor_WaasKirf(elements, Qmag)
    else:
        f = xray_scattering_factor(elements, Qmag)

    for n in range(len(elements)):
        for e in range(len(energy_kev)):
            Qff[:, n, e] += f[:, n]
    return Qff


def electron_scattering_factor(element, Qmag=0):
    """
    Read X-ray scattering factor table, calculate f(|Q|)
    Uses the coefficients for analytical approximation to the scattering factors 
      Peng, L. M.; Acta Crystallogr A  1996, 52 (2), 257–276. 
      Peng, L.-M.  Acta Cryst A 1998, 54 (4), 481–485. 
    Qff = xray_scattering_factor(element, Qmag=[0])
    :param element: [n*str] list or array of elements
    :param Qmag: [m] array of wavevector distance, in A^-1
    :return: [m*n] array of scattering factors
    """

    # Qmag should be a 1D array
    Qmag = np.asarray(Qmag).reshape(-1)

    try:
        data = np.genfromtxt(PENGFILE, skip_header=0, dtype=None, names=True, encoding='ascii', delimiter=',')
    except TypeError:
        # Numpy version < 1.14
        data = np.genfromtxt(PENGFILE, skip_header=0, dtype=None, names=True, delimiter=',')
    # elements must be a list e.g. ['Co','O']
    elements_l = np.asarray(element).reshape(-1)
    all_elements = [el for el in data['Element']]
    try:
        index = [all_elements.index(el) for el in elements_l]
    except ValueError as ve:
        raise Exception('Element not available: %s' % ve)
    data = data[index]
    coef = data[['a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'a4', 'b4', 'a5', 'b5']]

    Qff = np.zeros([len(Qmag), len(coef)])

    # Loop over elements
    for n in range(len(coef)):
        a1 = coef['a1'][n]
        b1 = coef['b1'][n]
        a2 = coef['a2'][n]
        b2 = coef['b2'][n]
        a3 = coef['a3'][n]
        b3 = coef['b3'][n]
        a4 = coef['a4'][n]
        b4 = coef['b4'][n]
        a5 = coef['a5'][n]
        b5 = coef['b5'][n]

        # Array multiplication over Qmags
        f = a1 * np.exp(-b1 * (Qmag / (4 * np.pi)) ** 2) + \
            a2 * np.exp(-b2 * (Qmag / (4 * np.pi)) ** 2) + \
            a3 * np.exp(-b3 * (Qmag / (4 * np.pi)) ** 2) + \
            a4 * np.exp(-b4 * (Qmag / (4 * np.pi)) ** 2) + \
            a5 * np.exp(-b5 * (Qmag / (4 * np.pi)) ** 2)
        Qff[:, n] = f
    return Qff


def scattering_factor_coefficients_neutron_ndb(*elements):
    """
    Load neutron scattering factor coefficents

    Values are extracted from Periodic Table https://github.com/pkienzle/periodictable
     - Values originally from Neutron Data Booklet, by A-J Dianoux, G. Lander (2003), with additions and corrections upto v1.7.0 (2023)

    :param elements: str element symbol, must appear in the selected table, or zero is returned
    :return: list[float, float, ...] - list of pairs of coefficients
    """
    coefs = np.zeros([len(elements), 2])
    b_lengths = neutron_scattering_length(elements, 'neutron data booklet')
    for n, (element, b_length) in enumerate(zip(elements, b_lengths)):
        coefs[n, :] = [b_length, 0]
    return coefs


def scattering_factor_coefficients_neutron_sears(*elements):
    """
    Load neutron scattering factor coefficents

    Values are taken form the table:
     - ITC Vol. C, Section 4.4.4., By V. F. Sears Table 4.4.4.1 (Jan 1995)

    :param elements: str element symbol, must appear in the selected table, or zero is returned
    :return: array[n, c] - c is list of pairs of coefficients
    """
    coefs = np.zeros([len(elements), 2], dtype=complex)
    b_lengths = neutron_scattering_length(elements, 'sears')
    for n, (element, b_length) in enumerate(zip(elements, b_lengths)):
        coefs[n, :] = [b_length, 0]
    return coefs


def scattering_factor_coefficients_xray_itc(*elements):
    """
    Load x-ray scattering factor coefficents

    Uses the coefficients for analytical approximation to the scattering factors - ITC, p578 Table 6.1.1.4

    :param elements: str element symbol, must appear in the selected table, or zero is returned
    :return: array[n, c] - c is list of pairs of coefficients
    """
    out = np.zeros([len(elements), 10])
    data = atom_properties(None,  ['Element', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'a4', 'b4', 'c'])
    all_elements = list(data['Element'])
    for n, element in enumerate(elements):
        symbol, occupancy, charge = split_element_symbol(element)
        if element in all_elements:
            symbol = element
        if symbol in all_elements:
            idx = all_elements.index(symbol)
            el, a1, b1, a2, b2, a3, b3, a4, b4, c = data[idx]
            out[n, :] = [a1, b1, a2, b2, a3, b3, a4, b4, c, 0]
    return out


def scattering_factor_coefficients_xray_waaskirf(*elements):
    """
    Load x-ray scattering factor coefficents

    Uses the coefficients for analytical approximation to the scattering factors from:
       "Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431"
    File from https://github.com/diffpy/libdiffpy/blob/master/src/runtime/f0_WaasKirf.dat

    :param elements: str element symbol, must appear in the selected table, or zero is returned
    :return: array[n, c] - c is list of pairs of coefficients
    """
    out = np.zeros([len(elements), 12])

    data = read_waaskirf_scattering_factor_coefs()
    for n, element in enumerate(elements):
        symbol, occupancy, charge = split_element_symbol(element)
        if element in data:
            symbol = element
        if symbol in data:
            a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5 = data[symbol]
            out[n, :] = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c, 0]
    return out


def scattering_factor_coefficients_electron_peng(*elements):
    """
    Load electron scattering factor coefficents

    Uses the coefficients for analytical approximation to the scattering factors
      Peng, L. M.; Acta Crystallogr A  1996, 52 (2), 257–276.
      Peng, L.-M.  Acta Cryst A 1998, 54 (4), 481–485.

    :param elements: str element symbol, must appear in the selected table, or zero is returned
    :return: array[n, c] - c is list of pairs of coefficients
    """
    out = np.zeros([len(elements), 10])

    try:
        data = np.genfromtxt(PENGFILE, skip_header=0, dtype=None, names=True, encoding='ascii', delimiter=',')
    except TypeError:
        # Numpy version < 1.14
        data = np.genfromtxt(PENGFILE, skip_header=0, dtype=None, names=True, delimiter=',')

    all_elements = list(data['Element'])
    for n, element in enumerate(elements):
        symbol, occupancy, charge = split_element_symbol(element)
        if element in all_elements:
            symbol = element
        if symbol in all_elements:
            idx = all_elements.index(symbol)
            a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = data[idx][['a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'a4', 'b4', 'a5', 'b5']]
            out[n, :] = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    return out


def scattering_factor_coefficients(*elements, table='itc'):
    """
    Load scattering factor coefficents from differnt tables

    table options:
        'itc' -> x-ray scattering factors from international tabels Volume C (ITC, p578 Table 6.1.1.4)
        'waaskirf' -> x-ray scattering factors from Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431
        'peng' -> electron scattering factors from Peng, L.-M.  Acta Cryst A 1998, 54 (4), 481–485.
        'ndb' -> neutron scattering lengths from Neutron data booklet
        'sears' -> neutron scattering lengths from the international tables

    Coefficients are used in the analytical approximation of the scattering factor:
        f0[k] = c + [SUM a_i * EXP(-b_i * (k ^ 2))]  i=1,j
    where k is the wavevector, j varies depending on which table is used.

    coefficients are returned in pairs (a_i, b_i), where c is given as the final element (c, 0):
        coef = [(a_0, b_0), (a_1, b_1), ..., (c, 0)]
    :param elements: str element symbol, must appear in the selected table, or zero is returned
    :param table: str table name
    :return: array[n, c] - c is list of pairs of coefficients
    """
    # scattering factor ables
    tables = {
        'itc': scattering_factor_coefficients_xray_itc,
        'waaskirf': scattering_factor_coefficients_xray_waaskirf,
        'peng': scattering_factor_coefficients_electron_peng,
        'ndb': scattering_factor_coefficients_neutron_ndb,
        'sears': scattering_factor_coefficients_neutron_sears
    }
    table = table.lower()
    if table not in tables:
        raise ValueError(f'Unknown scattering factor table: {table}')
    return tables[table](*elements)


def analytical_scattering_factor(q_mag, *coefs):
    """
    Calculate the analytical scattering factor

    f0[|Q|] = c + [SUM a_i * EXP(-b_i * (|Q| ^ 2))]  i=1,n
    coefs = (a_1, b_1, a_2, b_2, ..., a_n, b_n)
    f0 = analytical_scattering_factor(q_mag, *coefs)

    :param q_mag: [m] array of wavevector distance, in A^-1
    :param coefs: float values of coefficients
    :return: [m] array of scattering factors
    """
    q_mag = np.asarray(q_mag, dtype=float).reshape(-1)
    q = (q_mag / (4 * np.pi)) ** 2
    # pad and reshape coefs into [n,2]
    coefs = np.reshape(np.pad(coefs, [0, len(coefs) % 2]), [-1, 2])
    f = sum(a * np.exp(-b * q) for a, b in coefs)
    return f


def add_custom_form_factor_coefs(element, *coefs, dispersion_table=None):
    """
    Custom form factor coefficients
    :param element: element name to add or replace
    :param coefs: a1, b1, a2, b2, ... scattering factor coefficients
    :param dispersion_table: None or array([energy_kev, f1, f2])
    :return:
    """
    CUSTOM_FORM_FACTOR_COEFS.update({element: coefs})
    if dispersion_table is not None:
        dispersion_table = np.asarray(dispersion_table, dtype=float)
        if dispersion_table.ndim != 2 or dispersion_table.shape[0] != 3:
            raise Exception(f"dispersion table wrong shape, should be (3, n): {dispersion_table.shape}")
        CUSTOM_DISPERSION_TABLE.update({element: dispersion_table})


def scattering_factor_coefficients_custom(*elements, default_table='itc'):
    """
    Load custom scattering factor coefficients from internal table
    :param elements: str element symbol
    :param default_table: scattering factor table to use if element not in custom list
    :return: {'element': array([a1, b1, a2, b2, ...])}
    """
    default_coefs = scattering_factor_coefficients(*elements, table=default_table)
    coefs = {
        el: CUSTOM_FORM_FACTOR_COEFS[el] if el in CUSTOM_FORM_FACTOR_COEFS else default_coefs[n]
        for n, el in enumerate(elements)
    }
    return coefs


def dispersion_table_custom(*elements):
    """
    Load custom table of x-ray dispersion corrections
        energy, f1, f2 = xray_dispersion_table_custom('element')
    :param elements: str element symbols
    :return: {'element': np.array([energy_kev, f1, f2])}
    """
    # Generate table
    log_step = 0.001  # 0.007 in original tables
    min_energy = 10.0
    max_energy = 30000
    energy_kev = 10 ** np.arange(np.log10(min_energy), np.log10(max_energy) + log_step, log_step) / 1000.
    f1, f2 = xray_dispersion_corrections(elements, energy_kev)

    tables = {
        el: CUSTOM_DISPERSION_TABLE[el]
        if el in CUSTOM_DISPERSION_TABLE else np.array([energy_kev, f1[:, n], f2[:, n]])
        for n, el in enumerate(elements)
    }
    return tables


def custom_scattering_factor(elements, q_mag, energy_kev=None, default_table='itc'):
    """
    Read X-ray scattering factor table, calculate f(|Q|)
     Qff = custom_scattering_factor(element, q_mag, energy_kev=None)

    if energy_kev = array(energy in keV), the total atomic scattering amplitude will be returned:
        f(|Q|, E) = f0(|Q|) + f'(E) - if''(E)

    where f0 is the custom scattering factor calculated from analytical coefficients,
    f' and f'' are the energy dependent dispersion corrections interpolated from
    tabulated data.

    if energy_kev is None, f' and f'' are not included.

    Analytical coefficients and dispersion correction tables are taken for each element
    from the interntal custom table (see add_custom_form_factor_coefs()).
    If the element symbol is not found in the internal table, instead they are taken
    from the default_table.

    default_table options:
        'itc' -> x-ray scattering factors from international tabels Volume C (ITC, p578 Table 6.1.1.4)
        'waaskirf' -> x-ray scattering factors from Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431
        'peng' -> electron scattering factors from Peng, L.-M.  Acta Cryst A 1998, 54 (4), 481–485.
        'ndb' -> neutron scattering lengths from Neutron data booklet
        'sears' -> neutron scattering lengths from the international tables

    :param elements: [n*str] list or array of elements
    :param q_mag: [m] array wavevector distance |Q|, in A^-1
    :param energy_kev: [o] array energy in keV or None [o==1]
    :param default_table: scattering factor table to use if element not in custom list
    :return: [m*n*o] complex array of scattering factors
    """

    if energy_kev is not None:
        return custom_scattering_factor_resonant(elements, q_mag, energy_kev, default_table)

    elements = np.asarray(elements, dtype=str).reshape(-1)
    q_mag = np.asarray(q_mag).reshape(-1)
    qff = np.zeros([len(q_mag), len(elements), 1], dtype=complex)
    element_coefs = scattering_factor_coefficients_custom(*elements, default_table=default_table)
    for n, el in enumerate(elements):
        coefs = element_coefs[el]
        f0 = analytical_scattering_factor(q_mag, *coefs)  # (n_q, )
        qff[:, n, 0] = f0
    return qff


def custom_scattering_factor_resonant(elements, q_mag, energy_kev, default_table='itc'):
    """
    Read X-ray scattering factor table, calculate f(|Q|)
     Qff = custom_scattering_factor(element, q_mag, energy_kev)

    if energy_kev = array(energy in keV), the total atomic scattering amplitude will be returned:
        f(|Q|, E) = f0(|Q|) + f'(E) - if''(E)

    where f0 is the custom scattering factor calculated from analytical coefficients,
    f' and f'' are the energy dependent dispersion corrections interpolated from
    tabulated data.

    Analytical coefficients and dispersion correction tables are taken for each element
    from the interntal custom table (see add_custom_form_factor_coefs()).
    If the element symbol is not found in the internal table, instead they are taken
    from the default_table.

    default_table options:
        'itc' -> x-ray scattering factors from international tabels Volume C (ITC, p578 Table 6.1.1.4)
        'waaskirf' -> x-ray scattering factors from Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431
        'peng' -> electron scattering factors from Peng, L.-M.  Acta Cryst A 1998, 54 (4), 481–485.
        'ndb' -> neutron scattering lengths from Neutron data booklet
        'sears' -> neutron scattering lengths from the international tables

    :param elements: [n*str] list or array of elements
    :param q_mag: [m] array wavevector distance |Q|, in A^-1
    :param energy_kev: [o] array energy in keV
    :param default_table: scattering factor table to use if element not in custom list
    :return: [m*n*o] complex array of scattering factors
    """

    # Qmag and energy_kev should be a 1D array
    elements = np.asarray(elements, dtype=str).reshape(-1)
    q_mag = np.asarray(q_mag).reshape(-1)
    energy_kev = np.asarray(energy_kev).reshape(-1)
    qff = np.zeros([len(q_mag), len(elements), len(energy_kev)], dtype=complex)

    element_coefs = scattering_factor_coefficients_custom(*elements, default_table=default_table)
    tables = dispersion_table_custom(*elements)
    for n, el in enumerate(elements):
        coefs = element_coefs[el]
        tab_en, tab_f1, tab_f2 = tables[el]
        f0 = analytical_scattering_factor(q_mag, *coefs)  # (n_q, )
        f1 = np.interp(energy_kev, tab_en, tab_f1)  # (n_en, )
        f2 = np.interp(energy_kev, tab_en, tab_f2)  # (n_en, )
        qff[:, n, :] = np.array([
            [
                _f0 if np.isnan(_f1) or np.isnan(_f2) else _f0 + (_f1 - 1j * _f2)
                for _f1, _f2 in zip(f1, f2)  # loop over energy
            ] for _f0 in f0  # loop over q values
        ])
    return qff


def load_magnetic_ff_coefs():
    """
    Magnetic Form Factor, Coefficients of the analytical approximation
    Downloaded from McPhase website, 10/5/2020
    http://www.mcphase.de/manual/node130.html
    Some modifications were required after downloading to read the file.
    :return: dict['element']['coefficient']
    """
    # Read McPhase data
    with open(MAGFF_FILE) as f:
        lines = f.readlines()
    lines = [l.replace('= ', '=').split() for l in lines if l[0] != '#']

    # Build dict
    mff_mcphase = {}
    for ln in lines:
        if ln[0] not in mff_mcphase:
            mff_mcphase[ln[0]] = {}
        for ii in ln[2:]:
            if '=' in ii:
                k, val = ii.split('=')
            else:
                raise NameError('%s\nline: %s has the wrong format' % (ln, ii))
            mff_mcphase[ln[0]][k] = float(val)
        el = str2element(ln[0])
        if el not in mff_mcphase:
            # Create default atom from first mention
            mff_mcphase[el] = mff_mcphase[ln[0]]
    return mff_mcphase


def magnetic_ff_symbol(element):
    """Convert element symbol into style used in magnetic form factor table"""
    name, occ, charge = split_element_symbol(element)
    charge = '' if int(charge) == 0 else f"{abs(charge):.0f}"
    return f"{name.capitalize()}{charge}"


def magnetic_ff_coefs(*elements):
    """Return the magnetic form factor coefficients for an element"""
    mff = load_magnetic_ff_coefs()
    value_keys = (
        'j0A', 'j0a', 'j0B', 'j0b', 'j0C', 'j0c', 'j0D',
        'j2A', 'j2a', 'j2B', 'j2b', 'j2C', 'j2c', 'j2D',
        'j4A', 'j4a', 'j4B', 'j4b', 'j4C', 'j4c', 'j4D',
    )
    coefs = np.zeros([len(elements), len(value_keys)])
    for n, ele in enumerate(elements):
        key = magnetic_ff_symbol(ele)
        if key in mff:
            coefs[n, :] = [mff[key][val] for val in value_keys]
        # else:
        #     print(f"Not in magnetic form factor table: {ele}")
    return coefs


def print_magnetic_ff_coefs(element):
    """
    Print analytical coefficients for the magnetic form factor of an element
    :param element: element symbol
    :return: str
    """
    mff = load_magnetic_ff_coefs()
    name, occ, charge = split_element_symbol(element)
    name = name.capitalize()
    symbol = magnetic_ff_symbol(element)
    out = ''
    for key, coefs in mff.items():
        if name in key:
            star = '*' if symbol == key else ''
            coef_str = ', '.join([f"{coef}={val:8}" for coef, val in coefs.items()])
            out += f"{star+key:5}: {coef_str}\n"
    return out


def last_df_orbital(element):
    """
    Return the orbital configuration of the last d or f orbital in element
    """
    name, occ, charge = split_element_symbol(element)
    return next(reversed([o for o in orbital_configuration(name, charge) if 'd' in o or 'f' in o]), '')


def magnetic_ff_g_factor(*elements):
    """
    Return the lande g-factor of the given elements
    """
    g_factors = []
    for ele in elements:
        orbital = last_df_orbital(ele)
        g_factors.append(glande(*hunds_rule(orbital)) if orbital else 0)
    return np.squeeze(g_factors)


def magnetic_ff_j2j0_ratio(gfactor, ratio_type=1):
    """
    Return the ratio of <J2>/<J0> based on the Lande g-factor
    This comes from using the dipole approximation (small |Q|):
        F(|Q|) = <j0(|Q|)> + (2-g)/g * <j2(|Q|)>,  where g is the Lande g-factor
    The ratio is <j2> / j<0> = (2 - g) / g
    if ratio_type == 2:
    The ratio is <j2> / j<0> = (1 - 2 / g)
    """
    if abs(gfactor) < 0.0001:
        return 0.0
    if ratio_type == 2:
        return 1 - (2 / gfactor)  # equ. 13 http://www.physics.mcgill.ca/~dominic/papers201x/CJP_88_2010_p771.pdf
    return (2 - gfactor) / gfactor  # equ. 155 http://www.mcphase.de/manual/node130.html#jls


def magnetic_form_factor(*elements, qmag=0., orbital_state=None, gfactor=None, j2j0_ratio=None):
    """
    Calcualte the magnetic form factor of an element or list of elements at wavevector |Q|
    Analytical approximation of the magnetic form factor:
        <j0(|Q|)> = A*exp(-a*s^2) + B*exp(-b*s^2) + C*exp(-c*s^2) + D, where s = sin(theta)/lambda in A-1
        <j2(|Q|)> = A*Q^2*exp(-a*s^2) + B*Q^2*exp(-b*s^2) + C*Q^2*exp(-c*s^2) + D*Q^2
    Using the dipole approximation (small |Q|):
        F(|Q|) = <j0(|Q|)> + (2-g)/g * <j2(|Q|)>,  where g is the Lande g-factor
    See more about the approximatio here: https://www.ill.eu/sites/ccsl/ffacts/ffactnode3.html
    Coefficients for the analytical approximation are available in the International Tables of Crystallography, Vol C
    Here they have been copied from the web, published by the program McPhase: http://www.mcphase.de/manual/node130.html
    The Lande g-factor is determined from the spin and orbital quantum numbers of the element in it's given state
    If the orbital state and g-factor is not given, the neutral d/f state is used (see atom_valence_state())
    E.G.
        mff = magnetic_form_factor('Co2+', qmag=np.arange(0,4,0.1), orbital_state='3d5')

    :param elements: str element symbols
    :param qmag: magntude of the wavevector transfer, |Q|, in A^-1
    :param orbital_state: str, valence orbital to use to calculate the g-factor (must be same length as element)
    :param gfactor: float, g-factor to use to calculate orbital component (must be same length as element)
    :param j2j0_ratio: float, value of <J2>/<J0>, rather than using g-factor
    :return: array([len(qmag), len(element)])
    """
    # s = sin(th)/lambda = |Q|/4pi
    s2 = (np.asarray(qmag).reshape(-1) / (4 * np.pi)) ** 2

    if j2j0_ratio is not None:
        j2j0_ratio = np.asarray(j2j0_ratio).reshape(-1)
    else:
        if orbital_state is not None:
            orbital_state = np.asarray(orbital_state, dtype=str).reshape(-1)
            gfactor = np.array([glande(*hunds_rule(o)) for o in orbital_state])
        elif gfactor is None:
            gfactor = magnetic_ff_g_factor(*elements)
        j2j0_ratio = np.array([magnetic_ff_j2j0_ratio(g) for g in np.reshape(gfactor, -1)])

    # replicate last element if array is shorter than elements
    j2j0_ratio = j2j0_ratio[list(range(j2j0_ratio.size)) + [-1] * (len(elements) - j2j0_ratio.size)]

    coefs = magnetic_ff_coefs(*elements)  # (len(ele) * 21)
    qff = np.zeros([len(s2), len(elements)])

    # Loop over elements
    for n, ((A0, a0, B0, b0, C0, c0, D0, A2, a2, B2, b2, C2, c2, D2), rat) in enumerate(zip(coefs[:, :14], j2j0_ratio)):
        j0 = A0 * np.exp(-a0 * s2) + \
             B0 * np.exp(-b0 * s2) + \
             C0 * np.exp(-c0 * s2) + D0
        j2 = (A2 * np.exp(-a2 * s2) +
              B2 * np.exp(-b2 * s2) +
              C2 * np.exp(-c2 * s2) + D2) * s2
        # Dipole approximation for small q
        qff[:, n] = j0 + (rat * j2)
    return qff


def magnetic_form_factor_old(element, Qmag=0.):
    """
    Read Magnetic form factor table, calculate <j0(|Q|)>
    Analytical approximation of the magnetic form factor:
        <j0(|Q|)> = A*exp(-a*s^2) + B*exp(-b*s^2) + C*exp(-c*s^2) + D, where s = sin(theta)/lambda in A-1
    See more about the approximatio here: https://www.ill.eu/sites/ccsl/ffacts/ffactnode3.html
    Note: Currently only J0 terms are used and 5d atoms are not currently included
    Usage:
         Qff = read_mff(element,Qmag)
         element = str element name, e.g. 'Co'
         Qmag = magnitude of Q=sin(theta)/lambda in A-1 at which to calcualte, can be a list or array to return multiple values
         Qff = Magnetic form factor for element at Qmag
    E.G.
        Qff = read_mff('Co',np.arange(0,4,0.1))
    """

    # Qmag should be a 1D array
    # s = sin(th)/lambda = |Q|/4pi
    s = np.asarray(Qmag).reshape(-1) / (4 * np.pi)

    coef = atom_properties(element, ['j0_A', 'j0_a', 'j0_B', 'j0_b', 'j0_C', 'j0_c', 'j0_D'])

    # Nqpointings x Nelements
    Qff = np.zeros([len(s), len(coef)])

    # Loop over elements
    for n in range(len(coef)):
        A = coef['j0_A'][n]
        a = coef['j0_a'][n]
        B = coef['j0_B'][n]
        b = coef['j0_b'][n]
        C = coef['j0_C'][n]
        c = coef['j0_c'][n]
        D = coef['j0_D'][n]

        j0 = A * np.exp(-a * s ** 2) + \
             B * np.exp(-b * s ** 2) + \
             C * np.exp(-c * s ** 2) + D
        Qff[:, n] = j0
    return Qff


def attenuation(element_z, energy_kev):
    """
     Returns the x-ray mass attenuation, u/p, in cm^2/g
       e.g. A = attenuation(23,np.arange(7,8,0.01)) # Cu
            A = attenuation([23,24,25], 5.6)
            a = attenuation(19,4.5) # K
    """
    element_z = np.asarray(element_z).reshape(-1)
    energy_kev = np.asarray(energy_kev).reshape(-1)

    xma_data = np.loadtxt(XMAFILE)

    energies = xma_data[:, 0] / 1000.
    out = np.zeros([len(energy_kev), len(element_z)])
    for n, z in enumerate(element_z):
        # Interpolating the log values is much more reliable
        out[:, n] = np.exp(np.interp(np.log(energy_kev), np.log(energies), np.log(xma_data[:, z])))
        out[:, n] = np.interp(energy_kev, energies, xma_data[:, z])
    if len(element_z) == 1:
        return out[:, 0]
    return out


def atomic_scattering_factor(element, energy_kev=None):
    """
    Read atomic scattering factor table, giving f1+f2 for different energies
    From: http://henke.lbl.gov/optical_constants/asf.html
    :param element: str or list of str, name of element. If element string includes a number, this will multiply values
    :param energy_kev: float or list energy in keV (None to return original, uninterpolated list)
    :return: f1, f2, shape dependent on shapes of element and energy_kev:  float, or [ene] or [ele, ene]
    """
    asf = np.load(ASFFILE, allow_pickle=True)
    asf = asf.item()

    element = np.asarray(element, dtype=str).reshape(-1)
    energy = {}
    f1 = {}
    f2 = {}
    for el in element:
        name, occ, charge = split_element_symbol(el)
        energy[el] = np.array(asf[name]['energy']) / 1000.  # eV -> keV
        f1[el] = np.array(asf[name]['f1'])
        f2[el] = np.array(asf[name]['f2'])
        f1[el][f1[el] < -1000] = np.nan
        f2[el][f2[el] < -1000] = np.nan
        f1[el] = f1[el] * occ
        f2[el] = f2[el] * occ

    if energy_kev is None:
        if len(element) == 1:
            return energy[el], f1[el], f2[el]
        return energy, f1, f2

    # Interpolate values
    if len(element) == 1:
        if1 = np.interp(energy_kev, energy[el], f1[el])
        if2 = np.interp(energy_kev, energy[el], f2[el])
        return if1, if2
    if1 = np.zeros([len(element), np.size(energy_kev)])
    if2 = np.zeros([len(element), np.size(energy_kev)])
    for n, el in enumerate(element):
        if1[n, :] = np.interp(energy_kev, energy[el], f1[el])
        if2[n, :] = np.interp(energy_kev, energy[el], f2[el])
    return if1, if2


def photoabsorption_crosssection(elements, energy_kev):
    """
    Calculate the photoabsorption cross section from the atomic scattering factors
        u = 2*r0*lambda*f2
    See: https://henke.lbl.gov/optical_constants/intro.html
    :param elements: str or list of string element symbol, if list, absorption will be summed over elements
    :param energy_kev: float or array x-ray energy
    :return: float or array [len(energy)] m^2
    """
    f1, f2 = atomic_scattering_factor(elements, energy_kev)
    f2 = f2.reshape(np.size(elements), np.size(energy_kev)).sum(axis=0)  # sum over elements
    wavelength = energy2wave(energy_kev) * 1e-10
    return 2 * fg.r0 * wavelength * f2


def xray_dispersion_corrections(elements, energy_kev=None):
    """
    Read xray dispersion corrections from atomic scattering factor table, giving f' and f" for different energies
    From: http://henke.lbl.gov/optical_constants/asf.html
    :param elements: list of str, name of element
    :param energy_kev: float or list energy in keV (None to return original, uninterpolated list)
    :return: f', f" with shape (len(energy), len(elements))
    """
    asf = np.load(ASFFILE, allow_pickle=True)
    asf = asf.item()

    elements = np.asarray(elements, dtype=str).reshape(-1)
    if energy_kev is None:
        if len(elements) == 1:
            energy_kev = np.array(asf[elements[0]]['energy']) / 1000.
        else:
            log_step = 0.003  # 0.007 in tables
            min_energy = 10.0
            max_energy = 30000
            energy_kev = 10**np.arange(np.log10(min_energy), np.log10(max_energy) + log_step, log_step) / 1000.
    else:
        energy_kev = np.asarray(energy_kev, dtype=float).reshape(-1)
    if1 = np.zeros([len(energy_kev), len(elements)])
    if2 = np.zeros([len(energy_kev), len(elements)])
    for n, el in enumerate(elements):
        if el in asf:
            z = asf[el]['Z']
            energy = np.array(asf[el]['energy']) / 1000.  # eV -> keV
            f1 = np.array(asf[el]['f1'])
            f2 = np.array(asf[el]['f2'])
            f1[f1 < -1000] = np.nan
            f2[f2 < -1000] = np.nan
            # interpolate and subtract f0==Z (where does this come from???)
            if1[:, n] = np.interp(energy_kev, energy, f1) - z
            if2[:, n] = -np.interp(energy_kev, energy, f2)
    return if1, if2


def pointgroups():
    """Read pointgroup file, return dict"""
    pg_file = os.path.join(datadir, 'PointGroups.json')
    with open(pg_file, 'r') as fp:
        pg = json.load(fp)
    return pg


def load_pointgroup(pg_number):
    """
    Load point group using number
    Point Groups:
    Triclinic
      1 C1  (    1) GenPos:   1
      2 Ci  (   -1) GenPos:   2
    Monoclinic
      3 C2  (    2) GenPos:   2
      4 Cs  (    m) GenPos:   2
      5 C2h (  2/m) GenPos:   4
    Orthorhombic
      6 D2  (  222) GenPos:   4
      7 C2v (  mm2) GenPos:   4
      8 D2h (  mmm) GenPos:   8
    Tetragonal
      9 C4  (    4) GenPos:   4
     10 S4  (   -4) GenPos:   4
     11 C4h (  4/m) GenPos:   8
     12 D4  (  422) GenPos:   8
     13 C4v (  4mm) GenPos:   8
     14 D2d ( -42m) GenPos:   8
     15 D4h (4/mmm) GenPos:  16
    Trigonal
     16 C3  (    3) GenPos:   3
     17 C3i (   -3) GenPos:   6
     18 D3  (  312) GenPos:   6
     19 C3v (  3m1) GenPos:   6
     20 D3d ( -31m) GenPos:  12
    Hexagonal
     21 C6  (    6) GenPos:   6
     22 C3h (   -6) GenPos:   6
     23 C6h (  6/m) GenPos:  12
     24 D6  (  622) GenPos:  12
     25 C6v (  6mm) GenPos:  12
     26 D3h ( -6m2) GenPos:  12
     27 D6h (6/mmm) GenPos:  24
    Cubic
     28 T   (   23) GenPos:  12
     29 Th  (  m-3) GenPos:  24
     30 O   (  432) GenPos:  24
     31 Td  ( -43m) GenPos:  24
     32 Oh  ( m-3m) GenPos:  48
    :param pg_number: int or str, e.g. 'cubic'
    :return: dict
    """
    try:
        if pg_number.lower() in ['cubic']:
            pg_number = 32
        elif pg_number.lower() in ['hexagonal', 'hex']:
            pg_number = 27
        elif pg_number.lower() in ['trigonal']:
            pg_number = 20
        elif pg_number.lower() in ['tetragonal']:
            pg_number = 15
        elif pg_number.lower() in ['orthorhombic']:
            pg_number = 8
        elif pg_number.lower() in ['monoclinic', 'mono']:
            pg_number = 5
        elif pg_number.lower() in ['triclinic']:
            pg_number = 2
    except AttributeError:
        pg_number = int(pg_number)
    return pointgroups()[str(pg_number)]


def spacegroups():
    """
    Return a dict of all space groups
    Loaded from json file created from the Bilbao crystallographic server: https://www.cryst.ehu.es/
    :return: dict with keys:
        'space group number'
        'space group name html'
        'space group name'
        'web address generator'
        'web address wyckoff'
        'web address site check'     # address % (u, v, w),
        'general positions'
        'magnetic space groups',
        'positions centring'
        'positions multiplicity'
        'positions wyckoff letter'
        'positions symmetry'
        'positions coordinates'
        'subgroup number',
        'subgroup name',
        'subgroup index',
        'subgroup type'
    """
    sg_file = os.path.join(datadir, 'SpaceGroups.json')
    with open(sg_file, 'r') as fp:
        sg_dict = json.load(fp)
    return sg_dict


def spacegroup(sg_number):
    """
    Return a dict of information for a space group
    :param sg_number: int space group number, as in the international tables of Crystallography
    :return: dict
    """
    sg_number = str(int(sg_number))
    sg_dict = spacegroups()
    return sg_dict[sg_number]


def spacegroup_list(sg_numbers=None):
    """
    Return a structured list of space groups with general operations
    :param sg_numbers: list of space group numbers (None for whole list)
    :return: str
    """
    if sg_numbers is None:
        sg_numbers = range(1, 231)
    sg_numbers = np.asarray(sg_numbers, dtype=str).reshape(-1)

    sg_dict = spacegroups()

    out = ''
    fmt = '%3d %-10s : %3d : %s\n'
    for sgn in sg_numbers:
        sg = sg_dict[sgn]
        num = sg['space group number']
        name = sg['space group name']
        sym = ', '.join(sg['general positions'])
        nsym = len(sg['general positions'])
        out += fmt % (num, name, nsym, sym)
    return out


def spacegroup_subgroups(sg_number):
    """
    Return dict of maximal subgroups for spacegroup
    :param sg_number: space group number (1-230)
    :return: dict
    """
    all_sg = spacegroups()
    sg_number = str(int(sg_number))
    sg_dict = all_sg[sg_number]
    subgroups = sg_dict["subgroup number"]
    return [all_sg[num] for num in subgroups]


def spacegroup_subgroups_list(sg_number=None, sg_dict=None):
    """
    Return str of maximal subgroups for spacegroup
    :param sg_number: space group number (1-230)
    :param sg_dict: alternate input, spacegroup dict from spacegroup function
    :return: str
    """
    if sg_number is not None:
        sg_dict = spacegroup(sg_number)
    sub_num = sg_dict["subgroup number"]
    sub_name = sg_dict["subgroup name"]
    sub_index = sg_dict["subgroup index"]
    sub_type = sg_dict["subgroup type"]
    out = ''
    fmt = 'Parent: %3s Subgroup: %3s  %-10s  Index: %3s  Type: %2s\n'
    for num, name, index, stype in zip(sub_num, sub_name, sub_index, sub_type):
        out += fmt % (sg_number, num, name, index, stype)
    return out


def spacegroups_magnetic(sg_number=None, sg_dict=None):
    """
    Returns dict of magnetic space groups for required space group
    :param sg_number: space group number (1-230) or None to return all magnetic spacegroups
    :param sg_dict: alternate input, spacegroup dict from spacegroup function
    :return: dict
    """

    msg_file = os.path.join(datadir, 'SpaceGroupsMagnetic.json')
    with open(msg_file, 'r') as fp:
        msg_dict = json.load(fp)

    if sg_number is None and sg_dict is None:
        return msg_dict
    if sg_number is not None:
        sg_dict = spacegroup(sg_number)
    msg_numbers = sg_dict['magnetic space groups']
    return [msg_dict[num] for num in msg_numbers]


def spacegroup_magnetic(msg_number):
    """
    Return dict of magnetic spacegroup, given magnetic spacegroup number
    :param msg_number: magnetic space group number e.g. 61.433
    :return: dict with keys:
        'parent number': sg,
        'space group number': number,
        'space group name': label,
        'setting': setting,
        'type name': type_name,
        'related group': rel_number,
        'related name': rel_name,
        'related setting': rel_setting,
        'operators general': xyz_op,
        'operators magnetic': mxmymz_op,
        'operators time': time,
    """
    msg_number = np.asarray(msg_number, dtype=str).reshape(-1)

    msg_file = os.path.join(datadir, 'SpaceGroupsMagnetic.json')
    with open(msg_file, 'r') as fp:
        msg_dict = json.load(fp)

    if None in msg_number:
        return msg_dict
    elif len(msg_number) == 1:
        return msg_dict[msg_number[0]]
    else:
        return [msg_dict[num] for num in msg_number]


def spacegroup_magnetic_list(sg_number=None, sg_dict=None):
    """
    Return str list of magnetic space groups
    :param sg_number: space group number (1-230)
    :param sg_dict: alternate input, spacegroup dict from spacegroup function
    :return: str
    """
    mag_spacegroups = spacegroups_magnetic(sg_number, sg_dict)

    out = ''
    fmt = 'Parent: %3s Magnetic: %-10s  %-10s  Setting: %3s %30s  Operators: %s\n'
    for sg in mag_spacegroups:
        parent = sg['parent number']
        number = sg['space group number']
        name = sg['space group name']
        setting = sg['setting']
        typename = sg['type name']
        # ops = sg['operators magnetic']
        ops = sg['positions magnetic']
        out += fmt % (parent, number, name, setting, typename, ops)
    return out


def find_spacegroup(sg_symbol):
    """
    Find a spacegroup based on the identifying symbol
    :param sg_symbol: str, e.g. 'Fd-3m'
    :return: spacegroup dict or None if not found
    """
    sg_symbol = sg_symbol.replace(' ', '').replace('\"', '')
    sg_dict = spacegroups()
    if str(sg_symbol) in sg_dict:
        return sg_dict[str(sg_symbol)]
    sg_keys = list(sg_dict.keys())
    sg_names = [sg['space group name'] for sg in sg_dict.values()]
    if sg_symbol in sg_names:
        key = sg_keys[sg_names.index(sg_symbol)]
        return sg_dict[key]

    sg_dict_mag = spacegroups_magnetic()
    if str(sg_symbol) in sg_dict_mag:
        return sg_dict_mag[str(sg_symbol)]
    sg_keys_mag = list(sg_dict_mag.keys())
    sg_names_mag = [sg['space group name'] for sg in sg_dict_mag.values()]
    if sg_symbol in sg_names_mag:
        key = sg_keys_mag[sg_names_mag.index(sg_symbol)]
        return sg_dict_mag[key]
    # Find first matching spacegroup
    sg_symbol = sg_symbol.lower()
    for stored_symbol in sg_names:
        if sg_symbol in stored_symbol.lower():
            return sg_dict[sg_keys[sg_names.index(stored_symbol)]]
    for stored_symbol in sg_names_mag:
        if sg_symbol in stored_symbol.lower():
            return sg_dict_mag[sg_keys_mag[sg_names_mag.index(stored_symbol)]]
    return None


def wyckoff_labels(spacegroup_dict, UVW):
    """
    Return Wyckoff site labels for given positions
    :param spacegroup_dict: spacegroup dict from tables, if magnetic spacegroup given, uses the parent spacegroup
    :param UVW: n*3 array([u,v,w]) atomic positions in fractional coordinates
    :return: list[n] of Wyckoff site letters
    """

    if 'parent number' in spacegroup_dict:
        # magnetic spacegroup - doesn't contain wyckoff letters
        spacegroup_dict = spacegroup(spacegroup_dict['parent number'])

    general_positions = spacegroup_dict['general positions']
    wyckoff_positions = spacegroup_dict['positions coordinates'][::-1]
    wyckoff_letters = spacegroup_dict['positions wyckoff letter'][::-1]
    multiplicity = spacegroup_dict['positions multiplicity'][::-1]
    symmetry = spacegroup_dict['positions symmetry'][::-1]

    sites = ['%s%s (%s)' % (m, l, s) for l, m, s in zip(wyckoff_letters, multiplicity, symmetry)]

    UVW = np.asarray(UVW, dtype=float).reshape((-1, 3))
    uvw_letters = [wyckoff_letters[-1] for n in range(len(UVW))]
    for n in range(len(UVW)):
        u, v, w = UVW[n]
        # For each position, generate all general positions
        sym_uvw = fitincell(gen_sym_pos(general_positions, u, v, w))
        for wyckoff_ops, letter in zip(wyckoff_positions, sites):
            # looping from smallest to most general, check if first Wyckoff position is in general positions
            trial_uvw = fitincell(gen_sym_pos(wyckoff_ops[:1], u, v, w))[0]
            diff = fg.mag(sym_uvw - trial_uvw)
            if diff.min() < 0.01:
                uvw_letters[n] = letter
                break
    return uvw_letters


'--------------Element Properties & Charge----------------------'


def element_symbol(element_Z=None):
    """
    Returns the element sympol for element_Z
    :param element_z: int or array or None for all elements
    :return: str
    """
    symbols = atom_properties(None, 'Element')
    if element_Z is None:
        return symbols
    element_Z = np.asarray(element_Z).reshape(-1)
    return symbols[element_Z - 1]


def element_z(element):
    """
    Returns the element number Z
    :param element: str
    :return: int
    """
    z = atom_properties(element, 'Z')
    if len(z) == 1:
        return z[0]
    return z


def element_name(element=None):
    """
    Returns the element name
    :param element: str
    :return: int
    """
    name = atom_properties(element, 'Name')
    if len(name) == 1:
        return name[0]
    return name


def split_element_symbol(element):
    """
    From element symbol, split charge and occupancy
      symbol, occupancy, charge = split_element_symbol('Co3+')
    Any numbers appended by +/- are taken as charge, otherwise they are counted as occupancy.
    e.g.    element     >   symbol  |   occupancy   |   charge
            Co3+            Co          1.              3
            3.2O2-          O           3.2             -2
            fe3             Fe          3               0
    :param element: str
    :return: symbol: str
    :return: occupancy: float
    :return: charge: float
    """
    # Find element
    symbol = element_regex.findall(element)[0]
    # Find charge
    find_charge = re.findall(r'[\d\.]+[\+-]', element)
    if len(find_charge) > 0:
        chargestr = find_charge[0]
        element = element.replace(chargestr, '')
        if '-' in chargestr:
            charge = -float(chargestr[:-1])
        else:
            charge = float(chargestr[:-1])
    else:
        charge = 0.
    # Find occupancy
    find_occ = re.findall(r'[\d\.]+', element)
    if len(find_occ) > 0:
        occupancy = float(find_occ[0])
    else:
        occupancy = 1.
    return symbol, occupancy, charge


def element_charge_string(symbol, occupancy=1.0, charge=0.0, latex=False):
    """
    Return formatted string of element with occupancy and charge
    :param symbol: str - element string
    :param occupancy: float - element occupancy or number of elements
    :param charge: float - element charge
    :param latex: if True, returns string formatted with latex tags
    :return: str
    """
    if abs(charge) < 0.01:
        chstr = ''
    elif abs(charge - 1) < 0.01:
        chstr = '+'
    elif abs(charge + 1) < 0.01:
        chstr = '-'
    elif charge > 0:
        chstr = '%0.3g+' % charge
    else:
        chstr = '%0.3g-' % abs(charge)

    if latex:
        chstr = '$^{%s}$' % chstr

    if np.abs(occupancy - 1) < 0.01:
        outstr = '%s%s' % (symbol, chstr)
    else:
        outstr = '%0.3g[%s%s]' % (occupancy, symbol, chstr)
    return outstr


def split_compound(compound_name):
    """
    Convert a molecular or compound name into a list of elements and numbers.
    Assumes all element multiplications are to the LEFT of the element name.
    Values in brackets are multiplied out.
    E.g.
        split_compound('Mn0.3(Fe3.6(Co1.2)2)4(Mo0.7Pr44)3')
        >> ['Mn0.3', 'Fe14.4', 'Co9.6', 'Mo2.1', 'Pr132']
    :param compound_name: str
    :return: list of str
    """
    regex_element_num = re.compile('|'.join([r'%s[\d\.]*' % el for el in ELEMENT_LIST]))
    # Multiply out brackets
    compound_name = fg.replace_bracket_multiple(compound_name)
    return regex_element_num.findall(compound_name)


def orbital_configuration(element, charge=None):
    """
    Returns the orbital configuration of an element as a list of strings
    :param element: str, element name, e.g. 'Fe' or 'Fe3+' or '0.8Fe2+'
    :param charge: int, element charge (overwrites charge given in element)
    :return: ['1s2', '2s2', '2p6', ...]
    """
    symbol, occupancy, charge_str = split_element_symbol(element)
    z = element_z(symbol)
    if charge is None:
        charge = charge_str
    newz = z - int(charge)  # floor
    if newz < 1: newz = 1
    if newz > 118: newz = 118
    element = element_symbol(newz)
    config = atom_properties(element, 'Config')
    return config[0].split('.')


def default_atom_charge(element):
    """
    Returns the default charge value for the element
    :param element: str: 'Fe'
    :return: int or nan
    """

    element = np.asarray(element).reshape(-1)
    charge = np.zeros(len(element))
    for n in range(len(element)):
        symbol, occupancy, elecharge = split_element_symbol(element[n])
        group = atom_properties(symbol, 'Group')[0]
        if elecharge != 0:
            charge[n] = elecharge
        elif group == 1:
            charge[n] = 1 * occupancy
        elif group == 2:
            charge[n] = 2 * occupancy
        elif group == 16:
            charge[n] = -2 * occupancy
        elif group == 17:
            charge[n] = -1 * occupancy
        elif group == 18:
            charge[n] = 0
        else:
            charge[n] = np.nan
    if len(charge) == 1:
        return charge[0]
    return charge


def balance_atom_charge(list_of_elements, occupancy=None):
    """
    Determine the default charges and assign remaining charge to
    unspecified elements
    :param list_of_elements: list of element symbols ['Co', 'Fe', ...]
    :param occupancy: None or list of occupancies, same length as above
    :return: [list of charges]
    """

    if occupancy is None:
        occupancy = np.ones(len(list_of_elements))
    else:
        occupancy = np.asarray(occupancy)

    charge = np.zeros(len(list_of_elements))
    for n in range(len(list_of_elements)):
        _, occ, _ = split_element_symbol(list_of_elements[n])
        charge[n] = occupancy[n] * default_atom_charge(list_of_elements[n])
        occupancy[n] = occupancy[n] * occ

    remaining_charge = -np.nansum(charge)
    uncharged = np.sum(occupancy[np.isnan(charge)])
    for n in range(len(list_of_elements)):
        if np.isnan(charge[n]):
            charge[n] = occupancy[n] * remaining_charge / uncharged
    return charge


def arrange_atom_order(list_of_elements):
    """
    Arrange a list of elements in the correct chemical order
      ['Li','Co','O'] = arrange_atom_order(['Co', 'O', 'Li'])
    :param list_of_elements: [list of str]
    :return: [list of str]
    """

    list_of_elements = np.asarray(list_of_elements)
    group = atom_properties(list_of_elements, 'Group')
    idx = np.argsort(group)
    return list(list_of_elements[idx])


def count_atoms(list_of_elements, occupancy=None, divideby=1, latex=False):
    """
    Count atoms in a list of elements, returning condenced list of elements
    :param list_of_elements: list of element symbols
    :param occupancy: list of occupancies of each element (default=None)
    :param divideby: divide each count by this (default=1)
    :param latex: False*/True
    :return: [list of str]
    """

    if occupancy is None:
        occupancy = np.ones(len(list_of_elements))

    # Count elements
    ats = np.unique(list_of_elements)
    ats = arrange_atom_order(ats)  # arrange this by alcali/TM/O
    outstr = []
    for a in ats:
        atno = sum([occupancy[n] for n, x in enumerate(list_of_elements) if x == a]) / float(divideby)
        if np.abs(atno - 1) < 0.01:
            atstr = ''
        else:
            atstr = '%0.2g' % atno
            if latex:
                atstr = '$_{%s}$' % atstr

        outstr += ['%s%s' % (a, atstr)]
    return outstr


def count_charges(list_of_elements, occupancy=None, divideby=1, latex=False):
    """
    Count atoms in a list of elements, returning condenced list of elements with charges
    :param list_of_elements: list of element symbols
    :param occupancy: list of occupancies of each element (default=None)
    :param divideby: divide each count by this (default=1)
    :param latex: False*/True
    :return: [list of str]
    """

    if occupancy is None:
        occupancy = np.ones(len(list_of_elements))

    # Determine element charges
    charge = balance_atom_charge(list_of_elements, occupancy)

    # Count elements
    ats = np.unique(list_of_elements)
    ats = arrange_atom_order(ats)  # arrange this by alcali/TM/O
    atno = {}
    chno = {}
    for a in ats:
        atno[a] = sum([occupancy[n] for n, x in enumerate(list_of_elements) if x == a]) / float(divideby)
        chno[a] = sum([charge[n] for n, x in enumerate(list_of_elements) if x == a]) / (atno[a] * float(divideby))

    outstr = []
    for a in ats:
        outstr += [element_charge_string(a, atno[a], chno[a], latex)]
    return outstr


def hunds_rule(state):
    """
    Determine S,L,J numbers using Hunds Rules
     S, L, J = Hunds('3d3')
    :param state: str orbital state nl
    :returns S, L, J ints
    """
    n = int(state[0])
    if 's' in state:
        l = 0
    elif 'p' in state:
        l = 1
    elif 'd' in state:
        l = 2
    elif 'f' in state:
        l = 3
    else:
        raise ValueError('%s is not a known state' % state)

    nelec = int(state[2:])
    norbits = 2 * l + 1
    orbits = 2 * list(range(l, -l - 1, -1))
    orb_spin = np.zeros(2 * norbits)

    for op in range(nelec):
        if op >= norbits:
            orb_spin[op] = -1
        else:
            orb_spin[op] = 1

    S = 0.5 * np.sum(orb_spin)
    L = np.sum(np.abs(orb_spin) * (orbits))

    if nelec > norbits:
        # More than half-filled
        J = np.abs(L + S)
    else:
        # Less than half filled
        J = np.abs(L - S)

    return S, L, J


def glande(S, L, J):
    """
    Calculate the Lande g-value
        https://en.wikipedia.org/wiki/Land%C3%A9_g-factor
    :param S: int Spin value
    :param L: int orbital
    :param J: int L + S value
    :return: float G
    """

    S = np.asarray(S, dtype=float)
    L = np.asarray(L, dtype=float)
    J = np.asarray(J, dtype=float)
    J[abs(J) < 0.01] = 0.01
    gj = 1.5 + (S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
    if np.isnan(gj):
        gj = 0.0
    return gj


def atom_valence_state(element, charge=0):
    """
    Very simple way of determining the default neutral valence state of a transition metal
    :param element: str element symbole
    :param charge: float charge state (assumes this is a nominal charge state)
    :return: str e.g. '3d7'
    """
    orbitals = orbital_configuration(element, charge)
    dstates = [o for o in orbitals if 'd' in o or 'f' in o]
    if len(dstates) > 0:
        return dstates[-1]
    return orbitals[-1]


def molecular_weight(compound_name):
    """
    Calculate the molecular weight of given compound
    :param compound_name: str elements
    :return: float weight in g
    """
    el_list = split_compound(compound_name)
    weight = 0
    for el in el_list:
        name, occ, charge = split_element_symbol(el)
        weight += occ * atom_properties(name, 'Weight')[0]
    return weight


def xray_attenuation_length(elements, energy_kev, atom_per_volume, grazing_angle=90):
    """
    Calcualte the attenuation length in microns
    The depth into the material measured along the surface normal where the intensity of
    x-rays falls to 1/e of its value at the surface.
      A = sin(th) / n * mu
    :param elements: str or list of str, if list - absorption will be summed over elements
    :param energy_kev: float array
    :param atom_per_volume: float atoms per A^3
    :param grazing_angle: incidence angle relative to the surface, in degrees
    :return: float or array in microns
    """
    mu = photoabsorption_crosssection(elements, energy_kev)
    surface_normal = np.sin( np.deg2rad(grazing_angle))
    return surface_normal * 1e6 / (1e30 * atom_per_volume * mu)


def xray_transmission(elements, energy_kev, atom_per_volume, distance_um):
    """
    Calculate the transimssion of x-rays through a thick slab
      T/T0 = exp(-n*mu*d)
    :param elements: str or list of str, if list - absorption will be summed over elements
    :param energy_kev: float array
    :param atom_per_volume: float atoms per A^3
    :param distance_um: float distance in microns
    :return: float or array
    """
    mu = photoabsorption_crosssection(elements, energy_kev)  # float, or [ene]
    return np.exp(-atom_per_volume * 1e30 * mu * distance_um * 1e-6)


def xray_refractive_index(elements, energy_kev, atom_per_volume):
    """
    Calculate the complex index of refraction of a material
      n = 1 - (1/2pi)N*r0*lambda^2*(f1+if2) = 1 - Delta - iBeta
    :param elements: str or list of str, if list atomic scattering factors will be summed over elements
    :param energy_kev: float array
    :param atom_per_volume: float atoms per A^3
    :return: complex float or array
    """
    f1, f2 = atomic_scattering_factor(elements, energy_kev)
    ft = f1 + 1j * f2
    ft = ft.reshape(np.size(elements), np.size(energy_kev)).sum(axis=0)  # sum over elements
    atom_per_volume = atom_per_volume * 1e30  # atom per m^3
    wavelength = energy2wave(energy_kev) * 1e-10  # m
    return 1 - (ft * fg.r0 * atom_per_volume * wavelength**2 / (2 * fg.pi))


def xray_reflectivity(elements, energy_kev, atom_per_volume, grazing_angle):
    """
    Calculate the specular reflectivity of a material
    From: https://xdb.lbl.gov/Section4/Sec_4-2.html
    :param elements: str or list of str, if list - absorption will be summed over elements
    :param energy_kev: float array
    :param atom_per_volume: float atoms per A^3
    :param grazing_angle: incidence angle relative to the surface, in degrees
    :return: float or array
    """
    wavelength = energy2wave(energy_kev)
    refindex = xray_refractive_index(elements, energy_kev, atom_per_volume)
    pilambda = 2 * fg.pi / wavelength
    angle_rad = np.deg2rad(grazing_angle)
    ki = pilambda * np.sin(angle_rad)
    kt = pilambda * np.sqrt(refindex**2 - np.cos(angle_rad)**2)
    r = (ki - kt)/(ki + kt)
    return np.real(np.multiply(r, np.conj(r)))


def xray_harmonic_rejection(elements, energy_kev, atom_per_volume, grazing_angle, harmonic=2):
    """
    Calculate the specular reflectivity of a material
    From: https://xdb.lbl.gov/Section4/Sec_4-2.html
    :param elements: str or list of str, if list - absorption will be summed over elements
    :param energy_kev: float array
    :param atom_per_volume: float atoms per A^3
    :param grazing_angle: incidence angle relative to the surface, in degrees
    :param harmonic: int harmonic multiple
    :return: float or array
    """
    primary_r = xray_reflectivity(elements, energy_kev, atom_per_volume, grazing_angle)
    harmonic_r = xray_reflectivity(elements, np.multiply(energy_kev, harmonic), atom_per_volume, grazing_angle)
    return np.divide(harmonic_r, primary_r)


def molecular_attenuation_length(chemical_formula, energy_kev, density, grazing_angle=90):
    """
    Calcualte X-Ray Attenuation Length
    Equivalent to: https://henke.lbl.gov/optical_constants/atten2.html
    Based on formulas from: Henke, Gullikson, and Davis, Atomic Data and Nuclear Data Tables 54 no.2, 181-342 (July 1993)
    :param chemical_formula: str molecular formula
    :param energy_kev: float or array, x-ray energy in keV
    :param density: float density in g/cm^3
    :param grazing_angle: incidence angle relative to the surface, in degrees
    :return: float or array, in microns
    """
    elements = split_compound(chemical_formula)
    weight = molecular_weight(chemical_formula)
    atom_per_volume = 1e-24 * density * fg.Na / weight  # atoms per A^3
    return xray_attenuation_length(elements, energy_kev, atom_per_volume, grazing_angle)


def molecular_refractive_index(chemical_formula, energy_kev, density):
    """
    Calculate Complex Index of Refraction
        n = 1 - (1/2pi)N*r0*lambda^2*(f1+if2) = 1 - Delta - iBeta
    Equivalent to: https://henke.lbl.gov/optical_constants/getdb2.html
    Based on formulas from: Henke, Gullikson, and Davis, Atomic Data and Nuclear Data Tables 54 no.2, 181-342 (July 1993)
    :param chemical_formula: str molecular formula
    :param energy_kev: float or array, x-ray energy in keV
    :param density: float density in g/cm^3
    :return: n(complex), Delta, Beta
    """
    elements = split_compound(chemical_formula)
    weight = molecular_weight(chemical_formula)
    atom_per_volume = 1e-24 * density * fg.Na / weight  # atoms per A^3
    n = xray_refractive_index(elements, energy_kev, atom_per_volume)
    delta = 1 - np.real(n)
    beta = -np.imag(n)
    return n, delta, beta


def molecular_reflectivity(chemical_formula, energy_kev, density, grazing_angle):
    """
    Calculate the specular reflectivity of a material
    From: https://xdb.lbl.gov/Section4/Sec_4-2.html
    :param chemical_formula: str molecular formula
    :param energy_kev: float or array, x-ray energy in keV
    :param density: float, density in g/cm^3
    :param grazing_angle: float, incidence angle relative to the surface, in degrees
    :return: float or array
    """
    elements = split_compound(chemical_formula)
    weight = molecular_weight(chemical_formula)
    atom_per_volume = 1e-24 * density * fg.Na / weight  # atoms per A^3
    return xray_reflectivity(elements, energy_kev, atom_per_volume, grazing_angle)


def filter_transmission(chemical_formula, energy_kev, density, thickness_um=100):
    """
    Calculate transmission of x-ray through a slab of material
    Equivalent to https://henke.lbl.gov/optical_constants/filter2.html
    Based on formulas from: Henke, Gullikson, and Davis, Atomic Data and Nuclear Data Tables 54 no.2, 181-342 (July 1993)
    :param chemical_formula: str molecular formula
    :param energy_kev: float or array, x-ray energy in keV
    :param density: float density in g/cm^3
    :param thickness_um: slab thickness in microns
    :return: float or array
    """
    elements = split_compound(chemical_formula)
    weight = molecular_weight(chemical_formula)
    atom_per_volume = 1e-24 * density * fg.Na / weight  # atoms per A^3
    return xray_transmission(elements, energy_kev, atom_per_volume, thickness_um)


'-------------------Lattice Transformations------------------------------'


gen_lattice_parameters = fl.gen_lattice_parameters


def latpar2uv(*lattice_parameters, **kwargs):
    """
    Convert a,b,c,alpha,beta,gamma to UV=[A,B,C]
     UV = latpar2uv(a,b,c,alpha=90.,beta=90.,gamma=120.)
     Vector c is defined along [0,0,1]
     Vector a and b are defined by the angles
    """
    return fl.basis_1(*lattice_parameters, **kwargs)


def latpar2uv_rot(*lattice_parameters, **kwargs):
    """
    Convert a,b,c,alpha,beta,gamma to UV=[A,B,C]
     UV = latpar2uv_rot(a,b,c,alpha=90.,beta=90.,gamma=120.)
     Vector b is defined along [0,1,0]
     Vector a and c are defined by the angles
    """
    return fl.basis_3(*lattice_parameters, **kwargs)


def UV2latpar(UV):
    """
    Convert UV=[a,b,c] to a,b,c,alpha,beta,gamma
     a,b,c,alpha,beta,gamma = UV2latpar(UV)
    """
    return fl.basis2latpar(UV)


def RcSp(UV):
    """
    Generate reciprocal cell from real space unit vecors
    Usage:
    UVs = RcSp(UV)
      UV = [[3x3]] matrix of vectors [a,b,c]
    """

    # b1 = 2*np.pi*np.cross(UV[1],UV[2])/np.dot(UV[0],np.cross(UV[1],UV[2]))
    # b2 = 2*np.pi*np.cross(UV[2],UV[0])/np.dot(UV[0],np.cross(UV[1],UV[2]))
    # b3 = 2*np.pi*np.cross(UV[0],UV[1])/np.dot(UV[0],np.cross(UV[1],UV[2]))
    # UVs = np.array([b1,b2,b3])

    # UVs = 2 * np.pi * np.linalg.inv(UV).T
    return fl.reciprocal_basis(UV)


def indx(coords, basis_vectors):
    """
    Index cartesian coordinates on a lattice defined by basis vectors
    Usage (reciprocal space):
        [[h, k, l], ...] = index_lattice([[qx, qy, qz], ...], [a*, b*, c*])
    Usage (direct space):
        [u, v, w] = index_lattice([x, y, z], [a, b, c])

    :param coords: [nx3] array of coordinates
    :param basis_vectors: [3*3] array of basis vectors [a[3], b[3], c[3]]
    :return: [nx3] array of vectors in units of reciprocal lattice vectors
    """
    return fl.index_lattice(coords, basis_vectors)


def wavevector_difference(Q, ki):
    """
    Returns the difference between reciprocal lattice coordinates and incident wavevector, in A-1
    When the difference is zero, the condition for diffraction is met.
      Laue condition: Q = ha* + kb* + lc* == kf - ki
      Elastic scattering:            |kf| = |ki|
      Therefore:          |Q + ki| - |ki| = 0
      Expanding & simplifing: Q.Q + 2Q.ki = 0
    :param Q: [[nx3]] array of reciprocal lattice coordinates, in A-1
    :param ki: [x,y,z] incident wavevector, in A-1
    :return: [nx1] array of difference, in A-1
    """
    return np.abs(fg.mag(Q + ki) - fg.mag(ki))
    # Q = np.reshape(Q, [-1, 3])
    # return np.array([np.dot(q, q) + 2 * np.dot(q, ki) for q in Q])


def Bmatrix(basis_vectors):
    """
    Calculate the Busing and Levy B matrix from real space basis vectors, with units of 2pi
    "choose the x-axis parallel to a*, the y-axis in the plane of a* and b*, and the z-axis perpendicular to that plane"
    From: W. R. Busing and H. A. Levy, Acta Cryst. (1967). 22, 457-464
    "Angle calculations for 3- and 4-circle X-ray and neutron diffractometers"
    See also: https://docs.mantidproject.org/nightly/concepts/Lattice.html

    B = [[b1, b2 * cos(beta3), b3 * cos(beta2)],
        [0, b2 * sin(beta3), -b3 * sin(beta2) * cos(alpha1)],
        [0, 0, 1 / a3]]
    return 2pi * B  # equivalent to transpose([a*, b*, c*])

    :param basis_vectors: [3*3] array of basis vectors [a[3], b[3], c[3]]
    :return: [3*3] B matrix * 2 pi
    """
    return fl.basis2bandl(basis_vectors)


def umatrix(a_axis=None, b_axis=None, c_axis=None):
    """
    Define an orientation matrix in the diffractometer frame
    Diffractometer frame according to Fig. 1, H. You, J. Appl. Cryst 32 (1999), 614-623
      z-axis : axis parallel to the phi rotation axis when all angles 0 (towards wall (+x) in lab frame)
      x-axis : vector normal to phi axis where phi=0 (toward ceiling (+y) in lab frame)
      y-axis : vector normal to x,z axes (parallel to beam (+z) in lab frame)
    :param a_axis: direction of a in the diffractometer frame
    :param b_axis: direction of b in the diffractometer frame
    :param c_axis: direction of c in the diffractometer frame
    :return: [3*3] array
    """
    a_axis = fg.norm(np.cross(b_axis, c_axis)) if a_axis is None else fg.norm(a_axis)
    b_axis = fg.norm(np.cross(c_axis, a_axis)) if b_axis is None else fg.norm(b_axis)
    c_axis = fg.norm(np.cross(a_axis, b_axis)) if c_axis is None else fg.norm(c_axis)
    #return np.array([a_axis, b_axis, c_axis], dtype=float)
    if np.abs(np.dot(c_axis, a_axis)) > 0.99:
        raise Exception('Axes must not be parallel')
    return fg.normal_basis(c_axis, a_axis)


def ubmatrix(uv, u):
    """
    Return UB matrix (units of 2pi)
    :param uv: [3*3] unit vector [a,b,c]
    :param u: [3*3] orientation matrix in the diffractometer frame
    :return: [3*3] array
    """
    b = Bmatrix(uv)
    return np.dot(u, b)


def rotmatrixz(phi):
    """
    Generate diffractometer rotation matrix about z-axis (right handed)
    Equivalent to YAW in the Tait-Bryan convention
    Equivalent to -phi, -eta, -delta in You et al. diffractometer convention (left handed)
        r = rotmatrix_z(phi)
        vec' = np.dot(r, vec)
    vec must be 1D or column vector (3*n)
    :param phi: float angle in degrees
    :return: [3*3] array
    """
    phi = np.deg2rad(phi)
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # wiki - right handed


def rotmatrixy(chi):
    """
    Generate diffractometer rotation matrix chi about y-axis (right handed)
    Equivalent to PITCH in the Tait–Bryan convention
    Equivalent to -chi in You et al. diffractometer convention (left handed)
        r = rotmatrix_y(chi)
        vec' = np.dot(r, vec)
    vec must be 1D or column vector (3*n)
    :param chi: float angle in degrees
    :return: [3*3] array
    """
    chi = np.deg2rad(chi)
    c = np.cos(chi)
    s = np.sin(chi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotmatrixx(mu):
    """
    Generate diffractometer rotation matrix mu about x-axis (right handed)
    Equivalent to ROLL in the Tait–Bryan convention
    Equivalent to mu in You et al. diffractometer convention (right handed)
        r = rotmatrix_x(mu)
        vec' = np.dot(r, vec)
    vec must be 1D or column vector (3*n)
    :param mu: float angle in degrees
    :return: [3*3] array
    """
    mu = np.deg2rad(mu)
    c = np.cos(mu)
    s = np.sin(mu)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def diffractometer_rotation(phi=0, chi=0, eta=0, mu=0):
    """
    Generate the 6-axis diffracometer rotation matrix
      R = M * E * X * P
    Also called Z in H. You, J. Appl. Cryst 32 (1999), 614-623
    The diffractometer coordinate system has the convention (all angles zero):
        x-axis points vertically, perpendicular to beam (mu is about x)
        y-axis points along the direction of the beam
        z-axis points along the phi axis, perpendicular to x and y
    The vertical scattering plane is in the y-x axis
    The horizontal scattering plane is in the y-z axis
       vec' = np.dot(diffractometer_rotation(phi, chi, eta, mu), vec)
       vec must be 1D or column vector (3*n)
    :param phi: float angle in degrees, left handed roation about z'''
    :param chi: float angle in degrees, right handed rotation about y''
    :param eta: float angle in degrees, left handed rotation about z'
    :param mu: float angle in degrees, right handed rotation about x
    :return:  [3*3] array
    """
    P = rotmatrixz(-phi)  # left handed
    X = rotmatrixy(chi)
    E = rotmatrixz(-eta)  # left handed
    M = rotmatrixx(mu)
    return np.dot(M, np.dot(E, np.dot(X, P)))


def diff2lab(vec, lab=None):
    """
    Convert between diffractometer frame and lab frame
    Lab frame according to Diamond I16 beamline
    Diffractometer frame according to Fig. 1, H. You, J. Appl. Cryst 32 (1999), 614-623
      z-axis : axis parallel to the phi rotation axis when all angles 0 (towards wall (+x) in lab frame)
      x-axis : vector normal to phi axis where phi=0 (toward ceiling (+y) in lab frame)
      y-axis : vector normal to x,z axes (parallel to beam (+z) in lab frame)
    :param vec: [3*n] array of vectors
    :param lab: [3*3] transformation matrix, None=((0,1,0),(0,0,1),(1,0,0))
    :return: [3*n] array of vectors
    """
    if lab is None:
        lab = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # (x_lab || z_diff, y_lab || x_diff, z_lab || y_diff)
    return np.dot(lab, np.transpose(vec)).T


def labvector(vec, U=None, R=None, LAB=None):
    """
    Transform any vector through the orientation, rotation and lab transformations
    :param vec: [n*3] array of vectors in the diffractometer frame
    :param U: [3*3] oritenation matrix (see umatrix)
    :param R: [3x3] rotation matrix (see diffractometer_rotation)
    :param LAB: [3x3] transformation matrix between diffractometer frame and lab frame
    :return: [n*3] array of Q vectors in the lab coordinate system
    """
    if U is None:
        U = np.eye(3)
    if R is None:
        R = np.eye(3)
    if LAB is None:
        LAB = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # (x_lab || z_diff, y_lab || x_diff, z_lab || y_diff)
    return np.dot(LAB, np.dot(R, np.dot(U, np.transpose(vec)))).T


def labwavevector(hkl, UV, U=None, R=None, LAB=None):
    """
    Calculate the lab wavevector using the unit-vector, oritenation matrix and rotation matrix
    Returns vectors in the lab coordinate system, by default defined like Diamond Light Source:
      x-axis : away from synchrotron ring, towards wall
      y-axis : towards ceiling
      z-axis : along beam direction
    :param hkl: [3xn] array of (h, k, l) reciprocal lattice vectors
    :param UV: [3*3] Unit-vector matrix (see latpar2ub_rot)
    :param U: [3*3] oritenation matrix (see umatrix)
    :param R: [3x3] rotation matrix (see diffractometer_rotation)
    :param LAB: [3x3] transformation matrix between diffractometer frame and lab frame
    :return: [3xn] array of Q vectors in the lab coordinate system
    """
    if U is None:
        U = np.eye(3)
    if R is None:
        R = np.eye(3)
    if LAB is None:
        LAB = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # (x_lab || z_diff, y_lab || x_diff, z_lab || y_diff)
    B = Bmatrix(UV)
    return np.dot(LAB, np.dot(R, np.dot(U, np.dot(B, np.transpose(hkl))))).T


def diff6circleq(delta, gamma, energy_kev=None, wavelength=1.0, lab=None):
    """
    Calcualte wavevector in diffractometer axis using detector angles
    :param delta: float angle in degrees in vertical direction (about diff-z)
    :param gamma: float angle in degrees in horizontal direction (about diff-x)
    :param energy_kev: float energy in KeV
    :param wavelength: float wavelength in A
    :param lab: [3*3] lab transformation matrix
    :return: [1*3]
    """
    if energy_kev is not None:
        wavelength = energy2wave(energy_kev)

    k = 2 * np.pi / wavelength
    delta = np.deg2rad(delta)
    gamma = np.deg2rad(gamma)
    sd = np.sin(delta)
    cd = np.cos(delta)
    sg = np.sin(gamma)
    cg = np.cos(gamma)
    return diff2lab(k * np.array([sd, cd * cg - 1, cd * sg]), lab)


def diff6circlek(delta, gamma, energy_kev=None, wavelength=1.0, lab=None):
    """
    Calcualte incident and final wavevectors in diffractometer axis using detector angles
    :param delta: float angle in degrees in vertical direction (about diff-z)
    :param gamma: float angle in degrees in horizontal direction (about diff-x)
    :param energy_kev: float energy in KeV
    :param wavelength: float wavelength in A
    :param lab: [3*3] lab transformation matrix
    :return: [1*3], [1*3] : ki, kf
    """
    if energy_kev is not None:
        wavelength = energy2wave(energy_kev)

    k = 2 * np.pi / wavelength
    # q = kf - ki
    q = diff6circleq(delta, gamma, energy_kev, wavelength, lab)
    ki = k * diff2lab([0, 1, 0], lab)
    kf = q + ki
    return ki, kf


def diff6circle2hkl(ub, phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0, energy_kev=None, wavelength=1.0, lab=None):
    """
    Return [h,k,l] position of diffractometer axes with given UB and energy
    :param ub: [3*3] array UB orientation matrix following Busing & Levy
    :param phi: float sample angle in degrees
    :param chi: float sample angle in degrees
    :param eta: float sample angle in degrees
    :param mu: float sample angle in degrees
    :param delta: float detector angle in degrees
    :param gamma: float detector angle in degrees
    :param energy_kev: float energy in KeV
    :param wavelength: float wavelength in A
    :param lab: [3*3] lab transformation matrix
    :return: [h,k,l]
    """
    q = diff6circleq(delta, gamma, energy_kev, wavelength, lab)  # You Ql (12)
    z = diffractometer_rotation(phi, chi, eta, mu)  # You Z (13)

    inv_ub = np.linalg.inv(ub)
    inv_z = np.linalg.inv(z)

    hphi = np.dot(inv_z, q)
    return np.dot(inv_ub, hphi).T


def maxHKL(Qmax, UV):
    """
    Returns the maximum indexes for given max radius
    e.g.
        UVstar = RcSp([[3,0,0],[0,3,0],[0,0,10]]) # A^-1
        Qmax = 4.5 # A^-1
        max_hkl = maxHKL(Qmax,UVstar)
        max_hkl =
        >>> [3,3,8]
    """

    Qpos = [[Qmax, Qmax, Qmax],
            [-Qmax, Qmax, Qmax],
            [Qmax, -Qmax, Qmax],
            [-Qmax, -Qmax, Qmax],
            [Qmax, Qmax, -Qmax],
            [-Qmax, Qmax, -Qmax],
            [Qmax, -Qmax, -Qmax],
            [-Qmax, -Qmax, -Qmax]]
    hkl = indx(Qpos, UV)
    return np.ceil(np.abs(hkl).max(axis=0)).astype(int)


def genHKL(H, K=None, L=None):
    """
    Generate HKL array with all combinations within range specified
    Usage:
      HKL = genHKL(max)
       int max = generates each h,k,l from -max to max
      HKL = genHKL([min,max])
       int min, int max = generates h,l,l from min to max
      HKL = genHKL([hmin,hmax],[kmin,kmax],[lmin,lmax])
       hmin,hmax = min and max values for h, k and l respectivley

      array HKL = [[nx3]] array of [h,k,l] vectors

    E.G.
      HKL = genHKL(-3)
    E.G.
      HKL = genHKL([-1,1])
    E.G.
      HKL = genHKL([-2,2],[0,3],[1,1])
    """

    if K is None: K = H
    if L is None: L = H

    H = np.asarray(H)
    K = np.asarray(K)
    L = np.asarray(L)

    if H.size == 1:
        H = np.array([H, -H])
    if K.size == 1:
        K = np.array([K, -K])
    if L.size == 1:
        L = np.array([L, -L])

    if H[0] > H[1]:
        Hstep = -1
    else:
        Hstep = 1
    if K[0] > K[1]:
        Kstep = -1
    else:
        Kstep = 1.0
    if L[0] > L[1]:
        Lstep = -1.0
    else:
        Lstep = 1.0

    Hrange = np.arange(H[0], H[1] + Hstep, Hstep)
    Krange = np.arange(K[0], K[1] + Kstep, Kstep)
    Lrange = np.arange(L[0], L[1] + Lstep, Hstep)
    KK, HH, LL = np.meshgrid(Krange, Hrange, Lrange)

    return np.asarray([HH.ravel(), KK.ravel(), LL.ravel()], dtype=int).T


def fitincell(uvw):
    """
    Set all fractional coodinates between 0 and 1
    Usage:
      uvw = fitincell(uvw)
      uvw = [[nx3]] array of fractional vectors [u,v,w]
    """
    while np.any(uvw > 0.99) or np.any(uvw < -0.01):
        uvw[uvw > 0.99] = uvw[uvw > 0.99] - 1.
        uvw[uvw < -0.01] = uvw[uvw < -0.01] + 1.
    return uvw


'---------------------------Symmetry-------------------------------------'


def gen_sym_pos(sym_ops, x, y, z):
    """
    Generate positions from symmetry operations
    Usage:
      uvw = gen_sym_pos(sym_ops,x,y,z)
      sym_ops = [n*'x,y,z'] array of string symmetry operations
      x,y,z = fractional coordinates of atomic posiiton to be modified by symmetry
      uvw = [[nx3]] array of symmetry defined factional coordinates [u,v,w]

    E.G.
      uvw = gen_sym_pos(['x,y,z','y,-x,z+1/2'],0.1,0.2,0.3)
      uvw >> [[0.1,0.2,0.3] , [0.2,-0.1,0.8]]
    """
    uvw = np.zeros([len(sym_ops), 3])
    for n in range(len(sym_ops)):
        sym = sym_ops[n]
        sym = sym.lower()
        # Evaluate string symmetry operation in terms of x,y,z
        sym = sym.replace('/', './')
        sym = sym.strip('\"\'')
        sym = sym.replace('mx', 'x').replace('my', 'y').replace('mz', 'z')
        sym = sym.replace('2x', '2*x')  # 2x appears in some hexagonal spacegroups
        out = eval(sym, {'x': x, 'y': y, 'z': z})
        uvw[n] = np.array(out[0:3]) + 0.0  # add zero to remove -0.0 values
    return uvw


def gen_symcen_pos(sym_ops, cen_ops, x, y, z):
    """
    Generate positions from symmetry and centring operations
    Usage:
      uvw = gen_symcen_pos(sym_ops,cen_ops,x,y,z)
      sym_ops = [n*'x,y,z'] array of string symmetry operations
      cen_ops = [m*'x,y,z'] array of string centring operations
      x,y,z = fractional coordinates of atomic posiiton to be modified by symmetry
      uvw = [[nx3]] array of symmetry defined factional coordinates [u,v,w]

    E.G.
      uvw = gen_symcen_pos(['x,y,z','y,-x,z+1/2'],['x,y,z','x+1/3,y+1/3,z'],0.1,0.2,0.3)
      uvw >> [[0.1,0.2,0.3] , [0.433,0.533,0.3] , [0.2,-0.1,0.8] , [0.533,-0.233,0.8]]
    """

    NsymOps = len(sym_ops)
    NcenOps = len(cen_ops)
    Nops = NsymOps * NcenOps

    uvw = gen_sym_pos(sym_ops, x, y, z)

    sympos = np.zeros([Nops, 3])
    for m in range(NsymOps):
        cx, cy, cz = uvw[m]
        Ni, Nj = m * NcenOps, (m + 1) * NcenOps
        cen_uvw = gen_sym_pos(cen_ops, cx, cy, cz)
        sympos[Ni:Nj, :] = cen_uvw
    return sympos


def gen_sym_unique(sym_ops, x, y, z, cen_ops=None):
    """
    Generate positions from symmetry operations with idential positions removed
    Usage:
      uvw = gen_sym_unique(sym_ops,x,y,z)
    E.G.
      uvw = gen_sym_unique(['x,y,z','y,-x,z+1/2','x,y,z'],0.1,0.2,0.3)
      uvw >> [[0.1,0.2,0.3] , [0.2,-0.1,0.8]]
    :param sym_ops: list of str - symmetry operations
    :param x: float
    :param y: float
    :param z: float
    :param cen_ops: Optional - list of str for centring operations
    :return: array of positions
    """
    if cen_ops:
        sym_xyz = gen_symcen_pos(sym_ops, cen_ops, x, y, z)
    else:
        sym_xyz = gen_sym_pos(sym_ops, x, y, z)
    sym_xyz = fitincell(sym_xyz)
    # Remove identical positions
    unique_xyz, uniqueidx, matchidx = fg.unique_vector(sym_xyz, tol=0.01)
    return unique_xyz


def gen_symcen_ops(sym_ops, cen_ops):
    """
    Build complete list of symmetry operations from symmetry and centring vectors
    Usage:
      ops = gen_symcen_ops(sym_ops,cen_ops)
      sym_ops = [n*'x,y,z'] array of string symmetry operations
      cen_ops = [m*'x,y,z'] array of string centring operations
      ops = [n*m*'x,y,z'] array of string symmetry operations

    E.G.
      ops = gen_symcen_pos(['x,y,z','y,-x,z+1/2'],['x,y,z','x+1/3,y+1/3,z'])
      >> ops = ['x,y,z','x+1/3,y+1/3,z','y,-x,z+1/2','y+1/3,-x+1/3,z+1/2']
    """

    ops = []
    for sym in sym_ops:
        sym = sym.lower()
        sym = sym.strip('\"\'')
        s_op = sym.split(',')
        x = s_op[0].strip()
        y = s_op[1].strip()
        z = s_op[2].strip()
        if len(s_op) > 3:
            t = int(s_op[3])
        else:
            t = 1
        for cen in cen_ops:
            cen = cen.lower()
            cen = cen.strip('\"\'')
            op = cen.replace('x', 'a').replace('y', 'b').replace('z', 'c')  # avoid replacing x/y twice
            op = op.replace('a', x).replace('b', y).replace('c', z)
            op = op.replace('--', '')
            if op.count(',') > 2:
                op = op.split(',')
                op[3] = '%+d' % (int(op[3]) * t)
                op = ','.join(op)
            ops += [op]
    return ops


def gen_sym_ref(sym_ops, hkl):
    """
     Generate symmetric reflections given symmetry operations
         symhkl = gen_sym_ref(sym_ops,h,k,l)
    """

    hkl = np.asarray(hkl, dtype=float)

    # Get transformation matrices
    sym_mat = gen_sym_mat(sym_ops)
    nops = len(sym_ops)

    symhkl = np.zeros([nops, 3])
    for n, sym in enumerate(sym_mat):
        # multiply only by the rotational part
        symhkl[n, :] = np.dot(hkl, sym[:3, :3])

    return symhkl


def sum_sym_ref(symhkl):
    """
    Remove equivalent hkl positions and return the number of times each is generated.
    """
    # Remove duplicate positions
    uniquehkl, uniqueidx, matchidx = fg.unique_vector(symhkl[:, :3], 0.01)

    # Sum intensity at duplicate positions
    sumint = np.zeros(len(uniquehkl))
    for n in range(len(uniquehkl)):
        sumint[n] = matchidx.count(n)

    return uniquehkl, sumint


def gen_sym_mat(sym_ops):
    """
     Generate transformation matrix from symmetry operation
     Currently very ugly but it seems to work
     Tested in Test/test_gen_sym_mat - found to be fast and reliable.
     sym_mat = gen_sym_mat(['x,y,z','y,-x,z+1/2'])
     sym_mat[0] = [[ 1.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.],
                   [ 0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  1.]])
     sym_mat[1] = [[ 0. ,  1. ,  0. ,  0. ],
                   [-1. ,  0. ,  0. ,  0. ],
                   [ 0. ,  0. ,  1. ,  0.5],
                   [ 0.,   0.,   0.,   1.]]
    """

    sym_ops = np.asarray(sym_ops, dtype=str).reshape(-1)

    sym_mat = []
    for sym in sym_ops:
        sym = sym.lower()
        sym = sym.strip('\"\'')
        sym = sym.replace('/', './')  # float division
        ops = sym.split(',')
        mat = np.zeros((4, 4))
        mat[3, 3] = 1

        for n in range(3):
            op = ops[n]
            if 'x' in op: mat[n, 0] = 1
            if '-x' in op: mat[n, 0] = -1
            if 'y' in op: mat[n, 1] = 1
            if '-y' in op: mat[n, 1] = -1
            if 'z' in op: mat[n, 2] = 1
            if '-z' in op: mat[n, 2] = -1

            # remove these parts
            op = op.replace('-x', '').replace('x', '')
            op = op.replace('-y', '').replace('y', '')
            op = op.replace('-z', '').replace('z', '')
            op = op.replace('+', '')

            if len(op.strip()) > 0:
                mat[n, 3] = eval(op)
        sym_mat += [mat]
    return sym_mat


def sym_mat2str(sym_mat, time=None):
    """
    Generate symmetry operation string from matrix
    :param sym_mat: array [3x3] or [4x4]
    :param time: +/-1 or None
    :return: str 'x,y,z(,1)'
    """
    sym_mat = np.asarray(sym_mat)

    rot = sym_mat[:3, :3]
    if sym_mat.shape[1] == 4:
        trans = sym_mat[:, 3]
    else:
        trans = np.zeros(3)

    denominators = range(2, 8)
    out = []
    for n in range(3):
        # Convert rotational part
        xyz = '%1.3gx+%1.3gy+%1.3gz' % (rot[n][0], rot[n][1], rot[n][2])
        xyz = re.sub('[+-]?0[xyz]', '', xyz).replace('1', '').replace('+-', '-').strip('+')

        # Convert translational part
        if abs(trans[n]) < 0.01:
            add = ''
        else:
            chk = [(d * trans[n]) % 1 < 0.01 for d in denominators]
            if any(chk):
                denom = denominators[chk.index(True)]
                add = '+%1.0f/%1.0f' % (denom * trans[n], denom)
            else:
                add = '+%1.4g' % trans[n]
            add = add.replace('+-', '+')
        # print(n, rot[n], trans[n], xyz, add)
        out += [xyz + add]
    if time is not None:
        out += ['%+1.3g' % time]
    return ','.join(out)


def sym_op_recogniser(sym_ops):
    """
    Evaluates symmetry operations and returns str name of operation

    Doesn't work yet - works on simple systems but not trigonal or hexagonal operations - for example sg167
    """
    print('Doesnt work yet!!!')
    sym_ops = np.asarray(sym_ops, dtype=str).reshape(-1)
    sym_mats = gen_sym_mat(sym_ops)
    DEBUG = True
    out = []
    for op, m in zip(sym_ops, sym_mats):
        ns = op.count('-')  # 0=translation, 1=mirror, 2=rotation, 3=inversion
        parity = np.linalg.det(m[:3, :3])  # determinant of operation - +/-1
        translation = np.sum(np.abs(m[:3, 3]))
        msum = np.sum(m[:3, :3])
        asum = np.sum(np.abs(m[:3, :3]))
        if DEBUG:
            s = "{20s}: ns={}, parity={}, translation={}, sum={}, abssum={}"
            print(s.format(os, ns, parity, translation, msum, asum))
        if ns == 0:  # translation
            if translation < 0.01:
                out += ['1']
            else:
                out += ['t(%.2g,%.2g,%.2g)' % (m[3, 0], m[3, 1], m[3, 2])]
        elif ns == 3:  # inversion
            out += ['-1']
        elif parity > 0:  # rotation
            if translation < 0.01:
                out += ['Rotation']
            else:
                out += ['Screw']
        elif parity < 1:  # mirror
            if translation < 0.01:
                out += ['Mirror']
            else:
                out += ['Glide']
        else:
            out += ['Unknown']
    return out


def sym_op_det(sym_ops):
    """
    Return the determinant of a symmetry operation
    :param sym_ops: str e.g. 'x,-y,z+1/2' or 'y, x+y, -z, -1' or list of str ['x,y,z',...]
    :return: float |det| or list of floats
    """
    mat = gen_sym_mat(sym_ops)
    if len(mat) == 1:
        return np.linalg.det(mat[0][:3, :3])
    return [np.linalg.det(m[:3, :3]) for m in mat]


def gen_sym_axial_vector(sym_ops, x, y, z):
    """
     Transform axial vector by symmetry operations
    Usage:
      uvw = gen_symcen_pos(sym_ops,cen_ops,x,y,z)
      sym_ops = [n*'x,y,z'] array of string symmetry operations
      cen_ops = [m*'x,y,z'] array of string centring operations
      x,y,z = fractional coordinates of atomic posiiton to be modified by symmetry
      uvw = [[nx3]] array of symmetry defined factional coordinates [u,v,w]

    E.G.
      uvw = gen_symcen_pos(['x,y,z','y,-x,z+1/2'],['x,y,z','x+1/3,y+1/3,z'],0.1,0.2,0.3)
      uvw >> [[0.1,0.2,0.3] , [0.433,0.533,0.3] , [0.2,-0.1,0.8] , [0.533,-0.233,0.8]]
    :param sym_ops:
    :param x:
    :param y:
    :param z:
    :return:
    """
    mat = gen_sym_mat(sym_ops)
    return [np.linalg.det(m[:3, :3]) * np.dot(m[:3, :3], (x, y, z)) for m in mat]


def invert_sym(sym_op):
    """
    Invert the sign of the given symmetry operation
    Usage:
      new_op = invert_sym(sym_op)
      sym_op = str symmetry operation e.g. 'x,y,z'
      new_op = inverted symmetry
    E.G.
      new_op = invert_sym('x,y,z')
      >> new_op = '-x,-y,-z'
    """
    sym_op = sym_op.lower()
    new_op = sym_op.replace('x', '-x').replace('y', '-y').replace('z', '-z').replace('--', '+').replace('+-', '-')
    return new_op


def sym_op_time(operations):
    """
    Return the time symmetry of a symmetry operation
    :param operations: list(str) e.g. ['x,-y,z+1/2', 'y, x+y, -z, -1']
    :return: list, +/-1
    """
    operations = np.asarray(operations).reshape(-1)
    t_op = np.empty(len(operations))
    for n, sym_op in enumerate(operations):
        ops = sym_op.split(',')
        if len(ops) < 4:
            t_op[n] = 1
        else:
            t_op[n] = int(ops[3])
    return list(t_op)


def sym_op_mx(operations):
    """
    Convert each operation from x,y,z to mx,my,mz
    :param operations: ['x,y,z',]
    :return: ['mx,my,mz',]
    """
    operations = list(np.asarray(operations).reshape(-1))
    for n in range(len(operations)):
        if 'mx' in operations[n]: continue
        operations[n] = operations[n].replace('x', 'mx').replace('y', 'my').replace('z', 'mz')
    return operations


def symmetry_ops2magnetic(operations):
    """
    Convert list of string symmetry operations to magnetic symmetry operations
    See Vesta_Manual.pdf Section 9.1.1 "Creation and editing of a vector"
    Magnetic symmetry
        µ' = TPMµ
    T = Time operators x,y,z,(+1)
    P = Parity operator (determinant of M)
    M = Symmetry operator without translations

    :param operations: list of str ['x,y,z',]
    :return: list of str ['x,y,z']
    """
    operations = np.asarray(operations).reshape(-1)
    # Convert operations to matrices
    mat_ops = gen_sym_mat(operations)
    tim_ops = sym_op_time(operations)
    str_ops = []
    for n, mat in enumerate(mat_ops):
        # Get time operation
        t = tim_ops[n]

        # Only use rotational part
        m = mat[:3, :3]

        # Get parity
        p = np.linalg.det(m)

        # Generate string
        mag_str = sym_mat2str(t * p * m)
        # mag_str = mag_str.replace('x', 'mx').replace('y', 'my').replace('z', 'mz')
        str_ops += [mag_str]
    return str_ops


def orthogonal_axes(x_axis=(1, 0, 0), y_axis=(0, 1, 0)):
    """
    Returns orthogonal right-handed cartesian axes based on the plane of two non-perpendicular vectors
    E.G.
        x,y,z = orthogonal_axes([1,0,0],[0.5,0.5,0])
        >> x = array([1,0,0])
        >> y = array([0,1,0])
        >> z = array([0,0,1])
    E.G.
        x,y,z = orthogonal_axes([0,1,0],[0,0,1])
        >> x = array([0,1,0])
        >> y = array([0,0,1])
        >> z = array([1,0,0])
    """

    x_cart = fg.norm(x_axis)
    y_cart = fg.norm(y_axis)
    z_cart = fg.norm(np.cross(x_cart, y_cart))  # z is perp. to x+y
    y_cart = np.cross(x_cart, z_cart)  # make sure y is perp. to x
    return x_cart, y_cart, z_cart


'----------------------------Conversions-------------------------------'


def hkl2Q(hkl, UVstar):
    """
    Convert reflection indices (hkl) to orthogonal basis in A-1
    :param hkl: [nx3] array of reflections
    :param UV: [3x3] array of unit cell vectors [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
    :return: [nx3] array of wavevectors in an orthogonal basis, in A-1
    """
    return np.dot(hkl, UVstar)


def Q2hkl(qvec, UVstar):
    """
    Index vectors in an orthonal basis with a reciprocal basis
    :param qvec: [nx3] array of coordinates in an orthogonal basis in A-1
    :param UV: [3x3] array of unit cell vectors [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
    :return: [nx3] array of vectors in reciprocal lattice units
    """
    return fg.index_coordinates(qvec, UVstar)


def hkl2Qmag(hkl, UVstar):
    """
    Calcualte magnitude of Q from hkl reflection
    :param hkl: [nx3] array of reflections
    :param UV: [3x3] array of unit cell vectors
    :return: [nx1] array of wavevector magnitudes, in A-1
    """
    Q = np.dot(hkl, UVstar)
    return fg.mag(Q)


def hkl2twotheta(hkl, UVstar, energy_kev=17.794, wavelength_a=None):
    """
    Calcualte d-spacing from hkl reflection
    :param hkl: [nx3] array of reflections
    :param UV: [3x3] array of unit cell vectors
    :param energy_kev: float energy in keV
    :param wavelength_a: float wavelength in Angstroms
    :return: [nx1] array of d-spacing in A
    """
    q = np.dot(hkl, UVstar)
    q_mag = fg.mag(q)
    return cal2theta(q_mag, energy_kev, wavelength_a)


def hkl2dspace(hkl, UVstar):
    """
    Calcualte d-spacing from hkl reflection
    :param hkl: [nx3] array of reflections
    :param UV: [3x3] array of reciprocal unit cell vectors
    :return: [nx1] array of d-spacing in A
    """
    Q = np.dot(hkl, UVstar)
    Qmag = fg.mag(Q)
    return q2dspace(Qmag)


def lattice_hkl2dspace(hkl, *lattice_parameters, **kwargs):
    """
    Calcualte dspace from lattice parameters
    :param hkl: [nx3] array of reflections
    :param lattice_parameters: a,b,c,alpha,beta,gamma
    :return: float, d-spacing in A
    """
    return fl.dspacing(*hkl, *lattice_parameters, **kwargs)


def lattice_hkl2twotheta(hkl, *lattice_parameters, energy_kev=17.794, wavelength_a=None, **kwargs):
    """
    Calcualte dspace from lattice parameters
    :param hkl: [nx3] array of reflections
    :param energy_kev: float, radiation energy in keV
    :param lattice_parameters: a,b,c,alpha,beta,gamma
    :return: float, d-spacing in A
    """
    dspace = fl.dspacing(*hkl, *lattice_parameters, **kwargs)
    qmag = dspace2q(dspace)
    return cal2theta(qmag, energy_kev=energy_kev, wavelength_a=wavelength_a)
    return hkl2twotheta(hkl, uvs, energy_kev)


def calqmag(twotheta, energy_kev=17.794, wavelength_a=None):
    """
    Calculate |Q| at a particular 2-theta (deg) for energy in keV
      magQ = calqmag(twotheta, energy_kev=17.794)
       - equivalent to -
      qmag = 4 * pi * sin(theta) / wl

    :param twotheta: float or array of scattering angles, in degrees
    :param energy_kev: float photon energy in keV
    :param wavelength_a: float wavelength in Anstrom
    :return wavevector magnitude in inverse-Angstrom
    """
    if wavelength_a is None:
        wavelength_a = energy2wave(energy_kev)  # wavelength form photon energy
    # energy = energy_kev * 1000.0  # energy in eV
    theta = twotheta * np.pi / 360  # theta in radians
    # Calculate |Q|
    # magq = 4pi sin(theta) / lambda
    # magq = np.sin(theta) * energy * fg.e * 4 * np.pi / (fg.h * fg.c * 1e10)
    magq = np.sin(theta) * 4 * np.pi / wavelength_a
    return magq


def cal2theta(qmag, energy_kev=17.794, wavelength_a=None):
    """
    Calculate scattering angle at particular energy in keV from magnitude of the wavevector, |Q|
     twotheta = cal2theta(Qmag, energy_kev=17.794)
     - equivalent to -
     twotheta = 2 * arcsin( qmag * wl / 4pi )

    :param qmag: float or array of wavevector magnitudes, in inverse-Angstroms
    :param energy_kev: float photon energy in keV
    :param wavelength_a: float wavelength in Anstrom
    :return two-theta angle in degrees (or array if qmag is array)
    """
    if wavelength_a is None:
        wavelength_a = energy2wave(energy_kev)  # wavelength form photon energy
    # energy = energy_kev * 1000.0  # energy in eV
    # Calculate 2theta angles for x-rays
    # lambda = 2d sin(theta) = 4pi sin(theta) / qmag
    # theta = arcsin(qmag * lambda / 4pi)
    # twotheta = 2 * np.arcsin(qmag * 1e10 * fg.h * fg.c / (energy * fg.e * 4 * np.pi))
    twotheta = 2 * np.arcsin(qmag * wavelength_a / (4 * np.pi))
    # return x2T in degrees
    twotheta = twotheta * 180 / np.pi
    return twotheta


def caldspace(twotheta, energy_kev=17.794, wavelength_a=None):
    """
    Calculate d-spacing from two-theta
      dspace = caldspace(tth, energy_kev)
       - equivalent to -
      dspace = wl / (2 * sin(theta))

    :param twotheta: float or array of scattering angles, in degrees
    :param energy_kev: float photon energy in keV
    :param wavelength_a: float wavelength in Anstrom
    :return lattice d-spacing in Anstrom
    """
    qmag = calqmag(twotheta, energy_kev, wavelength_a)
    dspace = q2dspace(qmag)
    return dspace


def callattice(twotheta, energy_kev=17.794, hkl=(1, 0, 0)):
    """
    Calculate cubic lattice parameter, a from reflection two-theta
    :param twotheta: Bragg angle, deg
    :param energy_kev: energy in keV
    :param hkl: reflection (cubic only
    :return: float, lattice contant
    """
    qmag = calqmag(twotheta, energy_kev)
    dspace = q2dspace(qmag)
    return dspace * np.sqrt(np.sum(np.square(hkl)))


def q2dspace(qmag):
    """
    Calculate d-spacing from |Q|
         dspace = q2dspace(Qmag)
    """
    return 2 * np.pi / qmag


def dspace2q(dspace):
    """
    Calculate d-spacing from |Q|
         Qmag = q2dspace(dspace)
    """
    return 2 * np.pi / dspace


def q2units(qmag, units='tth', energy_kev=None):
    """
    Convert |Q| in A^-1 to choice of units
    :param qmag: float or array in inverse-Angstrom
    :param units: str: 'tth', 'dspace', 'q' (raises ValueError if wrong)
    :param energy_kev: float or None
    :return: float or array
    """
    units = units.lower().replace(' ', '').replace('-', '')
    if 'q' in units:
        return qmag
    if units in ['d', 'dspace', 'dspacing', 'angstrom', 'A']:
        return q2dspace(qmag)
    if units in ['tth', 'angle', 'twotheta', 'theta', 'deg', 'degrees']:
        return cal2theta(qmag, energy_kev)
    raise ValueError('%s is not a valid unit' % units)


def resolution2energy(res, twotheta=180.):
    """
    Calcualte the energy required to achieve a specific resolution at a given two-theta
    :param res: measurement resolution in A (==d-spacing)
    :param twotheta: Bragg angle in Degrees
    :return: float
    """
    theta = twotheta * np.pi / 360  # theta in radians
    return (fg.h * fg.c * 1e10) / (res * np.sin(theta) * fg.e * 2 * 1000.)


def wave2energy(wavelength):
    """
    Converts wavelength in A to energy in keV
     energy_kev = wave2energy(wavelength_a)
     Energy [keV] = h*c/L = 12.3984 / lambda [A]
    """

    # SI: E = hc/lambda
    lam = wavelength * fg.A
    E = fg.h * fg.c / lam

    # Electron Volts:
    Energy = E / fg.e
    return Energy / 1000.0


def energy2wave(energy_kev):
    """
    Converts energy in keV to wavelength in A
     wavelength_a = energy2wave(energy_kev)
     lambda [A] = h*c/E = 12.3984 / E [keV]
    """

    # Electron Volts:
    E = 1000 * energy_kev * fg.e

    # SI: E = hc/lambda
    lam = fg.h * fg.c / E
    wavelength = lam / fg.A
    return wavelength


def wavevector(energy_kev=None, wavelength=None):
    """Return wavevector = 2pi/lambda"""
    if wavelength is None:
        wavelength = energy2wave(energy_kev)
    return 2 * np.pi / wavelength


def neutron_wavelength(energy_mev):
    """
    Calcualte the neutron wavelength in Angstroms using DeBroglie's formula
      lambda [A] ~ sqrt( 81.82 / E [meV] )
    :param energy_mev: neutron energy in meV
    :return: wavelength in Anstroms
    """
    return fg.h / np.sqrt(2 * fg.mn * energy_mev * fg.e / 1000) / fg.A


def neutron_energy(wavelength_a):
    """
    Calcualte the neutron energy in milli-electronvolts using DeBroglie's formula
      E [meV] ~ 81.82 / lambda^2 [A]
    :param wavelength_a: neutron wavelength in Angstroms
    :return: energy in meV
    """
    return fg.h**2 / (2 * fg.mn * (wavelength_a*fg.A)**2 * fg.e / 1000)


def electron_wavelength(energy_ev):
    """
    Calcualte the electron wavelength in Angstroms using DeBroglie's formula
      lambda [nm] ~ sqrt( 1.5 / E [eV] )
    :param energy_ev: electron energy in eV
    :return: wavelength in Anstroms
    """
    return fg.h / np.sqrt(2 * fg.me * energy_ev * fg.e) / fg.A


def electron_energy(wavelength_a):
    """
    Calcualte the electron energy in electronvolts using DeBroglie's formula
      E [eV] ~ 1.5 / lambda^2 [nm]
    :param wavelength_a: electron wavelength in Angstroms
    :return: energy in eV
    """
    return fg.h**2 / (2 * fg.me * (wavelength_a * fg.A)**2 * fg.e)


def debroglie_wavelength(energy_kev, mass_kg):
    """
    Calcualte the wavelength in Angstroms using DeBroglie's formula
      lambda [A] = h  / sqrt( 2 * mass [kg] * E [keV] * 1e3 * e )
    :param energy_kev: energy in keV
    :param mass_kg: mass in kg
    :return: wavelength in Anstroms
    """
    return fg.h / (np.sqrt(2 * mass_kg * energy_kev * 1000 * fg.e) * fg.A)


def debroglie_energy(wavelength_a, mass_kg):
    """
    Calcualte the energy in electronvolts using DeBroglie's formula
      E [keV] = h^2 / (2 * e * mass [kg] * A^2 * lambda^2 [A] * 1e3)
    :param wavelength_a: wavelength in Angstroms
    :param mass_kg: mass in kg
    :return: energy in keV
    """
    return fg.h ** 2 / (2 * mass_kg * (wavelength_a * fg.A) ** 2 * fg.e * 1e3)


def scherrer_size(fwhm, twotheta, wavelength_a=None, energy_kev=None, shape_factor=0.9):
    """
    Use the Scherrer equation to calculate the size of a crystallite from a peak width
      L = K * lambda / fwhm * cos(theta)
    See: https://en.wikipedia.org/wiki/Scherrer_equation
    :param fwhm: full-width-half-maximum of a peak, in degrees
    :param twotheta: 2*Bragg angle, in degrees
    :param wavelength_a: incident beam wavelength, in Angstroms
    :param energy_kev: or, incident beam energy, in keV
    :param shape_factor: dimensionless shape factor, dependent on shape of crystallite
    :return: float, crystallite domain size in Angstroms
    """
    if wavelength_a is None:
        wavelength_a = energy2wave(energy_kev)
    delta_theta = np.deg2rad(fwhm)
    costheta = np.cos(np.deg2rad(twotheta / 2.))
    return shape_factor * wavelength_a / (delta_theta * costheta)


def scherrer_fwhm(size, twotheta, wavelength_a=None, energy_kev=None, shape_factor=0.9):
    """
    Use the Scherrer equation to calculate the size of a crystallite from a peak width
      L = K * lambda / fwhm * cos(theta)
    See: https://en.wikipedia.org/wiki/Scherrer_equation
    :param size: crystallite domain size in Angstroms
    :param twotheta: 2*Bragg angle, in degrees
    :param wavelength_a: incident beam wavelength, in Angstroms
    :param energy_kev: or, incident beam energy, in keV
    :param shape_factor: dimensionless shape factor, dependent on shape of crystallite
    :return: float, peak full-width-at-half-max in degrees
    """
    if wavelength_a is None:
        wavelength_a = energy2wave(energy_kev)
    costheta = np.cos(np.deg2rad(twotheta / 2.))
    return np.rad2deg(shape_factor * wavelength_a / (size * costheta))


def peakwidth_deg(domain_size_a, twotheta, wavelength_a=None, energy_kev=None, instrument_resolution=0):
    """
    Return the peak width in degrees for a given two-theta based on the
    crystallite domain size and instrument resolution.
      Equivalent to the Scherrer equation with shape factor 1.0
    :param domain_size_a: crystallite domain size in Anstroms
    :param twotheta: scattering angle in degrees
    :param wavelength_a: incident beam wavelength, in Angstroms
    :param energy_kev: or, incident beam energy, in keV
    :param instrument_resolution: instrument resolution in inverse Anstroms
    :return:
    """
    peak_width = np.sqrt(dspace2q(domain_size_a)**2 + instrument_resolution**2)
    q = calqmag(twotheta, energy_kev, wavelength_a)
    return cal2theta(q + peak_width, energy_kev, wavelength_a) - twotheta


def biso2uiso(biso):
    """
    Convert B isotropic thermal parameters to U isotropic thermal parameters
    :param biso: Biso value or array
    :return: Uiso value or array
    """
    biso = np.asarray(biso, dtype=float)
    return biso / (8 * np.pi ** 2)


def uiso2biso(uiso):
    """
    Convert U isotropic thermal parameters to B isotropic thermal parameters
    :param uiso: Uiso value or array
    :return: Biso value or array
    """
    uiso = np.asarray(uiso, dtype=float)
    return uiso * (8 * np.pi ** 2)


def euler_unit_vector(uvw, uv):
    """
    Convert vector in a specific basis to a cartesian basis and normalise to a unit vector
    :param uvw: [nx3] array as [[u,v,w]]
    :param uv: [3x3], basis vectors [a,b,c]
    :return: [nx3] array xyz/|xyz|, where x,y,z = u*a+v*b+w*c
    """
    xyz = np.dot(uvw, uv)
    return fg.norm(xyz)


def euler_moment(mxmymz, uv):
    """
    Convert moment mxmymz coordinates from cif into eulerian basis
    :param mxmymz: [nx3] array, units of Bohr magneton, directed along a,b,c
    :param uv: [3x3] array, basis vectors [a,b,c]
    :return: moments [nx3] array, units of Bohr magneton, directed along x,y,z
    """
    # Calculate moment
    momentmag = fg.mag(mxmymz).reshape([-1, 1])
    momentxyz = np.dot(mxmymz, uv)
    moment = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
    moment[np.isnan(moment)] = 0.
    return moment


def diffractometer_Q(eta, delta, energy_kev=8.0):
    """
    Calculate wavevector transfer, Q for diffractometer within the scattering plane.
       Qx,Qy = diffractometer_Q(eta,delta,energy_kev)
       eta = angle (deg) of sample
       delta = angle (deg) of detector
       energy_kev = energy in keV

       Coordinate system (right handed):
        x - direction along beam, +ve is from source to sample
        y - direction verticle
        z - direction horizontal

       Note: Currently only in the verticle geometry!
    """

    delta = np.radians(delta)
    eta = np.radians(eta)
    K = 1E-10 * 1000 * 2 * np.pi * (energy_kev * fg.e) / (fg.h * fg.c)  # K vector magnitude

    Qx = K * (np.cos(eta) - np.cos(delta - eta))
    Qy = K * (np.sin(delta - eta) + np.sin(eta))
    return Qx, Qy


def hkl2str(hkl):
    """
    Convert hkl to string (h,k,l)
    :param hkl:
    :return: str '(h,k,l)'
    """

    out = '(%1.3g,%1.3g,%1.3g)'
    hkl = np.asarray(hkl, dtype=float).reshape([-1, 3])
    hkl = np.around(hkl, 5)
    return '\n'.join([out % (x[0], x[1], x[2]) for x in hkl])


def cut2powder(qx, qy, qz, cut):
    """
    Convert 2D reciprocal space cut to powder pattern
    :param qx: [n,m] array of Q coordinates in x
    :param qy: [n,m] array of Q coordinates in y
    :param qz: [n,m] array or float of Q coordinates in z
    :param cut: [n,m] array of intensities
    :return: qm[o], pow[o]
    """
    qq = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    return fg.grid_intensity(qq.ravel(), cut.ravel())


'--------------------------Misc Crystal Programs------------------------'


def str2element(string):
    """Finds element name in string"""
    result = element_regex.findall(string)
    if len(result) == 0:
        return False
    return result[0].capitalize()


def debyewaller(uiso, Qmag=0):
    """
    Calculate the debye waller factor for a particular Q
     T = debyewaller(uiso,Qmag=[0])

        T = exp( -2*pi^2*Uiso/d^2 )
        T = exp( -Uiso/2 * Q^2 )
    """

    uiso = np.asarray(uiso, dtype=float).reshape(1, -1)
    Qmag = np.asarray(Qmag, dtype=float).reshape(-1, 1)

    # Tall = np.exp(-0.5 * np.dot(Qmag, uiso))  # Not sure where this comes from... Jon's notes?
    Tall = np.exp(-0.5 * np.dot(Qmag ** 2, uiso))
    return Tall


def powder_average(tth, energy_kev):
    """
    Return the powder average correction for intensities at a given two-theta
        Ip = I0*PA,    PA = 1/|Q|^2
    :param tth: two-theta in deg
    :param energy_kev: energy in keV
    :return: PA
    """
    q = calqmag(tth, energy_kev)
    return 1 / q ** 2


def group_intensities(q_values, intensity, min_overlap=0.01):
    """
    Group reflections within the overlap, returning the index max intensity of each group
    :param q_values: [1*n] array of floats, position parameter of each reflection
    :param intensity: [1*n] array of floats, intensity parameter of each reflection
    :param min_overlap: float, how close the reflections are to be grouped
    :return: group_idx: [1*m] array of int index
    """

    groups, array_index, group_index, counts = fg.group(q_values, min_overlap)

    # loop over groups and select reflection with largest intensity
    ref_n = np.zeros(len(groups), dtype=int)
    for n in range(len(groups)):
        args = np.where(array_index == n)[0]
        # find max intensity
        ref_n[n] = args[np.argmax(intensity[args])]
    return ref_n


def calc_vol(UV):
    """Calculate volume in Angstrom^3 from unit vectors"""
    a, b, c = UV
    return np.abs(np.dot(a, np.cross(b, c)))


def cif2table(cif):
    """
    Generate Latex table of atomic positions from cif dictionary
    :param cif: cif dict from readcif
    :return: str
    """

    keys = cif.keys()

    # Generate unit vectors
    a = cif['_cell_length_a']
    b = cif['_cell_length_b']
    c = cif['_cell_length_c']
    alpha = cif['_cell_angle_alpha']
    beta = cif['_cell_angle_beta']
    gamma = cif['_cell_angle_gamma']
    lp = r'a=%s\AA, b=%s\AA, c=%s\AA, $\alpha$=%s, $\beta$=%s, $\gamma$=%s' % (a, b, c, alpha, beta, gamma)

    # Get atom names & labels
    label = cif['_atom_site_label']

    # Thermal parameters
    if '_atom_site_U_iso_or_equiv' in keys:
        u_or_b = 'U$_{iso}$'
        uiso = cif['_atom_site_U_iso_or_equiv']
    elif '_atom_site_B_iso_or_equiv' in keys:
        u_or_b = 'B$_{iso}$'
        uiso = cif['_atom_site_B_iso_or_equiv']
    else:
        u_or_b = 'U$_{iso}$'
        uiso = ['0.000'] * len(label)
    # Occupancy
    if '_atom_site_occupancy' in keys:
        occ = cif['_atom_site_occupancy']
    else:
        occ = ['1.0'] * len(label)
    # Multiplicity
    if '_atom_site_site_symmetry_multiplicity' in keys:
        mult = [s + 'a' for s in cif['_atom_site_site_symmetry_multiplicity']]
    else:
        mult = ['1a'] * len(label)

    # Get coordinates
    u = cif['_atom_site_fract_x']
    v = cif['_atom_site_fract_y']
    w = cif['_atom_site_fract_z']

    # Get space group
    if '_symmetry_space_group_name_H-M' in keys:
        spacegroup = cif['_symmetry_space_group_name_H-M']
    elif '_space_group_name_H-M_alt' in keys:
        spacegroup = cif['_space_group_name_H-M_alt']
    else:
        spacegroup = ''
    if '_symmetry_Int_Tables_number' in keys:
        sgn = float(cif['_symmetry_Int_Tables_number'])
    elif '_space_group_IT_number' in keys:
        sgn = float(cif['_space_group_IT_number'])
    elif spacegroup == 'P1':
        sgn = 1
    else:
        sgn = 0

    # Aditional details:
    if 'FileTitle' in keys:
        name = cif['FileTitle']
    else:
        name = 'sample'

    extra = []
    if '_cell_measurement_temperature' in keys:
        extra += ['T = %s' % cif['_cell_measurement_temperature']]
    if '_diffrn_reflns_number' in keys:
        extra += ['%s reflections' % cif['_diffrn_reflns_number']]
    if '_refine_ls_wR_factor_ref' in keys:
        extra += ['R$_w$ = %4.2f\\%%' % (float(cif['_refine_ls_wR_factor_ref']) * 100)]
    extra = ', '.join(extra)

    # Create table str
    out = '%% %s\n' % cif['Filename']
    out += '\\begin{table}[htp]\n'
    out += '    \\centering\n'
    out += '       \\begin{tabular}{c|c|ccccc}\n'
    out += r'             & Site & x & y & z & Occ. & %s \\\\ \hline\n' % u_or_b
    fmt = '        %4s & %5s & %s & %s & %s & %s & %s \\\\\n'
    for n in range(len(label)):
        out += fmt % (label[n], mult[n], u[n], v[n], w[n], occ[n], uiso[n])
    out += '        \\end{tabular}\n'
    out += '    \\caption{%s with %s. %s}\n' % (name, lp, extra)
    out += '    \\label{tab:}\n'
    out += '\\end{table}\n'
    return out


def detector_rotate(detector_position_mm=(0, 1000., 0), delta=0., gamma=0., labmatrix=np.eye(3)):
    """
    Return a position array for a vector rotated by delta and gamma, like a detector along [0,1,0]
    :param detector_position_mm: detector position from sample in mm
    :param delta: vertical rotation about z-axis in Deg
    :param gamma: horizontal rotation about x-axis in Deg
    :param labmatrix: [3*3] orientation matrix to convert to alternative basis
    :return: array
    """
    det_position = np.array(detector_position_mm)
    D = rotmatrixz(-delta)  # left handed
    G = rotmatrixx(gamma)
    R = np.dot(G, D)
    return labvector(det_position, R=R, LAB=labmatrix)


def reflections_on_detector(qxqyqz, energy_kev, det_distance=1., det_normal=(0, -1., 0),
                            delta=0., gamma=0., det_width=1., det_height=1., labmatrix=np.eye(3)):
    """
    Return relative position of reflection on detector
    :param qxqyqz: [nx3] array of reflection coordinates in Q-space [qx, qy, qz]
    :param energy_kev: float, Incident beam energy in keV
    :param det_distance: float, Detector distance in meters
    :param det_normal: (dx,dy,dz) direction of detector normal
    :param delta: float, angle in degrees in vertical direction (about diff-z)
    :param gamma: float angle in degrees in horizontal direction (about diff-x)
    :param det_width: float, detector width along x-axis in meters
    :param det_height: float, detector height along z-axis in meters
    :param labmatrix: [3x3] orientation array to convert to difference basis
    :return uvw: [nx3] array of positions relative to the centre of the detector, or NaN if not incident
    :return wavevector_difference: [n] array of wavevector difference in A-1
    """

    # Lattice
    q = np.reshape(qxqyqz, [-1, 3])
    k = wavevector(energy_kev)
    ki = k * labvector([0, 1, 0], LAB=labmatrix)
    kf = q + ki
    kf_directions = fg.norm(kf)
    wv_difference = wavevector_difference(q, ki)  # difference in A-1

    # Detector basis
    det_position = det_distance * np.array([0, 1., 0])
    det_normal = fg.norm(det_normal)
    det_x_axis = det_width * fg.norm(np.cross(det_normal, (0, 0, 1.)))
    det_z_axis = det_height * fg.norm(np.cross(det_x_axis, det_normal))

    # Detector rotation
    D = rotmatrixz(-delta)  # left handed
    G = rotmatrixx(gamma)
    R = np.dot(G, D)
    det_position = labvector(det_position, R=R, LAB=labmatrix)
    det_normal = labvector(det_normal, R=R, LAB=labmatrix)
    det_x_axis = labvector(det_x_axis, R=R, LAB=labmatrix)
    det_z_axis = labvector(det_z_axis, R=R, LAB=labmatrix)

    # Detector corners
    det_corners = np.array([
        det_position + det_x_axis / 2 + det_z_axis / 2,
        det_position + det_x_axis / 2 - det_z_axis / 2,
        det_position - det_x_axis / 2 - det_z_axis / 2,
        det_position - det_x_axis / 2 + det_z_axis / 2,
    ])

    # Reflection direction near detector
    corner_angle = max(abs(fg.vectors_angle_degrees(det_position, det_corners)))
    ref_angles = abs(fg.vectors_angle_degrees(det_position, kf_directions))
    check = ref_angles < corner_angle

    # Reflection direction incident on detector
    incident_xyz = np.nan * np.zeros([len(kf_directions), 3])
    for n in np.flatnonzero(check):
        incident_xyz[n] = fg.plane_intersection((0, 0, 0), kf_directions[n], det_position, det_normal)

    # Relative position on detector
    incident_uvw = fg.index_coordinates(incident_xyz - det_position, [det_x_axis, det_z_axis, det_normal])
    incident_uvw[np.any(abs(incident_uvw) > 0.5, axis=1)] = [np.nan, np.nan, np.nan]
    return incident_uvw, wv_difference


def peaks_on_plane(peak_x, peak_y, peak_height, peak_width, max_x, max_y, pixels_width=1001, background=0):
    """
    Creates a rectangular grid and adds Gaussian peaks with height "intensity"
    :param peak_x: [nx1] array of x coordinates
    :param peak_y: [nx1] array of y coordinates
    :param peak_height: [nx1] array of peak heights
    :param peak_width: [nx1] or float, gaussian width
    :param max_x: grid will be created from -max_x : +max_x horizontally
    :param max_y: grid will be created from -max_y : +max_y vertically
    :param pixels_width: grid will contain pixels horizontally and the number vertically will be scaled
    :param background: if >0, add a normaly distributed background with average level = background
    :return: x, y, plane
    """
    # create plotting mesh
    pixels_height = int(pixels_width * max_y / max_x)
    mesh = np.zeros([pixels_height, pixels_width])
    mesh_x = np.linspace(-max_x, max_x, pixels_width)
    mesh_y = np.linspace(-max_y, max_y, pixels_height)
    xx, yy = np.meshgrid(mesh_x, mesh_y)
    # cast float peak_width to array
    peak_width = np.asarray(peak_width) * np.ones_like(peak_height)

    for n in range(len(peak_height)):
        # Add each reflection as a gaussian
        mesh += peak_height[n] * np.exp(-np.log(2) * (((xx - peak_x[n]) ** 2 + (yy - peak_y[n]) ** 2) / (peak_width[n] / 2) ** 2))

    # Add background (if not None or 0)
    if background:
        bkg = np.random.normal(background, np.sqrt(background), mesh.shape)
        mesh = mesh + bkg

    return xx, yy, mesh


def scale_intensity(intensity, wavevector_diff, resolution=0.5):
    """
    Scale intensity depending on wavevector difference and resolution
    :param intensity: intensity value
    :param wavevector_diff: distance from centre in inverse angstroms
    :param resolution: resolution in inverse angstroms
    :return:
    """
    return intensity * np.exp(-np.log(2) * (wavevector_diff / (0.5 * resolution)) ** 2)


def wavevector_resolution(energy_range_ev=200, wavelength_range_a=None, domain_size_a=1000):
    """
    Calculate the combined wavevector resolutuion in inverse Angstroms
    Combines instrument incident beam resolution and the mosaic of the crystal
    :param energy_range_ev: Incident x-ray beam resolution in eV
    :param wavelength_range_a: Incident beam resolution in Angstroms
    :param domain_size_a: crysallite size in Angstroms (provides mosic using Scherrer equation)
    :return: float, resolutuion in inverse Angstroms
    """
    if wavelength_range_a is None:
        beam_resolution = wavevector(energy_kev=energy_range_ev / 1000)
    else:
        beam_resolution = dspace2q(wavelength_range_a)
    sample_resolution = dspace2q(domain_size_a)
    return np.sqrt(sample_resolution**2 + beam_resolution**2)


def reciprocal_volume(twotheta, twotheta_min=0, energy_kev=None, wavelength_a=1.5):
    """
    Calculate the volume of reciprocal space covered between two scattering angles
    :param twotheta: larger scattering angle (Bragg angle, 2theta)
    :param twotheta_min: smaller scattering angle
    :param energy_kev: photon energy, in keV
    :param wavelength_a: wavelength in Angstroms
    :return: float, volume, in Angstroms^-3
    """
    q1 = calqmag(twotheta_min, energy_kev, wavelength_a)
    q2 = calqmag(twotheta, energy_kev, wavelength_a)
    vol1 = 4 * fg.pi * q1 ** 3 / 3
    vol2 = 4 * fg.pi * q2 ** 3 / 3
    return vol2 - vol1


def detector_angle_size(detector_distance_mm=565, height_mm=67, width_mm=34, pixel_size_um=172):
    """
    Return the angular spread of the detector
      delta_width, gamma_width, pixel_width = detector_angle_size(distance, height, widht, pixel)
    :param detector_distance_mm: Detector distance from sample in mm
    :param height_mm: Detector size in the vertical direction, in mm
    :param width_mm: Detector size in the horizontal direction, in mm
    :param pixel_size_um: Pixel size in microns
    :return x_angle: total angular spread in the diffractometer x-axis, in degrees
    :return z_angle: total angular spread in the diffractometer z-axis, in degrees
    :return pixel_angle: angular spread per pixel, in degrees
    """
    gam_width = 2 * np.rad2deg(np.arctan2(width_mm / 2, detector_distance_mm))
    del_height = 2 * np.rad2deg(np.arctan2(height_mm / 2, detector_distance_mm))
    pix_width = 2 * np.rad2deg(np.arctan2(pixel_size_um / 2, 1000 * detector_distance_mm))
    return gam_width, del_height, pix_width


def detector_coverage(wavelength_a=1.5, resolution=0.1, delta=0, gamma=0,
                      detector_distance_mm=565, height_mm=67, width_mm=34):
    """
    Determine volume of reciprocal space covered by the detector based on angular spread
    :param wavelength_a: Incicent beam wavelength, in Angstroms
    :param resolution: Incident beam resolution, in inverse-Angstroms
    :param delta: float or array, detector vertical rotation
    :param gamma: float or array, detector horizontal rotation
    :param detector_distance_mm: Detector distance from sample in mm
    :param height_mm: Detector size in the vertical direction, in mm
    :param width_mm: Detector size in the horizontal direction, in mm
    :return: volume in inverse-Angstroms
    """
    gam_width = np.rad2deg(np.arctan2(width_mm / 2, detector_distance_mm))
    del_height = np.rad2deg(np.arctan2(height_mm / 2, detector_distance_mm))
    gam_wid_a = 2 * dspace2q(scherrer_size(gam_width, gamma, wavelength_a=wavelength_a, shape_factor=1.0))
    del_wid_a = 2 * dspace2q(scherrer_size(del_height, delta, wavelength_a=wavelength_a, shape_factor=1.0))
    det_wid = 2 * resolution
    return gam_wid_a * del_wid_a * det_wid


def detector_volume(wavelength_a=1.5, resolution=0.1,
                    phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0,
                    detector_distance_mm=565, height_mm=67, width_mm=34, lab=np.eye(3)):
    """
    Determine volume of reciprocal space covered by the detector, return list of voxels covered by detector
      total_volume, hkl = detector_volume(1.5, 0.1, chi=90, eta=30, delta=60)
    The returned list of coordinates (hkl) is defined in the sample frame, where each coordinate specifies a voxel
    in reciprocal space with size resolution(inverse-Anstroms) ** 3
    As such, the total volume of reciprocal space measured is:
        total_volume = len(hkl) * fc.wavevector(energy_kev=energy_range_ev / 1000.) ** 3

    :param wavelength_a: Incicent beam wavelength, in Angstroms
    :param resolution: Incident beam resolution, in inverse-Angstroms (determines segmentation of calculation)
    :param phi: float, phi-axis rotation about sample-z axis
    :param chi: float, chi-axis rotation about sample-y' axis
    :param eta: float, eta-axis rotation about sample-z'' axis
    :param mu: float, mu-axis rotation about sample-x''' axis
    :param delta: float, detector rotation vertical rotation
    :param gamma: float, detector rotation horizontal rotation
    :param detector_distance_mm: Detector distance from sample in mm
    :param height_mm: Detector size in the vertical direction, in mm
    :param width_mm: Detector size in the horizontal direction, in mm
    :param lab: [3x3] Transformation matrix to change the lab coordinate system
    :return total_volume: float, total volume of reciprocal space in inverse Angstroms
    :return hkl: [nx3] array of voxels covered by detector, integer coordinates in the sample frame
    """
    # Rotate detector
    D = rotmatrixz(-delta)  # left handed
    G = rotmatrixx(gamma)
    Rdet = np.dot(G, D)
    position_mm = detector_distance_mm * labvector([0, 1., 0], R=Rdet, LAB=lab)
    normal_dir = labvector([0, -1, 0], R=Rdet, LAB=lab)
    x_axis = width_mm * fg.norm(np.cross(normal_dir, (0, 0, 1.)))
    z_axis = height_mm * fg.norm(np.cross(x_axis, normal_dir))
    corners = np.array([
        position_mm + x_axis / 2 + z_axis / 2,
        position_mm + x_axis / 2 - z_axis / 2,
        position_mm - x_axis / 2 - z_axis / 2,
        position_mm - x_axis / 2 + z_axis / 2,
    ])

    # Create a reciprocal lattice with basis size = resolution
    ub = resolution * np.eye(3)
    k = wavevector(wavelength=wavelength_a)
    ki = k * labvector([0, 1, 0], LAB=lab)  # incident beam

    # Determine the lattice position at each of the detector corners to determine a rough lattice
    gam_width = np.rad2deg(np.arctan2(width_mm / 2, detector_distance_mm))
    del_height = np.rad2deg(np.arctan2(height_mm / 2, detector_distance_mm))
    hkl_corners = np.array([
        diff6circle2hkl(ub, phi, chi, eta, mu, delta=delta + del_height, gamma=gamma + gam_width,
                        wavelength=wavelength_a, lab=lab),
        diff6circle2hkl(ub, phi, chi, eta, mu, delta=delta - del_height, gamma=gamma + gam_width,
                        wavelength=wavelength_a, lab=lab),
        diff6circle2hkl(ub, phi, chi, eta, mu, delta=delta - del_height, gamma=gamma - gam_width,
                        wavelength=wavelength_a, lab=lab),
        diff6circle2hkl(ub, phi, chi, eta, mu, delta=delta + del_height, gamma=gamma - gam_width,
                        wavelength=wavelength_a, lab=lab),
    ])
    h_range = range(int(min(hkl_corners[:, 0])) - 2, int(max(hkl_corners[:, 0])) + 2)
    k_range = range(int(min(hkl_corners[:, 1])) - 2, int(max(hkl_corners[:, 1])) + 2)
    l_range = range(int(min(hkl_corners[:, 2])) - 2, int(max(hkl_corners[:, 2])) + 2)
    HH, KK, LL = np.meshgrid(h_range, k_range, l_range)
    hkl = np.asarray([HH.ravel(), KK.ravel(), LL.ravel()]).T

    # Rotate lattice
    R = diffractometer_rotation(phi=phi, chi=chi, eta=eta, mu=mu)
    q = labvector(resolution * hkl, R=R, LAB=lab)

    # Find lattice points incident on detector
    kf = q + ki
    directions = fg.norm(kf)
    # Check magnitude of kf matches ki
    scatters = np.flatnonzero(np.abs(fg.mag(kf) - k) < resolution)
    # check which lattice directions are in the quadrent of detector
    corner_angle = min(np.dot(fg.norm(corners), fg.norm(position_mm)))
    vec_angles = np.dot(directions[scatters], fg.norm(position_mm))
    # Check with lattice point directions intercept the detector plane
    # Only loop over lattice points with correct magnitude and in the right direction
    check = vec_angles > corner_angle
    iuvw = np.nan * np.zeros([len(hkl), 3])
    for n in scatters[check]:
        ixyz = fg.plane_intersection((0, 0, 0), directions[n], position_mm, normal_dir)
        # incident positions on detector
        iuvw[n] = fg.index_coordinates(np.subtract(ixyz, position_mm), [x_axis, z_axis, normal_dir])
    iuvw[np.any(abs(iuvw) > 0.5, axis=1)] = [np.nan, np.nan, np.nan]

    # Remove non-incident reflections
    idx = ~np.isnan(iuvw[:, 0])  # iuvw same size as Q
    tot_space = np.sum(idx)
    tot_recspace = tot_space * resolution ** 3  # A-3
    # print(f"Measured Reciprocal space: {tot_space:.0f} voxels, {tot_recspace: .4g} A-3")
    return tot_recspace, hkl[idx, :]


def detector_volume_scan(wavelength_a=1.5, resolution=0.1,
                         phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0,
                         detector_distance_mm=565, height_mm=67, width_mm=34, lab=np.eye(3)):
    """
    Return total reciprocal space volume covered by scanning an axis with a detector
      total, hkl, overlaps = detector_volume_scan(1.5, 0.1, phi=np.arange(-180,180))
    Rotation axes can be specified as either floats or arrays, where all arrays must have the same length.

    :param wavelength_a: Incicent beam wavelength, in Angstroms
    :param resolution: Incident beam resolution, in inverse-Angstroms (determines segmentation of calculation)
    :param phi: float or array, phi-axis about sample-z axis
    :param chi: float or array, chi-axis about sample-y' axis
    :param eta: float or array, eta-axis about sample-z'' axis
    :param mu: float or array, mu-axis about sample-x''' axis
    :param delta: float or array, detector vertical rotation
    :param gamma: float or array, detector horizontal rotation
    :param detector_distance_mm: Detector distance from sample in mm
    :param height_mm: Detector size in the vertical direction, in mm
    :param width_mm: Detector size in the horizontal direction, in mm
    :param lab: [3x3] Transformation matrix to change the lab coordinate system
    :return total: float, total volume covered, in cubic inverse Angstroms
    :return hkl: [nx3] array of voxels covered by detector, integer coordinates in the sample frame
    :return overlaps: [n] array of the number of times each voxel has been repeatedly measured (0 = only measurred once)
    """
    # pad the scan axes
    scan_angles = [phi, chi, eta, mu, delta, gamma]
    axes = np.zeros([min([np.size(s) for s in scan_angles if np.size(s) > 1]), len(scan_angles)])
    for n in range(len(scan_angles)):
        axes[:, n] = scan_angles[n]
    # loop over the scan
    hkl_list = np.empty([0, 3], dtype=int)
    for phi, chi, eta, mu, delta, gamma in axes:
        vol, hkl = detector_volume(wavelength_a, resolution, phi, chi, eta, mu, delta, gamma,
                                   detector_distance_mm, height_mm, width_mm, lab)
        hkl_list = np.vstack([hkl_list, hkl])
    # Find the overlapping points
    hkl, counts = np.unique(hkl_list, return_counts=True, axis=0)
    overlaps = counts - 1
    total = len(hkl) * resolution ** 3
    return total, hkl, overlaps


def diffractometer_step(wavelength_a=1.5, resolution=0.1,
                        phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0, lab=np.eye(3)):
    """
    Determine the euler angle steps required to scan through reciprocal space at the required resolutuion
        phi_step, chi_step, eta_step, mu_step = diffractometer_step(1.5, 0.1, chi=90, eta=20, delta=40)
    The returned step size will provide a rotation that moves the wavevector transfer by approximately the resolutuion.
    Any steps returning 0 are unsensitive to the rotation in this position
    :param wavelength_a: Incicent beam wavelength, in Angstroms
    :param resolution: float, combined resolution required, in inverse Angstroms
    :param phi: float, phi-axis about sample-z axis
    :param chi: float, chi-axis about sample-y' axis
    :param eta: float, eta-axis about sample-z'' axis
    :param mu: float, mu-axis about sample-x''' axis
    :param delta: float, detector vertical rotation
    :param gamma: float, detector horizontal rotation
    :param lab: [3x3] Transformation matrix to change the lab coordinate system
    :returns: phi_step, chi_step, eta_step, mu_step
    """
    ub = resolution * np.eye(3)
    # Determine current location of diffractometer in orthogonal basis coordinates
    hkl = diff6circle2hkl(ub, phi=phi, chi=chi, eta=eta, mu=mu, delta=delta, gamma=gamma,
                          wavelength=wavelength_a, lab=lab)
    # Convert hkl back to wavevector units
    q0 = np.dot(ub, hkl)
    q = labvector(q0, R=diffractometer_rotation(phi=phi, chi=chi, eta=eta, mu=mu), LAB=lab)
    # Add the resolution in various directions and determine the max angular difference
    q2 = q + [
        [resolution, 0, 0],
        [0, resolution, 0],
        [0, 0, resolution],
        resolution * fg.norm([1, 1, 0]),
        resolution * fg.norm([1, 0, 1]),
        resolution * fg.norm([0, 1, 1]),
        resolution * fg.norm([1, 1, 1]),
    ]
    ang = max(np.abs(fg.vectors_angle_degrees(q, q2)))

    # Add this angle in each axis and check how much is moved
    q_phi = fg.mag(q - labvector(q0, R=diffractometer_rotation(phi=phi + ang, chi=chi, eta=eta, mu=mu), LAB=lab))
    q_chi = fg.mag(q - labvector(q0, R=diffractometer_rotation(phi=phi, chi=chi + ang, eta=eta, mu=mu), LAB=lab))
    q_eta = fg.mag(q - labvector(q0, R=diffractometer_rotation(phi=phi, chi=chi, eta=eta + ang, mu=mu), LAB=lab))
    q_mu = fg.mag(q - labvector(q0, R=diffractometer_rotation(phi=phi, chi=chi, eta=eta, mu=mu + ang), LAB=lab))

    # Scale this movement back to the resolution
    phi_ang = ang * resolution / q_phi
    chi_ang = ang * resolution / q_chi
    eta_ang = ang * resolution / q_eta
    mu_ang = ang * resolution / q_mu
    # if the movement is unrealistically large, the axis is unsensitive
    if phi_ang > 20: phi_ang = 0
    if chi_ang > 20: chi_ang = 0
    if eta_ang > 20: eta_ang = 0
    if mu_ang > 20: mu_ang = 0
    return phi_ang, chi_ang, eta_ang, mu_ang

