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

Version 3.1.1
Last updated: 26/05/20

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

Acknoledgements:
    April 2020  Thanks to ChunHai Wang for helpful suggestions in readcif!

@author: DGPorter
"""

import sys, os, re
import json
import numpy as np
from warnings import warn

from . import functions_general as fg

__version__ = '3.1.1'

# File directory - location of "Dans Element Properties.txt"
datadir = os.path.abspath(os.path.dirname(__file__))  # same directory as this file
datadir = os.path.join(datadir, 'data')
ATOMFILE = os.path.join(datadir, 'Dans Element Properties.txt')

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
    'D', 'd', # add Deuterium
]
element_regex = re.compile('|'.join(ELEMENT_LIST))

# Symmetry translations (remove these to turn symmetry operations into magnetic one)
TRANSLATIONS = [
    '+1/2', '+1/3', '+2/3', '+1/4', '+3/4', '+1/6', '+5/6', '+5/4',
    '1/2+', '1/3+', '2/3+', '1/4+', '3/4+', '1/6+', '5/6+', '5/4+',
    '-1/2', '-1/3', '-2/3', '-1/4', '-3/4', '-1/6', '-5/6', '-5/4',
    '1/2-', '1/3-', '2/3-', '1/4-', '3/4-', '1/6-', '5/6-', '5/4-',
]

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

    cifvals = {}
    cifvals['Filename'] = filename
    cifvals['Directory'] = dirName
    cifvals['FileTitle'] = fname

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
                if lines[n+1][0] == ';': n += 1
                strarg = []
                while n+1 < len(lines) and (len(lines[n+1]) == 0 or lines[n+1][0].strip() not in ['_', ';']):
                    strarg += [lines[n+1].strip('\'"')]
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
                #cols = lines[n].split()
                # this fixes error on symmetry arguments having spaces
                # this will only work if the last argument in the loop is split by spaces (in quotes)
                #cols = cols[:len(loopvals) - 1] + [''.join(cols[len(loopvals) - 1:])]
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
    UV = latpar2UV(a, b, c, alpha, beta, gamma)

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
        symtim = [np.int(s.split(',')[-1]) for s in symops_tim]  # +1
        symmag = cifvals['_space_group_symop.magn_operation_mxmymz']

        # Centring vectors also given in this case
        symcen_tim = cifvals['_space_group_symop.magn_centering_xyz']
        symcen = [','.join(s.split(',')[:3]) for s in symcen_tim]  # x,y,z
        symcentim = [np.int(s.split(',')[-1]) for s in symcen_tim]  # +1
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
            old = re.findall('\d[xyz]', C)
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

    # print mag_pos
    # print p1mag
    # print p1vec
    # print fg.mag(p1vec)
    # Add values to the dict
    crys = {}
    crys['filename'] = cifvals['Filename']
    crys['name'] = cifvals['FileTitle']
    crys['unit vector'] = UV
    crys['parent cell'] = UV
    crys['origin'] = np.array([[0., 0., 0.]])
    crys['symmetry'] = symops
    crys['atom type'] = p1typ
    crys['atom label'] = p1lbl
    crys['atom position'] = p1pos
    crys['atom occupancy'] = p1occ
    crys['atom uiso'] = p1Uiso
    crys['atom uaniso'] = np.zeros([len(p1pos), 6])
    crys['mag moment'] = p1vec
    crys['mag time'] = p1tim
    crys['normalise'] = 1.
    crys['misc'] = [element, label, cif_pos, occ, Uiso]
    crys['space group'] = spacegroup
    crys['space group number'] = sgn
    crys['cif'] = cifvals
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
        #'_atom_site_symmetry_multiplicity',
        #'_atom_site_Wyckoff_symbol',
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
        #'_atom_site_symmetry_multiplicity',
        #'_atom_site_Wyckoff_symbol',
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
        #'_atom_site_moment.symmform',
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
        elements = np.char.lower(np.asarray(elements).reshape(-1))
        all_elements = [el.lower() for el in data['Element']]
        # This will error if the required element doesn't exist
        try:
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
    elename = ' '.join(['%10s'%ele for ele in prop['Element']])
    out = '%8s : %s\n'%('', elename)
    for key in keys:
        propval = ' '.join(['%10s'%ele for ele in prop[key]])
        out += '%8s : %s\n' % (key, propval)
    return out


def neutron_scattering_length(element):
    """
    Return neutron scattering length, b, in fm
    Uses bound coherent scattering length from NIST
    https://www.ncnr.nist.gov/resources/n-lengths/
     b = neutron_scattering_length('Co')
    :param element: [n*str] list or array of elements
    :return: [n] array of scattering lengths
    """
    b = atom_properties(element, ['Coh_b'])
    return b


def xray_scattering_factor(element, Qmag=0):
    """
    Read X-ray scattering factor table, calculate f(|Q|)
    Uses the oefficients for analytical approximation to the scattering factors - ITC, p578
     Qff = read_xsf(element,Qmag=[0])
    :param element: [n*str] list or array of elements
    :param Qmag: [m] array of wavevector distance, in A^-1
    :return: [m*n] array of scattering factors
    """

    # Qmag should be a 1D array
    Qmag = np.asarray(Qmag).reshape(-1)

    coef = atom_properties(element, ['a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'a4', 'b4', 'c'])

    Qff = np.zeros([len(Qmag), len(coef)])

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

        f = a1 * np.exp(-b1 * (Qmag / (4 * np.pi)) ** 2) + \
            a2 * np.exp(-b2 * (Qmag / (4 * np.pi)) ** 2) + \
            a3 * np.exp(-b3 * (Qmag / (4 * np.pi)) ** 2) + \
            a4 * np.exp(-b4 * (Qmag / (4 * np.pi)) ** 2) + c
        Qff[:, n] = f
    return Qff


def magnetic_form_factor(element, Qmag=0.):
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
    Qmag = np.asarray(Qmag).reshape(-1)

    coef = atom_properties(element, ['j0_A', 'j0_a', 'j0_B', 'j0_b', 'j0_C', 'j0_c', 'j0_D'])

    # Nqpointings x Nelements
    Qff = np.zeros([len(Qmag), len(coef)])

    # Loop over elements
    for n in range(len(coef)):
        A = coef['j0_A'][n]
        a = coef['j0_a'][n]
        B = coef['j0_B'][n]
        b = coef['j0_b'][n]
        C = coef['j0_C'][n]
        c = coef['j0_c'][n]
        D = coef['j0_D'][n]

        j0 = A * np.exp(-a * Qmag ** 2) + \
             B * np.exp(-b * Qmag ** 2) + \
             C * np.exp(-c * Qmag ** 2) + D
        Qff[:, n] = j0
    return Qff


def attenuation(element_z, energy_keV):
    """
     Returns the x-ray mass attenuation, u/p, in cm^2/g
       e.g. A = attenuation(23,np.arange(7,8,0.01)) # Cu
            A = attenuation([23,24,25], 5.6)
            a = attenuation(19,4.5) # K
    """
    element_z = np.asarray(element_z).reshape(-1)
    energy_keV = np.asarray(energy_keV).reshape(-1)

    xma_file = os.path.join(datadir, 'XRayMassAtten_mup.dat')
    xma_data = np.loadtxt(xma_file)

    energies = xma_data[:, 0] / 1000.
    out = np.zeros([len(energy_keV), len(element_z)])
    for n, z in enumerate(element_z):
        # Interpolating the log values is much more reliable
        out[:, n] = np.exp(np.interp(np.log(energy_keV), np.log(energies), np.log(xma_data[:, z])))
        out[:, n] = np.interp(energy_keV, energies, xma_data[:, z])
    if len(element_z) == 1:
        return out[:, 0]
    return out


def atomic_scattering_factor(element, energy_kev=None):
    """
    Read atomic scattering factor table, giving f1+f2 for different energies
    From: http://henke.lbl.gov/optical_constants/asf.html
    :param element: str name of element
    :param energy_kev: float or list energy in keV (None to return original, uninterpolated list)
    :return: f1, f2
    """
    asf_file = os.path.join(datadir, 'atomic_scattering_factors.npy')
    asf = np.load(asf_file, allow_pickle=True)
    asf = asf.item()
    energy = np.array(asf[element]['energy'])/1000. # eV -> keV
    f1 = np.array(asf[element]['f1'])
    f2 = np.array(asf[element]['f2'])
    f1[f1 < -1000] = np.nan
    f2[f2 < -1000] = np.nan

    if energy_kev is None:
        return energy, f1, f2

    # Interpolate values
    if1 = np.interp(energy_kev, energy, f1)
    if2 = np.interp(energy_kev, energy, f2)
    return if1, if2


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
        sg_numbers = range(1,231)
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


def spacegroup_subgroups_list(sg_number):
    """
    Return str of maximal subgroups for spacegroup
    :param sg_number: space group number (1-230)
    :return: str
    """
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


def spacegroups_magnetic(sg_number):
    """
    Returns dict of magnetic space groups for required space group
    :param sg_number: space group number (1-230)
    :return: dict
    """
    sg_dict = spacegroup(sg_number)
    msg_numbers = sg_dict['magnetic space groups']

    msg_file = os.path.join(datadir, 'SpaceGroupsMagnetic.json')
    with open(msg_file, 'r') as fp:
        msg_dict = json.load(fp)
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


def spacegroup_magnetic_list(sg_number):
    """
    Return str list of magnetic space groups
    :param sg_number: space group number (1-230)
    :return: str
    """
    mag_spacegroups = spacegroups_magnetic(sg_number)
    out = ''
    fmt = 'Parent: %3s Magnetic: %-10s  %-10s  Setting: %3s %30s  Operators: %s\n'
    for sg in mag_spacegroups:
        parent = sg['parent number']
        number = sg['space group number']
        name = sg['space group name']
        setting = sg['setting']
        typename = sg['type name']
        #ops = sg['operators magnetic']
        ops = sg['positions magnetic']
        out += fmt % (parent, number, name, setting, typename, ops)
    return out


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
    return symbols[element_Z-1]


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
    find_charge = re.findall('[\d\.]+[\+-]', element)
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
    find_occ = re.findall('[\d\.]+', element)
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
    regex_element_num = re.compile('|'.join(['%s[\d\.]*' % el for el in ELEMENT_LIST]))
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
    newz = z - int(charge) # floor
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
            charge[n] = 1*occupancy
        elif group == 2:
            charge[n] = 2*occupancy
        elif group == 16:
            charge[n] = -2*occupancy
        elif group == 17:
            charge[n] = -1*occupancy
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
    :param list_of_elements:
    :return: [list of charges]
    """

    if occupancy is None:
        occupancy = np.ones(len(list_of_elements))
    else:
        occupancy = np.asarray(occupancy)

    charge = np.zeros(len(list_of_elements))
    for n in range(len(list_of_elements)):
        _, occ, _ = split_element_symbol(list_of_elements[n])
        charge[n] = occupancy[n]*default_atom_charge(list_of_elements[n])
        occupancy[n] = occupancy[n]*occ

    remaining_charge = -np.nansum(charge)
    uncharged = np.sum(occupancy[np.isnan(charge)])
    for n in range(len(list_of_elements)):
        if np.isnan(charge[n]):
            charge[n] = occupancy[n]*remaining_charge/uncharged
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
        atno = sum([occupancy[n] for n, x in enumerate(list_of_elements) if x == a])/float(divideby)
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
        atno[a] = sum([occupancy[n] for n, x in enumerate(list_of_elements) if x == a])/float(divideby)
        chno[a] = sum([charge[n] for n, x in enumerate(list_of_elements) if x == a])/(atno[a]*float(divideby))

    outstr = []
    for a in ats:
        outstr += [element_charge_string(a, atno[a], chno[a], latex)]
    return outstr


'-------------------Lattice Transformations------------------------------'


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

    UVs = 2 * np.pi * np.linalg.inv(UV).T
    return UVs


def indx(Q, UV):
    """
    Index Q(x,y,z) on on lattice [h,k,l] with unit vectors UV
    Usage:
      HKL = indx(Q,UV)
      Q = [[nx3]] array of vectors
      UV = [[3x3]] matrix of vectors [a,b,c]
    """
    HKL = np.dot(Q, np.linalg.inv(UV))
    return HKL


def Bmatrix(UV):
    """
    Calculate the Busing and Levy B matrix from a real space UV
    """

    a1, a2, a3 = fg.mag(UV)
    alpha3 = fg.ang(UV[0, :], UV[1, :])
    alpha2 = fg.ang(UV[0, :], UV[2, :])
    alpha1 = fg.ang(UV[1, :], UV[2, :])
    # print a1,a2,a3
    # print alpha1,alpha2,alpha3
    UVs = RcSp(UV) / (2 * np.pi)
    b1, b2, b3 = fg.mag(UVs)
    beta3 = fg.ang(UVs[0, :], UVs[1, :])
    beta2 = fg.ang(UVs[0, :], UVs[2, :])
    beta1 = fg.ang(UVs[1, :], UVs[2, :])
    # print b1,b2,b3
    # print beta1,beta2,beta3

    B = np.array([[b1, b2 * np.cos(beta3), b3 * np.cos(beta2)],
                  [0, b2 * np.sin(beta3), -b3 * np.sin(beta2) * np.cos(alpha1)],
                  [0, 0, 1 / a3]])
    return B


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
        out = eval(sym)
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
        x, y, z = sym.split(',')
        x = x.strip()
        y = y.strip()
        z = z.strip()
        for cen in cen_ops:
            cen = cen.lower()
            cen = cen.strip('\"\'')
            op = cen.replace('x', 'a').replace('y', 'b').replace('z', 'c')  # avoid replacing x/y twice
            op = op.replace('a', x).replace('b', y).replace('c', z)
            op = op.replace('--', '')
            ops += [op]
    return ops


def gen_sym_ref(sym_ops, hkl):
    """
     Generate symmetric reflections given symmetry operations
         symhkl = gen_sym_ref(sym_ops,h,k,l)
    """

    hkl = np.asarray(hkl, dtype=np.float)

    # Get transformation matrices
    sym_mat = gen_sym_mat(sym_ops)
    nops = len(sym_ops)

    symhkl = np.zeros([nops, 4])
    for n, sym in enumerate(sym_mat):
        symhkl[n, :] = np.dot(hkl, sym)

    return symhkl[:, :3]


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
     sym_mat = gen_syn_mat(['x,y,z','y,-x,z+1/2'])
     sym_mat[0] = [[ 1.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.],
                   [ 0.,  0.,  1.,  0.]])
     sym_mat[1] = [[ 0. ,  1. ,  0. ,  0. ],
                   [-1. ,  0. ,  0. ,  0. ],
                   [ 0. ,  0. ,  1. ,  0.5]]
    """
    sym_mat = []
    for sym in sym_ops:
        sym = sym.lower()
        ops = sym.split(',')
        mat = np.zeros([3, 4])

        for n in range(len(ops)):
            op = ops[n]
            op = op.strip('\"\'')
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
            op = op.replace('/', './')  # Allow float division

            if len(op.strip()) > 0:
                mat[n, 3] = eval(op)
        sym_mat += [mat]
    return sym_mat


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


def orthogonal_axes(x_axis=[1, 0, 0], y_axis=[0, 1, 0]):
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


def latpar2UV(a, b, c, alpha=90., beta=90., gamma=120.):
    """
    Convert a,b,c,alpha,beta,gamma to UV
     UV = latpar2UV(a,b,c,alpha=90.,beta=90.,gamma=120.)
     Vector c is defined along [0,0,1]
     Vector a and b are defined by the angles
    """

    # From http://pymatgen.org/_modules/pymatgen/core/lattice.html
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)
    val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) \
          / (np.sin(alpha_r) * np.sin(beta_r))
    # Sometimes rounding errors result in values slightly > 1.
    val = abs(val)
    gamma_star = np.arccos(val)
    aa = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
    bb = [-b * np.sin(alpha_r) * np.cos(gamma_star),
          b * np.sin(alpha_r) * np.sin(gamma_star),
          b * np.cos(alpha_r)]
    cc = [0.0, 0.0, c]

    return np.round(np.array([aa, bb, cc]), 8)


def latpar2UV_rot(a, b, c, alpha=90., beta=90., gamma=120.):
    """
    Convert a,b,c,alpha,beta,gamma to UV
     UV = latpar2UV_rot(a,b,c,alpha=90.,beta=90.,gamma=120.)
     Vector b is defined along [0,1,0]
     Vector a and c are defined by the angles
    """

    # From http://pymatgen.org/_modules/pymatgen/core/lattice.html
    alpha_r = np.radians(alpha)
    beta_r = np.radians(gamma)
    gamma_r = np.radians(beta)
    val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) \
          / (np.sin(alpha_r) * np.sin(beta_r))
    # Sometimes rounding errors result in values slightly > 1.
    val = abs(val)
    gamma_star = np.arccos(val)
    aa = [a * np.sin(beta_r), a * np.cos(beta_r), 0.0]
    bb = [0.0, b, 0.0]
    cc = [-c * np.sin(alpha_r) * np.cos(gamma_star),
          c * np.cos(alpha_r),
          c * np.sin(alpha_r) * np.sin(gamma_star)]

    return np.round(np.array([aa, bb, cc]), 8)


def UV2latpar(UV):
    """
    Convert UV=[a,b,c] to a,b,c,alpha,beta,gamma
     a,b,c,alpha,beta,gamma = UV2latpar(UV)
    """

    a = np.sqrt(np.sum(UV[0] ** 2))
    b = np.sqrt(np.sum(UV[1] ** 2))
    c = np.sqrt(np.sum(UV[2] ** 2))
    alpha = np.arctan2(np.linalg.norm(np.cross(UV[1], UV[2])), np.dot(UV[1], UV[2])) * 180 / np.pi
    beta = np.arctan2(np.linalg.norm(np.cross(UV[0], UV[2])), np.dot(UV[0], UV[2])) * 180 / np.pi
    gamma = np.arctan2(np.linalg.norm(np.cross(UV[0], UV[1])), np.dot(UV[0], UV[1])) * 180 / np.pi
    return a, b, c, alpha, beta, gamma


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


def hkl2dspace(hkl, UVstar):
    """
    Calcualte d-spacing from hkl reflection
    :param hkl: [nx3] array of reflections
    :param UV: [3x3] array of unit cell vectors
    :return: [nx1] array of d-spacing in A
    """
    Q = np.dot(hkl, UVstar)
    Qmag = fg.mag(Q)
    return q2dspace(Qmag)


def calqmag(twotheta, energy_kev=17.794):
    """
    Calculate |Q| at a particular 2-theta (deg) for energy in keV
     magQ = calqmag(twotheta, energy_kev=17.794)
    """

    energy = energy_kev * 1000.0  # energy in eV
    theta = twotheta * np.pi / 360  # theta in radians
    # Calculate |Q|
    magq = np.sin(theta) * energy * fg.e * 4 * np.pi / (fg.h * fg.c * 1e10)
    return magq


def cal2theta(qmag, energy_kev=17.794):
    """
    Calculate theta at particular energy in keV from |Q|
     twotheta = cal2theta(Qmag,energy_kev=17.794)
    """

    energy = energy_kev * 1000.0  # energy in eV
    # Calculate 2theta angles for x-rays
    twotheta = 2 * np.arcsin(qmag * 1e10 * fg.h * fg.c / (energy * fg.e * 4 * np.pi))
    # return x2T in degrees
    twotheta = twotheta * 180 / np.pi
    return twotheta


def caldspace(twotheta, energy_kev=17.794):
    """
    Calculate d-spacing from two-theta
     dspace = caldspace(tth, energy_kev)
    """
    qmag = calqmag(twotheta, energy_kev)
    dspace = q2dspace(qmag)
    return dspace


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
     Energy = wave2energy(wavelength)
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
     wavelength = energy2wave(energy)
    """

    # Electron Volts:
    E = 1000 * energy_kev * fg.e

    # SI: E = hc/lambda
    lam = fg.h * fg.c / E
    wavelength = lam / fg.A
    return wavelength


def biso2uiso(biso):
    """
    Convert B isotropic thermal parameters to U isotropic thermal parameters
    :param biso: Biso value or array
    :return: Uiso value or array
    """
    biso = np.asarray(biso, dtype=np.float)
    return biso / (8 * np.pi ** 2)


def uiso2biso(uiso):
    """
    Convert U isotropic thermal parameters to B isotropic thermal parameters
    :param uiso: Uiso value or array
    :return: Biso value or array
    """
    uiso = np.asarray(uiso, dtype=np.float)
    return uiso * (8 * np.pi ** 2)


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


def symmetry_ops2magnetic(operations):
    """
    Convert list of string symmetry operations to magnetic symmetry operations
    i.e. remove translations
    """
    # convert string to list
    if type(operations) is str:
        operations = [operations]
    # Use RegEx to find translations
    mag_op = []
    for op in operations:
        translations = re.findall('[\+\-]?\d/\d[\+\-]?', op)
        op = fg.multi_replace(op, translations, '')
        # also remove +/-1
        translations = re.findall('[\+\-]?\d+?[\+\-]?', op)
        op = fg.multi_replace(op, translations, '')
        mag_op += [op]
    return mag_op


def hkl2str(hkl):
    """
    Convert hkl to string (h,k,l)
    :param hkl:
    :return: str '(h,k,l)'
    """

    out = '(%1.3g,%1.3g,%1.3g)'
    hkl = np.asarray(hkl, dtype=np.float).reshape([-1, 3])
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
    qq = np.sqrt(qx**2 + qy**2 + qz**2)
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
        T = exp( -Uiso/2Q^2 )
    """

    uiso = np.asarray(uiso, dtype=np.float).reshape(1, -1)
    Qmag = np.asarray(Qmag, dtype=np.float).reshape(-1, 1)

    Tall = np.exp(-0.5 * np.dot(Qmag, uiso))
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
    return 1/q**2


def calc_vol(UV):
    "Calculate volume in Angstrom^3 from unit vectors"
    a, b, c = UV
    return np.dot(a, np.cross(b, c))


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
        uiso = ['0.000']*len(label)
    # Occupancy
    if '_atom_site_occupancy' in keys:
        occ = cif['_atom_site_occupancy']
    else:
        occ = ['1.0']*len(label)
    # Multiplicity
    if '_atom_site_site_symmetry_multiplicity' in keys:
        mult = [s+'a' for s in cif['_atom_site_site_symmetry_multiplicity']]
    else:
        mult = ['1a']*len(label)

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
        extra += ['R$_w$ = %4.2f\\%%' % (float(cif['_refine_ls_wR_factor_ref'])*100)]
    extra = ', '.join(extra)

    # Create table str
    out = '%% %s\n' % cif['Filename']
    out += '\\begin{table}[htp]\n'
    out += '    \\centering\n'
    out += '       \\begin{tabular}{c|c|ccccc}\n'
    out += '             & Site & x & y & z & Occ. & %s \\\\ \hline\n' % u_or_b
    fmt = '        %4s & %5s & %s & %s & %s & %s & %s \\\\\n'
    for n in range(len(label)):
        out += fmt % (label[n], mult[n], u[n], v[n], w[n], occ[n], uiso[n])
    out += '        \\end{tabular}\n'
    out += '    \\caption{%s with %s. %s}\n' % (name, lp, extra)
    out += '    \\label{tab:}\n'
    out += '\\end{table}\n'
    return out