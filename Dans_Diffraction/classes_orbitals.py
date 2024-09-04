# -*- coding: utf-8 -*-
"""
 Orbitals class "classes_orbitals.py"
 Build an ionic compound or molecule from ions with defined atomic orbitals

 orbital = Orbital('4d4')
 atom = Atom('Co3+')
 compound = CompoundString('Ca2RuO4')

 Orbital:
    individual atomic orbital with properties, n,l and fill
    functions:
        add_electron
        remove_electron
        next_orbital
        last_orbital

 Atom:
    Single element composed of a list of orbitals (self.orbitals)
    functions:
        valence_orbitals
        assign_charge
        unoccupied2charge
        add_electron
        remove_electron
        transfer_electron

 Compound:
    Collection of atoms, with charge automatically balanced by assigning standard charges to common valences
    Atoms are collected in the list self.atom_list

By Dan Porter, PhD
Diamond
2020

Version 1.1.0
Last updated: 04/09/24

Version History:
09/05/20 0.1.0  Version History started.
12/07/20 1.0.0  Program functional, performed some testing
04/09/24 1.1.0  Added missing orbitals in Atom.assign_charge

@author: DGPorter
"""

import sys, os, re
import numpy as np
from warnings import warn

from . import functions_general as fg
from . import functions_crystallography as fc

__version__ = '1.1.0'

OXIDATION_FILE = os.path.join(fc.datadir, 'Element_OxidationStates.txt')


def oxidation_states(element=None, element_z=None):
    """
    Return available oxidation states for element
    :param element: str or None for all elements
    :param element_z: int, alternative to element symbol
    :return: list of float
    """
    available_states = {}
    with open(OXIDATION_FILE) as f:
        for n in range(118):
            line = f.readline().split()
            z = int(line[0])
            symbol = line[2]
            if len(line) > 3:
                available_states[symbol] = [int(val) for val in line[3:]]
            else:
                available_states[symbol] = []
            if symbol == element:
                return available_states[symbol]
            elif z == element_z:
                return available_states[symbol]
    return available_states


def atomstring2list(compound_string):
    return [Atom(at) for at in fc.split_compound(compound_string)]


def orbital_list(n_electrons, first_orbital=None, first_n=1, first_l=0):
    """
    Create a list of orbitals filling n electrons
    :param n_electrons:
    :param first_orbital: None or str:
    :param first_n:
    :param first_l:
    :return: [Orbital]
    """
    if first_orbital is None:
        first_orbital = Orbital(n=first_n, l=first_l)
    else:
        first_orbital = Orbital(first_orbital)

    n_electrons = first_orbital.add_electron(n_electrons)
    orbitals = [first_orbital]
    while n_electrons > 0:
        orbitals += [orbitals[-1].next_orbital()]
        n_electrons = orbitals[-1].add_electron(n_electrons)
    return orbitals


class Orbital:
    """
    Individual orbital
    orb_3d7 = Orbital('3d7')
    or
    orb_3d7 = Orbital(n=3,l=2,fill=7)

    print(orb_3d7)
    >> 3d7
    """
    _level_max = [0, 1, 2, 3, 3, 2, 1]
    _state = ['s', 'p', 'd', 'f']
    _state_max = [2, 6, 10, 14]

    def __init__(self, orbital_string=None, n=1, l=0, fill=0):
        if orbital_string is not None:
            self.standard = orbital_string
            self.n = int(orbital_string[0])
            self.l_name = orbital_string[1].lower()
            self.l = self._state.index(orbital_string[1].lower())
            if len(orbital_string) == 2:
                self.fill = 0.
            else:
                self.fill = float(orbital_string[2:])
            self.max_fill = self._state_max[self.l]
            self.max_l = self._level_max[self.n - 1]
        else:
            self.n = int(n)
            self.l = int(l)
            self.fill = float(fill)
            self.l_name = self._state[l]
            self.max_fill = self._state_max[self.l]
            self.max_l = self._level_max[self.n - 1]
            self.standard = self.generate_string_standard()

    def __call__(self):
        return self.generate_string_standard()

    def __repr__(self):
        return 'Orbital(%s)' % self.generate_string_standard()

    def __str__(self):
        return self.generate_string_standard()

    def __eq__(self, other):
        if isinstance(other, Orbital):
            if self.n == other.n and self.l == other.l:
                return True
        return False

    def generate_string_standard(self):
        return '%d%s%1.3g' % (self.n, self.l_name, self.fill)

    def generate_string_fdmnes(self, spin=False):
        if spin:
            # split occupancy accross up and down states
            return '%d %d %4.2f %4.2f' % (self.n, self.l, self.fill/2, self.fill/2)
        else:
            return '%d %d %4.2f' % (self.n, self.l, self.fill)

    def generate_string_latex(self):
        return '%d%s$^{%1.3g}$' % (self.n, self.l_name, self.fill)

    def add_electron(self, n=1.):
        new_fill = self.fill + n
        if new_fill > self.max_fill:
            new_fill = 1.0 * self.max_fill
        elif new_fill < 0:
            new_fill = 0.0
        unused = (self.fill + n) - new_fill
        self.fill = new_fill
        return unused

    def remove_electron(self, n=1.):
        new_fill = self.fill - n
        if new_fill > self.max_fill:
            new_fill = 1.0 * self.max_fill
        elif new_fill < 0:
            new_fill = 0.0
        unused = (self.fill - n) - new_fill
        self.fill = new_fill
        return unused

    def next_orbital(self, fill=0):
        if self.l + 1 > self.max_l:
            next_l = 0
            next_n = self.n + 1
        else:
            next_l = self.l + 1
            next_n = self.n + 0
        return Orbital(n=next_n, l=next_l, fill=fill)

    def last_orbital(self, fill=0):
        if self.l == 0:
            next_n = self.n - 1
            if next_n < 0: next_n = 0
            next_l = self._level_max[next_n]
        else:
            next_n = self.n + 0
            next_l = self.l - 1
        return Orbital(n=next_n, l=next_l, fill=fill)


class Atom:
    """
    Atom - collection of orbitals
    Co = Atom('Co')
    print(Co)
    >> 27 Co  1s2 2s2 2p6 3s2 3p6 3d7 4s2
    Co = Atom('Co3+')
    print(Co)
    >>  27 Co3+  1s2 2s2 2p6 3s2 3p6 3d5 4s1
    """

    def __init__(self, element, charge=None, occupancy=None):
        element, str_occupancy, str_charge = fc.split_element_symbol(element)
        if charge is None:
            charge = str_charge
        if occupancy is None:
            occupancy = str_occupancy
        self.element_str = element
        self.element_symbol = element
        self.occupancy = occupancy
        self.z, self.name, config = fc.atom_properties(element, ['Z', 'Name', 'Config'])[0]
        self.orbitals = [Orbital(s) for s in config.split('.')]
        #self.oxidation_states = oxidation_states(element)
        self.charge = charge
        self.assign_charge(charge)

    def __repr__(self):
        elestr = fc.element_charge_string(self.element_symbol, occupancy=self.occupancy, charge=self.charge)
        return "Atom(%s)" % elestr

    def __str__(self):
        return 'Atom: %s\n' % self.generate_string_standard()

    def __getitem__(self, item):
        return self.orbitals.__getitem__(item)

    def valence_orbitals(self):
        valence = []
        for orbital in self.orbitals[:-1]:
            if orbital.max_fill - orbital.fill > 0.01:
                valence += [orbital]
        valence += [self.orbitals[-1]]
        return valence

    def find_orbital(self, orbital):
        """
        Returns requested orbital (useful for findind Orbital.last_orbital
        :param orbital: Orbital
        :return: Orbital
        """
        idx = self.orbitals.index(orbital)
        return self.orbitals[idx]

    def find_orbital_str(self, orbital_str):
        """
        Returns requested orbital
        :param orbital_str: str, e.g. '4d'
        :return: Orbital
        """
        neworb = Orbital(orbital_str)
        return self.find_orbital(neworb)

    def assign_standard_charge(self):
        """
        Add standard charge based on element group
            Group 1 metals  always +1
            Group 2 metals  always +2
            Oxygen  usually -2  except in peroxides and F2O (see below)
            Hydrogen    usually +1  except in metal hydrides where it is -1 (see below)
            Fluorine    always -1
            Chlorine    usually -1  except in compounds with O or F (see below)
        :return:
        """
        if self.z in [3, 11, 19, 37, 55, 87]:  # group 1
            self.assign_charge(1)
        elif self.z in [4, 12, 20, 38, 56, 88]:  # group 2
            self.assign_charge(2)
        elif self.z in [8]:  # O
            self.assign_charge(-2)
        elif self.z in [1]:  # H
            self.assign_charge(1)
        elif self.z in [9]:  # F
            self.assign_charge(-1)
        elif self.z in [17]:  # Cl
            self.assign_charge(-1)

    def assign_charge(self, charge):
        """
        charge = number of missing electrons per atom
        Add charge to atom, select new orbital configuration based on surrounding element't config.
        For non-integer charge, remaining component will be added or subtracted from final orbital
        Charge is averaged over the full occupancy of the sites
        :param charge: flaot
        :return:
        """

        self.charge = charge
        intcharge = np.floor(charge)
        deccharge = charge % 1
        # self.orbitals = [Orbital(s) for s in fc.orbital_configuration(self.element_symbol, intcharge)]
        # # add empty orbitals
        # for n, orbital in enumerate(self.orbitals[:-1]):
        #     next_orbital = orbital.next_orbital(fill=0)  # empty orbital
        #     if next_orbital not in self.orbitals:
        #         self.orbitals.insert(n + 1, next_orbital)
        neutral_orbitals = [Orbital(s) for s in fc.orbital_configuration(self.element_symbol, 0)]
        charge_orbitals = [Orbital(s) for s in fc.orbital_configuration(self.element_symbol, intcharge)]
        # add missing orbitals
        for n, orbital in enumerate(neutral_orbitals):
            if orbital not in charge_orbitals:
                orbital.fill = 0
                charge_orbitals.insert(n, orbital)
        self.orbitals = charge_orbitals
        if deccharge > 0:
            self.orbitals[-1].remove_electron(deccharge)

    def assign_occupancy(self, occupancy):
        self.occupancy = occupancy
        self.assign_charge(self.charge)

    def unoccupied2charge(self):
        """
        Convert non-integer occupancy to charge
        :return:
        """
        full_occ = np.ceil(self.occupancy)
        newcharge = self.charge * self.occupancy / full_occ  # average the charge among multiple ions
        if abs(newcharge) > 0.01:
            self.occupancy = full_occ
            self.assign_charge(newcharge)

    def check_charge(self):
        return self.occupancy*(self.z - sum([orb.fill for orb in self.orbitals]))

    def add_electron(self, n_electron, add_to_state=None):
        """
        Adds electrons to lowest unfilled or selectred orbital, adding additinal orbitals if full
        :param n_electron: float number of electrons to add
        :param add_to_state: None (lowest unfilled orbital) or str e.g. '4d'
        :return:
        """
        if add_to_state is None:
            state = self.valence_orbitals()[0]
        else:
            state = self.find_orbital_str(add_to_state)

        unused = state.add_electron(n_electron)
        while unused > 0:
            next_orbital = state.next_orbital()
            try:
                state = self.find_orbital(next_orbital)
            except ValueError:  # orbital not in list
                self.orbitals += [next_orbital]
                state = self.orbitals[-1]
            unused = state.remove_electron(unused)
        self.charge -= n_electron

    def remove_electron(self, n_electron, add_to_state=None):
        """
        Removes electrons from highest or selectred orbital, removing orbitals if empty
        :param n_electron: float number of electrons to add
        :param add_to_state: None (lowest unfilled orbital) or str e.g. '4d'
        :return:
        """
        if add_to_state is None:
            state = self.orbitals[-1]
        else:
            state = self.find_orbital_str(add_to_state)

        unused = state.remove_electron(n_electron)
        while unused > 0:
            prev_orbital = state.last_orbital()
            state = self.find_orbital(prev_orbital)
            unused = state.remove_electron(unused)
        self.charge += n_electron
        self.clean_orbitals()

    def transfer_electron(self, state_from, state_to, n_electron=1.):
        """
        Transfer an electron from one state to another
        :param state_from: str e.g. '5s' transfer electrons from this orbital
        :param state_to: str e.g. '4d' transfer electrons to this orbital (or next available)
        :param n_electron: float number of electrons
        :return: None
        """
        self.remove_electron(n_electron, state_from)
        self.add_electron(n_electron, state_to)

    def clean_orbitals(self):
        """
        Removes highest orbitals with no electrons
        """
        while abs(self.orbitals[-1].fill) < 0.01:
            self.orbitals = self.orbitals[:-1]

    def generate_string_standard(self):
        elestr = fc.element_charge_string(self.element_symbol, occupancy=self.occupancy, charge=self.charge)
        orbstr = ' '.join([orb.generate_string_standard() for orb in self.orbitals])
        return '%3d %10s  %s' % (self.z, elestr, orbstr)

    def generate_string_fdmnes(self, spin=False):
        elestr = fc.element_charge_string(self.element_symbol, occupancy=self.occupancy, charge=self.charge)
        empty_orbital = self.orbitals[-1].next_orbital()
        valence = self.valence_orbitals()
        orbitals = valence + [empty_orbital]
        # keep atom neutral by adding charge electrons to outer shell
        unused = orbitals[-1].add_electron(self.charge)
        fdm = '  '.join([orb.generate_string_fdmnes(spin) for orb in orbitals])
        std = ' '.join([orb.generate_string_standard() for orb in orbitals])
        return '%3d %d  %s ! %s %s' % (self.z, len(orbitals), fdm, elestr, std)

    def generate_string_latex(self):
        orbstr = ' '.join([orb.generate_string_latex() for orb in self.orbitals])
        elestr = fc.element_charge_string(self.element_symbol, self.occupancy, self.charge, latex=True)
        return '%4s:  %s' % (elestr, orbstr)


class Compound:
    """
    Compound - collection of atoms
    LiCoO2 = Compound([Atom, Atom, Atom])
    print(LiCoO2)
    >> Li0.7CoO2:
    >>   3 Li0.7+  1s2 2s0.3
    >>  27 Co3.3+  1s2 2s2 2p6 3s2 3p6 3d5 4s0.7
    >>   8  O2-  1s2 2s2 2p6
    >>   8  O2-  1s2 2s2 2p6
    """

    def __init__(self, atom_list):
        self.atom_list = atom_list  # list of Atom objects
        self.balance_charge()
        self.compound_string = self.generate_charge_name()

    def __repr__(self):
        return 'Compound(%s)' % self.compound_string

    def __str__(self):
        return 'Compound: %s:\n%s' % (self.compound_string, self.generate_string_standard())

    def __getitem__(self, item):
        return self.atom_list.__getitem__(item)

    def charge_list(self):
        return [at.occupancy*at.check_charge() for at in self.atom_list]

    def balance_charge(self):
        """
        Add together known charges of elements (e.g. O, Na, Ca), use to estimate the charges of remaining ions
        :return:
        """
        for at in self.atom_list:
            if abs(at.charge) < 0.1:
                at.assign_standard_charge()
            at.unoccupied2charge()

        charge_list = [at.check_charge() for at in self.atom_list]
        charge_sum = sum(charge_list)

        # Assign remaining charge
        tot_uncharged = np.sum([at.occupancy for at in self.atom_list if abs(at.charge) < 0.1])
        if tot_uncharged < 0.1: return
        for at in self.atom_list:
            if abs(at.charge) < 0.1:
                at.assign_charge(-charge_sum/tot_uncharged)
                at.unoccupied2charge()

    def check_charge(self):
        return sum([at.check_charge() for at in self.atom_list])

    def generate_charge_name(self):
        names = [fc.element_charge_string(at.element_symbol, at.occupancy, at.charge) for at in self.atom_list]
        cnames = []
        for name in names:
            if names.count(name) > 1:
                newname = '%1.3g(%s)' % (names.count(name), name)
            else:
                newname = name
            if newname not in cnames:
                cnames += [newname]
        return ' '.join(cnames)

    def generate_string_standard(self):
        """Generate string showing atomic orbitals"""
        return '\n'.join([at.generate_string_standard() for at in self.atom_list])

    def generate_string_fdmnes(self, spin=False):
        """Generate string showing atomic orbittals in FDMNES format"""
        return '\n'.join([at.generate_string_fdmnes(spin) for at in self.atom_list])

    def generate_string_fdmnes_absorber(self, absorber, spin=False):
        """
        Generate string showing atomic orbittals in FDMNES format
        :param absorber: str element symbol of principal absorber
        :param spin: Bool, if True element populations will be split by spin up/down
        :return: str, element list
        """
        ele_list = [at.element_symbol for at in self.atom_list if at.element_symbol == absorber]
        ele_list += [at.element_symbol for at in self.atom_list if at.element_symbol != absorber]
        out = [at.generate_string_fdmnes(spin) for at in self.atom_list if at.element_symbol == absorber]
        out += ['%3d 0 ! %s' % (at.z, at.element_symbol) for at in self.atom_list if at.element_symbol != absorber]
        return '\n'.join(out), ele_list

    def generate_string_latex(self):
        """Generate string showing atomic orbittals in Latex format"""
        return '\n'.join([at.generate_string_latex() for at in self.atom_list])


class CompoundString(Compound):
    """
    Compound - collection of atoms
    LiCoO2 = Compound('Li0.7CoO2')
    print(LiCoO2)
    >> Li0.7CoO2:
    >>   3 Li0.7+  1s2 2s0.3
    >>  27 Co3.3+  1s2 2s2 2p6 3s2 3p6 3d5 4s0.7
    >>   8  O2-  1s2 2s2 2p6
    >>   8  O2-  1s2 2s2 2p6
    """
    def __init__(self, compound_string):
        self.compound_string = compound_string
        atom_list = atomstring2list(compound_string)
        super().__init__(atom_list)


class CrystalOrbitals(Compound):
    """
    Crystal Orbitals
    The compound is created as a class structure with each atom having a set of orbital electronic states
    """
    def __init__(self, xtl):
        self.xtl = xtl

        atom_type = np.asarray(self.xtl.Structure.type)
        atom_occ = np.asarray(self.xtl.Structure.occupancy)

        # Count elements
        _, atom_count = np.unique(atom_type, return_counts=True)
        atlist = fc.count_atoms(atom_type, occupancy=atom_occ, divideby=np.min(atom_count))
        atomlist = [Atom(a) for a in atlist]
        super().__init__(atomlist)
