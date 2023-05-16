# -*- coding: utf-8 -*-
"""
Properties Class "classes_properties.py"
 functions for calculating various properties, working with the Crystal class.

By Dan Porter, PhD
Diamond
2017

Version 1.7
Last updated: 26/01/21

Version History:
10/11/17 0.1    Program created
06/01/18 1.0    Program renamed
11/03/18 1.1    Added properties.info(), element.info()
23/02/19 1.2    Added xray_edges
15/08/19 1.3    Added molcharge
30/03/20 1.4    Added latex_table, info returns str, removed getattr from xray_edges, check if element exists
12/05/20 1.5    Added orbitals function, exchange_paths
15/10/20 1.6    Added scattering lengths + factors
26/01/21 1.7    Added calculation of xray attenuation, transmission and refractive index


@author: DGPorter
"""

import numpy as np

from . import functions_general as fg
from . import functions_crystallography as fc
from .classes_orbitals import CrystalOrbitals

__version__ = '1.7'


class Properties:
    """
    Properties functions for the Crystal Object
    Example Usage:
        xtl = Crystal('file.cif')
        print(xtl.Properties)
        xtl.Properties.density()
        print(xtl.Properties.molcharge())
        xtl.Properties.Co.K  # returns K edge of element Co (if Co in Crystal)
    """
    def __init__(self, xtl):
        """initialise"""
        self.xtl = xtl

        types = np.unique(self.xtl.Structure.type)
        all_elements = fc.atom_properties(None, 'Element')
        for element in types:
            if str(element) in all_elements:
                setattr(self, str(element), Element(str(element)))

    def update_cif(self, cifvals):
        """
        Update cif dict with new values
        :param cifvals: cif dict from readcif
        :return: cifvals
        """

        cifvals['_chemical_formula_sum'] = self.molname()
        cifvals['_chemical_formula_weight'] = '%1.4f' % self.weight()
        return cifvals

    def orbitals(self):
        """ Return orbital Compound from classes_orbitals"""
        orbitals = CrystalOrbitals(self.xtl)
        self.xtl.Orbitals = orbitals
        return orbitals

    def volume(self):
        """Returns the volume in A^3"""
        return self.xtl.Cell.volume()

    def density(self):
        """Return the density in g/cm"""

        vol = self.xtl.Cell.volume()*1e-24 # cm^3
        weight = self.xtl.Structure.weight()/fg.Na # g
        return weight/vol

    def atoms_per_volume(self):
        """Return no. atoms per volume in atoms per A^3"""

        occ = self.xtl.Structure.occupancy
        natoms = np.sum(occ)
        vol = self.volume()  # A^3
        return natoms / vol

    def weight(self):
        """Return the molecular weight in g/mol"""
        return self.xtl.Structure.weight()

    def neutron_scatteringlengths(self):
        """
        Return the bound coherent neutron scattering length for each atom in the structure
        :return: [m] array of scattering lengths for each atom
        """
        return fc.neutron_scattering_length(self.xtl.Structure.type)

    def xray_scattering_factor(self, hkl):
        """
        Return the x-ray scattering factor for given hkl reflections
        :param hkl: [nx3] array of (h,k,l) reflections
        :return: [nxm] array of scattering factors for each atom and reflection
        """
        qmag = self.xtl.Cell.Qmag(hkl)
        return fc.xray_scattering_factor(self.xtl.Structure.type, qmag)

    def magnetic_form_factor(self, hkl):
        """
        Return the magnetic form factor for given hkl reflections
        :param hkl: [nx3] array of (h,k,l) reflections
        :return: [nxm] array of scattering factors for each atom and reflection
        """
        qmag = self.xtl.Cell.Qmag(hkl)
        return fc.magnetic_form_factor(self.xtl.Structure.type, qmag)

    def xray_edges(self):
        """
        Returns the x-ray absorption edges available and their energies
        string, energies = self.xray_edges()
        E.G.
        string = ['Ru K', 'Ru L1', 'Ru L2' ...]
        energies = [22., ...]
        """

        types = np.unique(self.xtl.Structure.type)
        edges = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5']
        out_str = []
        out_eng = []
        for element in types:
            for edge in edges:
                energy = fc.atom_properties(element, edge)[0]
                if energy > 0.1:
                    out_str += ['%s %s' % (element, edge)]
                    out_eng += [energy]
        return out_str, out_eng

    def molfraction(self, Z=1):
        """
        Display the molecular weight of a compound and atomic fractions
        :param Z: filling number
        :return: str
        """

        atom_type = self.xtl.Structure.type
        occ = self.xtl.Structure.occupancy

        # Count elements
        ats = np.unique(atom_type)
        weights = fc.atom_properties(ats,'Weight')
        atno = np.zeros(len(ats))
        for n,element in enumerate(ats):
            atno[n] = sum([occ[m] for m,x in enumerate(atom_type) if x == element])

        outstr=''

        total_weight = self.weight()
        outstr += 'Weight = %6.2f g/mol\n' % (total_weight / Z)

        for n, element in enumerate(ats):
            ff = 100*atno[n]*weights[n]/total_weight
            outstr += '{:3s} [{:8.3f}] {:5.2g} {:5.2f}%\n'.format(element,weights[n],atno[n]/Z,ff)
        return outstr

    def molname(self, element=None, element_no=None, latex=False):
        """
        Generate molecular name of crystal
            element : str : specify element that will have a coordination number
            element_no : float : The coordination number of element
            latex : True/False : if true, outputs in latex format
        :return: str
        """

        atom_type = np.asarray(self.xtl.Structure.type)
        occ = np.asarray(self.xtl.Structure.occupancy)

        # Count elements
        ats = np.unique(atom_type)
        ats = fc.arrange_atom_order(ats)  # arrange this by alcali/TM/O
        atno = {}
        for a in ats:
            atno[a] = sum([occ[n] for n,x in enumerate(atom_type) if x == a])

        # default - set max Z atom to it's occupancy
        if element is None:
            z = fc.atom_properties(ats, 'Z')
            element = ats[np.argmax(z)]
            #element = min(atno, key=atno.get)

        if element_no is None:
            element_no = np.max(occ[atom_type == element])

        divideby = atno[element]/ element_no
        name = fc.count_atoms(atom_type, occ, divideby, latex)
        name = ''.join(name)
        return name

    def molcharge(self, element=None, element_no=None, latex=False):
        """
        Generate molecular charge composition of crystal
            element : str : specify element that will have a coordination number
            element_no : float : The coordination number of element
            latex : True/False : if true, outputs in latex format
            charge: True/False : if true, outputs chemical charge format
        :return: str
        """

        atom_type = np.asarray(self.xtl.Structure.type)
        occ = np.asarray(self.xtl.Structure.occupancy)

        # Count elements
        ats = np.unique(atom_type)
        ats = fc.arrange_atom_order(ats)  # arrange this by alcali/TM/O
        atno = {}
        for a in ats:
            atno[a] = sum([occ[n] for n,x in enumerate(atom_type) if x == a])

        # default - set max Z atom to it's occupancy
        if element is None:
            z = fc.atom_properties(ats, 'Z')
            element = ats[np.argmax(z)]
            #element = min(atno, key=atno.get)

        if element_no is None:
            element_no = np.max(occ[atom_type == element])

        divideby = atno[element]/ element_no
        name = fc.count_charges(atom_type, occ, divideby, latex)
        name = ' '.join(name)
        return name

    def absorption(self, energy_kev=None):
        """
        Returns the sample absorption coefficient in um^-1 at the requested energy in keV
        """

        if energy_kev is None:
            energy_kev = fc.getenergy()

        # determine u/p = sum(wi*(u/p)i) http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html
        wi = self.xtl.Structure.mass_fraction()
        Z = fc.atom_properties(self.xtl.Structure.type, 'Z')
        upi = fc.attenuation(Z, energy_kev)
        up = np.sum(wi*upi, axis=1)

        u = up*self.density()/10000
        return u

    def xray_transmission(self, energy_kev=None, thickness_um=100):
        """
        Calculate transmission of x-ray through a slab of material
        Equivalent to https://henke.lbl.gov/optical_constants/filter2.html
        Based on formulas from: Henke, Gullikson, and Davis, Atomic Data and Nuclear Data Tables 54 no.2, 181-342 (July 1993)
        :param energy_kev: float or array, x-ray energy in keV
        :param thickness_um: slab thickness in microns
        :return: float or array
        """
        if energy_kev is None:
            energy_kev = fc.getenergy()

        atom_type = self.xtl.Structure.type
        occ = self.xtl.Structure.occupancy
        natoms = np.sum(occ)
        vol = self.volume()
        atom_per_volume = natoms / vol # atoms per A^3
        elements = ['%s%s' % (at, o) for at, o in zip(atom_type, occ)]
        return fc.xray_transmission(elements, energy_kev, atom_per_volume, thickness_um)

    def xray_attenuation_length(self, energy_kev=None, grazing_angle=90):
        """
        Calcualte X-Ray Attenuation Length
        Equivalent to: https://henke.lbl.gov/optical_constants/atten2.html
        Based on formulas from: Henke, Gullikson, and Davis, Atomic Data and Nuclear Data Tables 54 no.2, 181-342 (July 1993)
        :param energy_kev: float or array, x-ray energy in keV
        :param grazing_angle: incidence angle relative to the surface, in degrees
        :return: float or array, in microns
        """
        if energy_kev is None:
            energy_kev = fc.getenergy()

        atom_type = self.xtl.Structure.type
        occ = self.xtl.Structure.occupancy
        natoms = np.sum(occ)
        vol = self.volume()
        atom_per_volume = natoms / vol  # atoms per A^3
        elements = ['%s%s' % (at, o) for at, o in zip(atom_type, occ)]
        return fc.xray_attenuation_length(elements, energy_kev, atom_per_volume, grazing_angle)

    def xray_reflectivity(self, energy_kev=None, grazing_angle=2):
        """
        Calculate the specular reflectivity of a material
          NOT CURRENTLY WORKING
        :param elements: str or list of str, if list - absorption will be summed over elements
        :param energy_kev: float array
        :param grazing_angle: incidence angle relative to the surface, in degrees
        :return: float or array
        """
        if energy_kev is None:
            energy_kev = fc.getenergy()
        raise Warning("Warning: this doesn\'t work yet")

        atom_type = self.xtl.Structure.type
        occ = self.xtl.Structure.occupancy
        natoms = np.sum(occ)
        vol = self.volume()
        atom_per_volume = natoms / vol  # atoms per A^3
        elements = ['%s%s' % (at, o) for at, o in zip(atom_type, occ)]

        refindex = fc.xray_refractive_index(elements, energy_kev, atom_per_volume)
        costh = np.cos(np.deg2rad(grazing_angle))
        ki = costh
        kt = np.sqrt(refindex * np.conj(refindex) - costh ** 2)
        r = (ki - kt) / (ki + kt)
        return np.real(r * np.conj(r))

    def diamagnetic_susceptibility(self, atom_type='volume'):
        """
        Calculate diamagnetic contribution to susceptibility

        type = 'volume' > returns volume susceptiblity in -dimensionless-
        type = 'cgs volume' > returns volume susceptiblity in emu/cm^3
        type = 'mass' > returns mass susceptiblity in m^3/kg
        type = 'cgs mass' > returns mass susceptiblity in emu/g
        type = 'molar' > returns molar susceptiblity in m^3/mol
        type = 'cgs molar' > returns molar susceptiblity in emu/mol
        """

        # Formula taken from Blundel "Magnetism in Condensed Matter" S2.3 Diamagnetism (p21)
        # X = -(N/V)(e^2*u0/6me)(Zeff*r^2)
        # X = diamagnetic susceptibility, N=no ions, V = volume
        # Zeff = no valence electrons, r = ionic radius

        C = (fg.u0 * fg.e ** 2) / (6 * fg.me)
        V = self.xtl.Cell.volume() * 1e-30  # m^3
        M = self.xtl.Structure.weight() / fg.Na  # g
        mol = 1 / fg.Na

        # susceptibility type
        if atom_type.lower() == 'mass':
            D = M / 1000  # kg
        elif atom_type.lower() == 'cgs mass':
            D = M * (4 * fg.pi)
        elif atom_type.lower() == 'molar':
            D = mol
        elif atom_type.lower() == 'cgs molar':
            D = mol * (4 * fg.pi * 1e-6)
        elif atom_type.lower() == 'cgs volume':
            D = V * (4 * fg.pi)
        else:
            D = V

        atom_types = np.unique(self.xtl.Structure.type)
        X = 0.0
        for at in atom_types:
            N = np.sum(self.xtl.Structure.occupancy[self.xtl.Structure.type == at])
            Zeff = fc.atom_properties(str(at), 'ValenceE')[0]
            r = fc.atom_properties(str(at), 'Radii')[0] * 1e-12
            X += -(N / D) * C * (Zeff * r * r)
        return X

    def atomic_neighbours(self, structure_index=0, radius=2.5, disp=False):
        """
        Returns the relative positions of atoms within a radius of the selected atom
            vector, label = atomic_distances(idx, radius)
                idx = index of central atom, choosen from xtl.Structrue.info()
                radius = radius about the central atom to find other ions, in Angstroms
                disp = False*/True display the output
                vector = [nx3] array of atomic positions relative to central atom
                label = labels of the neighboring ions
        """

        Rindex = self.xtl.Cell.calculateR(self.xtl.Structure.uvw()[structure_index, :])

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.generate_lattice(1, 1, 1)

        R = self.xtl.Cell.calculateR(uvw)

        diff = R - Rindex

        mag = fg.mag(diff)
        jj = np.argsort(mag)
        diff = diff[jj,:]
        label = label[jj]
        mag = mag[jj]
        ii = np.where(mag<radius)[0]

        if disp:
            for i in ii:
                print('{:3.0f} {:4s} {:6.3f} {:6.3f} {:6.3f} dist = {:6.3f}'.format(i, label[i], diff[i, 0], diff[i, 1],
                                                                                    diff[i, 2], mag[i]))
        return diff[ii,:], label[ii]

    def exchange_paths(self, cen_idx=None, nearest_neighbor_distance=7.0, exchange_type='O', bond_angle=90.,
                       search_in_cell=True, group_neighbors=True, disp=False, return_str=False):
        """
        Calcualte likely exchange pathways between neighboring ions within a certain radius
        Ions in exchange path are selected by being close to the bond between neighboring atoms of the same type to the
        central atom.
        :param cen_idx: index of central ion in xtl.Structure, or None to select 1st magnetic ion in list
        :param nearest_neighbor_distance: Maximum radius to serach to
        :param exchange_type: str or None. Exchange path only incoroporates these elements, or None for any
        :param bond_angle: float. Search for exchange ions withing this angle to the bond, in degrees
        :param search_in_cell: Bool. If True, only looks for neighbors within the unit cell
        :param group_neighbors: Bool. If True, only shows neighbors with the same bond distance once
        :param disp: Bool. If True, prints details of the calcualtion
        :param return_str: Bool. If True, returns a formatted string of the results
        :return: exchange_path, excange_distance (, output_str)
        """

        if cen_idx is None:
            # Select magnetic ions
            mag_idx = np.where(fg.mag(self.xtl.Structure.mxmymz()) > 0.01)[0]
            if len(mag_idx) == 0:
                print('No Magnetic ions')
                if return_str:
                    return "No Magnetic ions"
                return [], []
            cen_idx = mag_idx[0]

        if nearest_neighbor_distance is None:
            # Default to average cell length (does this any make sense?)
            nearest_neighbor_distance = np.mean(self.xtl.Cell.lp()[:3])

        cen_uvw = self.xtl.Structure.uvw()[cen_idx, :]
        cen_xyz = self.xtl.Cell.calculateR(cen_uvw)[0]
        cen_type = self.xtl.Structure.type[cen_idx]

        if disp:
            print('Centre: %3d %4s (%6.2f,%6.2f,%6.2f)' % (cen_idx, cen_type, cen_uvw[0], cen_uvw[1], cen_uvw[2]))

        # Generate lattice of muliple cells to remove cell boundary problem
        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.generate_lattice(1, 1, 1)
        xyz = self.xtl.Cell.calculateR(uvw)
        all_bonds = cen_xyz - xyz

        # Exchange type
        # exchange_type = 1  # all atoms
        exchange_allowed = atom_type == exchange_type

        # Inside unit cell
        if search_in_cell:
            incell = np.all((uvw >= 0) * (uvw <= 1), axis=1)
        else:
            incell = 1

        # Bond distances
        mag = fg.mag(all_bonds)
        neighbor_idx = np.where(incell * (atom_type == cen_type) * (mag < nearest_neighbor_distance) * (mag > 0.01))[0]

        # Sort order by distance to central atom
        srt_idx = np.argsort(mag[neighbor_idx])
        neighbor_idx = neighbor_idx[srt_idx]

        # Goup distances
        if group_neighbors:
            group_val, array_index, group_index, group_count = fg.group(mag[neighbor_idx], 0.001)
            neighbor_idx = neighbor_idx[group_index]

        exchange_paths = []
        exchange_distances = []
        for nidx in neighbor_idx:
            pos = xyz[nidx, :]
            n_uvw = uvw[nidx, :]
            n_type = label[nidx]
            n_dist = mag[nidx]
            if disp:
                print('\nNeighbour:')
                print('%3d %4s (%6.2f,%6.2f,%6.2f)' % (nidx, n_type, n_uvw[0], n_uvw[1], n_uvw[2]))
                print(' %s-%s distance: %5.3f Å' % (cen_type, n_type, n_dist))
            # Distance to average position
            av_pos = (cen_xyz + pos) / 2
            av_dis = fg.mag(av_pos - cen_xyz) * 1.1
            av_dis_idx = fg.mag(av_pos - xyz) < av_dis
            if disp:
                print(' %d ions < %5.2f A to bond' % (sum(av_dis_idx), av_dis))

            # Distace from Ru1
            dist1_idx = (mag > 0.01) * (mag < av_dis)
            if disp:
                print(' %d ions < %5.2f A to Ru1' % (sum(dist1_idx), av_dis))

            # Distance from Ru2
            all_bonds2 = pos - xyz
            mag2 = fg.mag(all_bonds2)
            dist2_idx = (mag2 > 0.01) * (mag2 < av_dis)
            if disp:
                print(' %d ions < %5.2f A to Ru2' % (sum(dist2_idx), av_dis))

            # bond angle from ru1
            bond = cen_xyz - pos
            bond_angles1 = np.array([fg.ang(bond, bnd) for bnd in all_bonds])
            ang1_idx = abs(bond_angles1) < np.deg2rad(bond_angle)
            if disp:
                print(' %d ions < 30Deg to Ru1' % (sum(ang1_idx)))
                print(' %d close ions < 30Deg to Ru1' % (sum(ang1_idx*dist1_idx)))
                for nn in np.where(dist1_idx)[0]:
                    print('   %4s (%6.2f,%6.2f,%6.2f) angle = %5.2f Deg' % (
                        label[nn], uvw[nn, 0], uvw[nn, 1], uvw[nn, 2], np.rad2deg(bond_angles1[nn])))

            # bond angle from ru2
            bond = pos - cen_xyz
            bond_angles2 = np.array([fg.ang(bond, bnd) for bnd in all_bonds2])
            ang2_idx = abs(bond_angles2) < np.deg2rad(bond_angle)
            if disp:
                print(' %d ions < 30Deg to Ru2' % (sum(ang2_idx)))
                print(' %d close ions < 30Deg to Ru2' % (sum(ang2_idx * dist2_idx)))
                for nn in np.where(dist2_idx)[0]:
                    print('   %4s (%6.2f,%6.2f,%6.2f) angle = %5.2f Deg' % (
                        label[nn], uvw[nn, 0], uvw[nn, 1], uvw[nn, 2], np.rad2deg(bond_angles2[nn])))

            # Atoms near bond
            exchange_idx = av_dis_idx * (dist1_idx + dist2_idx) * ang1_idx * ang2_idx * exchange_allowed
            near_bond_idx = np.where(exchange_idx)[0]
            if disp:
                print('Exchange ions: %d' % len(near_bond_idx))

            # Sort by distance to central atom
            srt_idx = np.argsort(mag[exchange_idx])
            near_bond_idx = near_bond_idx[srt_idx]

            exchange_dist = 0.
            exchange_paths += [[[cen_idx, cen_type, cen_xyz]]]
            for idx in near_bond_idx:
                near_uvw = uvw[idx, :]
                near_xyz = xyz[idx, :]
                near_type = label[idx]

                if disp:
                    dist12 = '%4sRu1: %6.2f' % (near_type, mag[idx])
                    dist21 = '%4sRu2: %6.2f' % (near_type, mag2[idx])
                    ang12 = 'ang1: %3.0fDeg' % np.rad2deg(bond_angles1[idx])
                    ang21 = 'ang2: %3.0fDeg' % np.rad2deg(bond_angles2[idx])
                    print('   %4s (%6.2f,%6.2f,%6.2f) %s %s %s %s' % (
                        near_type, near_uvw[0], near_uvw[0], near_uvw[0],
                        dist12, dist21, ang12, ang21
                    ))
                exchange_dist += fg.mag(near_xyz - exchange_paths[-1][-1][2])
                exchange_paths[-1] += [[idx, near_type, near_xyz]]
            exchange_dist += fg.mag(pos - exchange_paths[-1][-1][2])
            exchange_paths[-1] += [[nidx, n_type, pos]]
            exchange_distances += [exchange_dist]
            if disp:
                print('Exchange dist: %5.3f Å' % exchange_dist)

        outstr = 'Exchange Paths from [%3d] %4s (%5.2f,%5.2f,%5.2f) < %1.2f Å\n' % (
            cen_idx, cen_type, cen_uvw[0], cen_uvw[1], cen_uvw[2], nearest_neighbor_distance)

        # Print results
        for ex, dis in zip(exchange_paths, exchange_distances):
            str_exchange_path = '.'.join([e[1] for e in ex])
            n_uvw = uvw[ex[-1][0], :]
            str_neigh = '%4s(%5.2f,%5.2f,%5.2f)' % (ex[-1][1], n_uvw[1], n_uvw[0], n_uvw[2])
            bond_dist = mag[ex[-1][0]]
            outstr += '%s %12s BondDist=%5.3f Å. ExchangeDist=%5.3f Å\n' % (str_neigh, str_exchange_path, bond_dist, dis)
        if disp:
            print('\n\n%s' % outstr)
        if return_str:
            return exchange_paths, exchange_distances, outstr
        return exchange_paths, exchange_distances

    def latex_table(self):
        """Return latex table of structure properties from CIF"""
        return fc.cif2table(self.xtl.cif)

    def beamline_info(self, energy_kev=None):
        """Prints various properties useful for experiments"""

        out = '\n'
        out += '-----------%s-----------\n' % self.xtl.name
        out += ' Formula: {}\n'.format(self.molname())
        out += 'Magnetic: {}\n'.format(self.xtl.Structure.ismagnetic())
        out += '  Weight: %5.2f g/mol\n' % (self.weight())
        out += ' Density: %5.2f g/cm^3\n' % (self.density())
        # Symmetry
        out += '\nSymmetry: %r\n' % self.xtl.Symmetry
        # Cell info
        out += self.xtl.Cell.info()

        if energy_kev is not None:
            out += '\n    Energy = %7.4f keV\n' % energy_kev
            out += 'Wavelength = %6.3f A\n' % fc.energy2wave(energy_kev)
            # Xray edges
            out += '\nNear X-Ray Edges:\n'
            for edge_str, edge_en in zip(*self.xray_edges()):
                if abs(edge_en - energy_kev) < 1:
                    out += '    %5s : %7.4f keV\n' % (edge_str, edge_en)
            # Absorption
            out += '   Absorption coef: {} um^-1\n'.format(self.absorption(energy_kev))
            out += 'Attenuation length: {} um\n'.format(self.xray_attenuation_length(energy_kev))
            out += '  |Q| @ tth=180deg: {} A^-1\n'.format(fc.calqmag(180, energy_kev))
            out += '    d @ tth=180deg: {} A\n'.format(fc.caldspace(180, energy_kev))
            out += '           Max HKL: ({:d},{:d},{:d})\n'.format(*self.xtl.Cell.max_hkl(energy_kev, 180))
            allhkl = self.xtl.Scatter.get_hkl(True, False, energy_kev=energy_kev)
            out += '          No. Refs: {}\n'.format(len(allhkl))
        else:
            # Xray edges
            out += '\nX-Ray Edges:\n'
            for edge_str, edge_en in zip(*self.xray_edges()):
                out += '    %2s : %7.4f keV\n' % (edge_str, edge_en)
        out += '\n'
        return out

    def info(self):
        """Prints various properties of the crystal"""

        out = '\n'
        out += '-----------%s-----------\n'%self.xtl.name
        out += ' Weight: %5.2f g/mol\n' %(self.weight())
        out += ' Volume: %5.2f A^3\n' %(self.volume())
        out += 'Density: %5.2f g/cm^3\n' %(self.density())
        out += '\nAtoms:\n'
        types = np.unique(self.xtl.Structure.type)
        props = fc.atom_properties(types) # returns a numpy structured array
        prop_names = props.dtype.names
        for key in prop_names:
            ele = '%20s :' % key
            ele += ''.join([' %10s :' %(item) for item in props[key]])
            out += ele + '\n'
        return out

    def __repr__(self):
        return self.info()


class Element:
    """
    Element Class
    """
    def __init__(self, element='Co'):
        """Initialise properties"""

        props = fc.atom_properties(element) # returns a numpy structured array
        prop_names = props.dtype.names
        self.properties = props
        for key in prop_names:
            setattr(self, key, props[key][0])

    def info(self):
        """Display atomic properties"""

        prop_names = self.properties.dtype.names
        out = ''
        for key in prop_names:
            out += '%20s : %s\n' % (key, self.properties[key][0])
        return out

    def __repr__(self):
        return self.info()