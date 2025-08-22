# -*- coding: utf-8 -*-
"""
Properties Class "classes_properties.py"
 functions for calculating various properties, working with the Crystal class.

By Dan Porter, PhD
Diamond
2017

Version 2.0
Last updated: 19/02/24

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
16/05/23 1.8    Fixed reflectivity calculation
20/07/23 1.9    Added FDMNES indata writer
19/02/24 2.0    Added relative_positions function

@author: DGPorter
"""

import numpy as np

from . import functions_general as fg
from . import functions_crystallography as fc
from . import functions_scattering as fs
from .classes_orbitals import CrystalOrbitals

__version__ = '2.0'


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
        return fc.magnetic_form_factor(*self.xtl.Structure.type, qmag=qmag)

    def scattering_factors(self, scattering_type, hkl, energy_kev=None,
                           use_sears=False, use_wasskirf=False):
        """
        Return an array of scattering factors based on the radiation
        :param scattering_type: str radiation, see "get_scattering_function()"
        :param hkl: [mx1] or None, float array of wavevector magnitudes for reflections
        :param energy_kev: [ox1] or None, float array of energies in keV
        :param use_sears: if True, use neutron scattering lengths from ITC Vol. C, By V. F. Sears
        :param use_wasskirf: if True, use x-ray scattering factors from Waasmaier and Kirfel
        :return: [nxmxo] array of scattering factors
        """
        qmag = self.xtl.Cell.Qmag(hkl)
        # Scattering factors
        ff = fs.scattering_factors(
            scattering_type=scattering_type,
            atom_type=self.xtl.Structure.type,
            qmag=qmag,
            enval=energy_kev,
            use_sears=use_sears,
            use_wasskirf=use_wasskirf,
        )
        return np.squeeze(ff)

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

    def resonant_element(self):
        """
        Returns the likely x-ray absorbing element in this material
          Returns the first d or f block element,
          otherwise returns the first element
        """
        elements = np.unique(self.xtl.Structure.type)
        # use the first d or f block element
        block = fc.atom_properties(elements, 'Block')
        if 'd' in block:
            absorber = elements[block == 'd'][0]
        elif 'f' in block:
            absorber = elements[block == 'f'][0]
        else:
            # otherwise use first element
            absorber = elements[0]
        return absorber

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
        weights = fc.atom_properties(ats, 'Weight')
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
            atno[a] = sum([occ[n] for n, x in enumerate(atom_type) if x == a])

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
        From: https://xdb.lbl.gov/Section4/Sec_4-2.html
        :param energy_kev: float array
        :param grazing_angle: incidence angle relative to the surface, in degrees
        :return: float or array
        """
        if energy_kev is None:
            energy_kev = fc.getenergy()

        atom_type = self.xtl.Structure.type
        occ = self.xtl.Structure.occupancy
        natoms = np.sum(occ)
        vol = self.volume()
        atom_per_volume = natoms / vol  # atoms per A^3
        elements = ['%s%s' % (at, o) for at, o in zip(atom_type, occ)]
        return fc.xray_reflectivity(elements, energy_kev, atom_per_volume, grazing_angle)

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

    def relative_positions(self, position, min_distance=0., max_distance=3.2, elements=None):
        """
        Returns atomic positions relative to given position
            rel_pos, atom_type, label, occ, uiso, mxmymz = Properties.relative_positions([0, 0, 0])
            rel_distance = np.sqrt(np.sum(np.square(rel_pos), axis=1))

        Note: all output positions are sorted in increasing order of distance from position.

        :param position: [x,y,z] position vector in Angstroms
        :param min_distance: min distance to find atoms, in Angstroms
        :param max_distance: max distance to find atoms, in Angstroms
        :param elements: str or list of str, atom types to return
        :return relative_position: [[dx,dy,dz], ...] relative positions in Angstroms
        :return atom_type: ['El', ..] elements
        :return label: ['El1', ..] site labels
        :return occ: [occ, ..] occupancy
        :return uiso: [uiso, ..] Uiso parameters
        :return mxmymz: [[mx,my,mz], ..] magnetic moments
        """

        position = np.reshape(position, 3)

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.generate_lattice(1, 1, 1)

        if elements is not None:
            elements = np.reshape(elements, -1)  # convert to str array
            ele_idx = np.zeros(len(atom_type), dtype=bool)
            for element in elements:
                ele_idx += atom_type == element
            uvw = uvw[ele_idx, :]
            atom_type = atom_type[ele_idx]
            label = label[ele_idx]
            occ = occ[ele_idx]
            uiso = uiso[ele_idx]
            mxmymz = mxmymz[ele_idx, :]

        R = self.xtl.Cell.calculateR(uvw)

        diff = R - position

        mag = fg.mag(diff)
        jj = np.argsort(mag)
        diff = diff[jj, :]
        label = label[jj]
        mag = mag[jj]
        atom_type = atom_type[jj]
        occ = occ[jj]
        uiso = uiso[jj]
        mxmymz = mxmymz[jj, :]
        ii = np.flatnonzero((mag > min_distance) * (mag < max_distance))

        return diff[ii, :], atom_type[ii], label[ii], occ[ii], uiso[ii], mxmymz[ii, :]

    def atomic_neighbours(self, structure_index=0, radius=2.5, disp=False, return_str=False):
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
        ii = np.where(mag < radius)[0]

        if disp or return_str:
            s = 'idx Label x      y      z     \n'
            fmt = '{:3.0f} {:4s} {:6.3f} {:6.3f} {:6.3f} dist = {:6.3f} \u212B'
            s += '\n'.join([
                fmt.format(jj[i], label[i], diff[i, 0], diff[i, 1], diff[i, 2], mag[i]) for i in ii
            ])
            if disp:
                print(s)
            if return_str:
                return s
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

    def fdmnes_runfile(self, output_path=None, comment='', energy_range=None, radius=4.0, edge='K',
                       absorber=None, green=True, scf=False, quadrupole=False, magnetism=False, spinorbit=False,
                       azi_ref=(1, 0, 0), correct_azi=False, hkl_reflections=(1, 0, 0)):
        """
        Write FDMNES run file
        :param output_path: Specify the output filename, e.g. 'Sim/Crystal/out'
        :param comment: A comment written in the input file
        :param energy_range: str energy range in eV relative to Fermi energy
        :param radius: calculation radius
        :param edge: absorptin edge, 'K', 'L3', 'L2', 'L23'
        :param absorber: absorbing element, 'Co'
        :param green: True/False, Green's function (muffin-tin potential)
        :param scf: True/False, Self consistent solution
        :param quadrupole: False/True, E1E2 terms allowed
        :param magnetism: False/True, allow magnetic calculation
        :param spinorbit: False/True, allow magnetic calculation with spin-orbit coupling
        :param azi_ref: azimuthal reference, [1,0,0]
        :param correct_azi: if True, correct azimuthal reference for real cell (use in hexagonal systems)
        :param hkl_reflections: list of hkl reflections [[1,0,0],[0,1,0]]
        :return: None
        """

        if output_path is None:
            output_path = 'Sim/%s/out' % fg.saveable(self.xtl.name)

        # Get crystal parameters
        uvw, element, label, occupancy, uiso, mxmymz = self.xtl.Structure.get()
        # Convert magnetic structure to euler angles
        mag_r, mag_theta, mag_phi = fg.cart2sph(mxmymz, True).T

        # Lattice parameters
        a, b, c, alpha, beta, gamma = self.xtl.Cell.lp()

        if absorber is None:
            absorber = self.resonant_element()
        absorber_idx = np.where(element == absorber)[0]
        nonabsorber_idx = np.where(element != absorber)[0]

        if correct_azi:
            UV = self.xtl.Cell.UV()
            UVs = self.xtl.Cell.UVstar()
            # sl_ar = np.dot(np.dot(azi_ref, UVs), np.linalg.inv(UVs))  # Q*/UV*
            fdm_ar = np.dot(np.dot(azi_ref, UVs), np.linalg.inv(UV))  # Q*/UV
            fdm_ar = fdm_ar / np.sqrt(np.sum(fdm_ar ** 2))  # normalise length to 1
        else:
            fdm_ar = azi_ref

        if energy_range is None:
            energy_range = '-19. 1. -5 0.1 -2. 0.05 5 0.1 10. 0.5 25 1 31. '

        if scf:
            SCF = ''
        else:
            SCF = '!'

        if quadrupole:
            quadrupole = ''
        else:
            quadrupole = '!'

        if green:
            green = ''
        else:
            green = '!'

        if magnetism:
            mag = ' Magnetism                    ! performs magnetic calculations\n'
            spin = True
        elif spinorbit:
            mag = ' Spinorbit                    ! performs magnetic calculations with spin orbit coupling\n'
            spin = True
        else:
            mag = '! magnetism                    ! performs magnetic calculations\n'
            spin = False

        param_string = ''

        # Write top matter
        param_string += '! FDMNES indata file\n'
        param_string += '! {}\n'.format(self.xtl.name)
        param_string += '! {}\n'.format(comment)
        param_string += '! indata file generated by crystal_diffraction.classes_properties\n'
        param_string += '\n'

        # Calculation Parameters
        param_string += ' Filout\n'
        param_string += '   {}\n\n'.format(output_path)
        param_string += '  Range                       ! Energy range of calculation (eV). Energy of photoelectron relative to Fermi level.\n'
        param_string += ' %s \n\n' % energy_range
        param_string += ' Radius                       ! Radius of the cluster where final state calculation is performed\n'
        param_string += '   {:3.1f}                        ! For a good calculation, this radius must be increased up to 6 or 7 Angstroems\n\n'.format(radius)
        param_string += ' Edge                         ! Threshold type\n'
        param_string += '  {}\n\n'.format(edge)
        param_string += '%s SCF                          ! Self consistent solution\n' % SCF
        param_string += '%s Green                        ! Muffin tin potential - faster\n' % green
        param_string += '%s Quadrupole                   ! Allows quadrupolar E1E2 terms\n' % quadrupole
        param_string += mag
        param_string += ' Density                      ! Outputs the density of states as _sd1.txt\n'
        param_string += ' Sphere_all                   ! Outputs the spherical tensors as _sph_.txt\n'
        param_string += ' Cartesian                    ! Outputs the cartesian tensors as _car_.txt\n'
        param_string += ' energpho                     ! output the energies in real terms\n'
        param_string += ' Convolution                  ! Performs the convolution\n\n'

        # Azimuthal reference
        param_string += ' Zero_azim                    ! Define basis vector for zero psi angle\n'
        param_string += '  {:6.3g} {:6.3g} {:6.3g}        '.format(fdm_ar[0], fdm_ar[1], fdm_ar[2])
        if correct_azi:
            param_string += '! Same as I16, Reciprocal ({} {} {}) in units of real SL. \n'.format(azi_ref[0], azi_ref[1], azi_ref[2])
        else:
            param_string += '\n'

        # Reflections
        hkl_reflections = np.reshape(hkl_reflections, [-1, 3])
        param_string += 'rxs                           ! Resonant x-ray scattering at various peaks, peak given by: h k l sigma pi azimuth.\n'
        for hkl in hkl_reflections:
            param_string += ' {} {} {}    1 1                 ! ({} {} {}) sigma-sigma\n'.format(hkl[0], hkl[1], hkl[2],
                                                                                                 hkl[0], hkl[1], hkl[2])
            param_string += ' {} {} {}    1 2                 ! ({} {} {}) sigma-pi\n'.format(hkl[0], hkl[1], hkl[2],
                                                                                              hkl[0], hkl[1], hkl[2])
        param_string += ' \n'

        # Atom electronic configuration
        orbitals = self.xtl.Properties.orbitals()
        param_string += ' Atom ! s=0,p=1,d=2,f=3, must be neutral, get d states right by moving e to 2s and 2p sites\n'
        orb_str, ele_list = orbitals.generate_string_fdmnes_absorber(absorber, spin)
        param_string += orb_str
        param_string += ' \n\n'

        # Atom positions
        param_string += ' Crystal                      ! Periodic material description (unit cell)\n'
        param_string += ' {:9.5f} {:9.5f} {:9.5f} {:9.5f} {:9.5f} {:9.5f}\n'.format(a, b, c, alpha, beta, gamma)
        param_string += '! Coordinates - 1st atom is the absorber\n'
        # Write atomic coordinates
        fmt = '{0:2.0f} {1:20.15f} {2:20.15f} {3:20.15f} ! {4:-3.0f} {5:2s}\n'
        fmag = ' {:3.2f} {:3.2f}  ! moment ({},{},{})\n'
        for n in absorber_idx:
            if spin and mag_r[n] > 0:
                # Add magnetic moment axis in Eulerian angles phi(z), theta(y)
                param_string += fmag.format(mag_phi[n], mag_theta[n], mxmymz[n, 0], mxmymz[n, 1], mxmymz[n, 2])
            ele_index = ele_list.index(element[n])
            param_string += fmt.format(ele_index + 1, uvw[n, 0], uvw[n, 1], uvw[n, 2], n, element[n])
        for n in nonabsorber_idx:
            if spin and mag_r[n] > 0:
                param_string += fmag.format(mag_phi[n], mag_theta[n], mxmymz[n, 0], mxmymz[n, 1], mxmymz[n, 2])
            ele_index = ele_list.index(element[n])
            param_string += fmt.format(ele_index + 1, uvw[n, 0], uvw[n, 1], uvw[n, 2], n, element[n])
        param_string += '\n'

        # Write end matter
        param_string += ' End\n'
        return param_string

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