# -*- coding: utf-8 -*-
"""
Properties Class "classes_properties.py"
 functions for calculating various properties, working with the Crystal class.

By Dan Porter, PhD
Diamond
2017

Version 1.3
Last updated: 15/08/19

Version History:
10/11/17 0.1    Program created
06/01/18 1.0    Program renamed
11/03/18 1.1    Added properties.info(), element.info()
23/02/19 1.2    Added xray_edges
15/08019 1.3    Added molcharge


@author: DGPorter
"""

import numpy as np

from . import functions_general as fg
from . import functions_crystallography as fc

__version__ = '1.3'


class Properties:
    """
    Properties functions for the Crystal Object
    """
    def __init__(self, xtl):
        """initialise"""
        self.xtl = xtl

        types = np.unique(self.xtl.Structure.type)
        for element in types:
            setattr(self,str(element),Element(str(element)))

    def volume(self):
        """Returns the volume in A^3"""
        return self.xtl.Cell.volume()

    def density(self):
        """Return the density in g/cm"""

        vol = self.xtl.Cell.volume()*1e-24 # cm^3
        weight = self.xtl.Structure.weight()/fg.Na # g
        return weight/vol

    def weight(self):
        """Return the molecular weight in g/mol"""
        return self.xtl.Structure.weight()

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
            atom = getattr(self, element)
            for edge in edges:
                energy = getattr(atom, edge)
                if energy > 0.1:
                    out_str += ['%s %s'%(element, edge)]
                    out_eng += [energy]
        return out_str, out_eng

    def molfraction(self, Z=1):
        """
        Display the molecular weight of a compound and atomic fractions
        :param Z: filling number
        :return: str
        """

        type = self.xtl.Structure.type
        occ = self.xtl.Structure.occupancy

        # Count elements
        ats = np.unique(type)
        weights = fc.atom_properties(ats,'Weight')
        atno = np.zeros(len(ats))
        for n,element in enumerate(ats):
            atno[n] = sum([occ[m] for m,x in enumerate(type) if x == element])

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

        type = np.asarray(self.xtl.Structure.type)
        occ = np.asarray(self.xtl.Structure.occupancy)

        # Count elements
        ats = np.unique(type)
        ats = fc.arrange_atom_order(ats)  # arrange this by alcali/TM/O
        atno = {}
        for a in ats:
            atno[a] = sum([occ[n] for n,x in enumerate(type) if x == a])

        # default - set max Z atom to it's occupancy
        if element is None:
            z = fc.atom_properties(ats, 'Z')
            element = ats[ np.argmax(z) ]
            #element = min(atno, key=atno.get)

        if element_no is None:
            element_no = np.max(occ[type == element])

        divideby = atno[element]/ element_no
        name = fc.count_atoms(type, occ, divideby, latex)
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

        type = np.asarray(self.xtl.Structure.type)
        occ = np.asarray(self.xtl.Structure.occupancy)

        # Count elements
        ats = np.unique(type)
        ats = fc.arrange_atom_order(ats)  # arrange this by alcali/TM/O
        atno = {}
        for a in ats:
            atno[a] = sum([occ[n] for n,x in enumerate(type) if x == a])

        # default - set max Z atom to it's occupancy
        if element is None:
            z = fc.atom_properties(ats, 'Z')
            element = ats[ np.argmax(z) ]
            #element = min(atno, key=atno.get)

        if element_no is None:
            element_no = np.max(occ[type == element])

        divideby = atno[element]/ element_no
        name = fc.count_charges(type, occ, divideby, latex)
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
        Z = fc.atom_properties(self.xtl.Structure.type ,'Z')
        upi = np.array([fc.attenuation(Zn, energy_kev) for Zn in Z])
        up = np.sum(wi*upi)

        u = up*self.density()/10000
        return u

    def diamagnetic_susceptibility(self, type='volume'):
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

        C = (fg.u0*fg.e**2)/(6*fg.me)
        V = self.xtl.Cell.volume()*1e-30 # m^3
        M = self.xtl.Structure.weight()/fg.Na # g
        mol = 1/fg.Na

        # susceptibility type
        if type.lower() == 'mass':
            D = M/1000 # kg
        elif type.lower() == 'cgs mass':
            D = M*(4*fg.pi)
        elif type.lower() == 'molar':
            D = mol
        elif type.lower() == 'cgs molar':
            D = mol*(4*fg.pi*1e-6)
        elif type.lower() == 'cgs volume':
            D = V*(4*fg.pi)
        else:
            D = V

        atom_types = np.unique(self.xtl.Structure.type)
        X = 0.0
        for at in atom_types:
            N = np.sum(self.xtl.Structure.occupancy[self.xtl.Structure.type==at])
            Zeff = fc.atom_properties(str(at),'ValenceE')[0]
            r = fc.atom_properties(str(at),'Radii')[0]*1e-12
            X += -(N/D)*C*(Zeff*r*r)
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

        Rindex = self.xtl.Cell.calculateR(self.xtl.Structure.uvw()[structure_index,:])

        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.generate_lattice(1,1,1)

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
                print('{:3.0f} {:4s} {:6.3f} {:6.3f} {:6.3f} dist = {:6.3f}'.format(i,label[i],diff[i,0],diff[i,1],diff[i,2],mag[i]))
        return diff[ii,:],label[ii]

    def info(self):
        """Prints various properties of the crystal"""

        print('-----------%s-----------'%self.xtl.name)
        print(' Weight: %5.2f g/mol' %(self.weight()))
        print(' Volume: %5.2f A^3' %(self.volume()))
        print('Density: %5.2f g/cm' %(self.density()))
        print('\nAtoms:')
        types = np.unique(self.xtl.Structure.type)
        props = fc.atom_properties(types) # returns a numpy structured array
        prop_names = props.dtype.names
        for key in prop_names:
            out = '%20s :' %(key)
            out += ''.join([' %10s :' %(item) for item in props[key]])
            print(out)


class Element:
    """
    Element Class
    """
    def __init__(self,element='Co'):
        """Initialise properties"""

        props = fc.atom_properties(element) # returns a numpy structured array
        prop_names = props.dtype.names
        self.properties = props
        for key in prop_names:
            setattr(self,key,props[key][0])

    def info(self):
        """Display atomic properties"""

        prop_names = self.properties.dtype.names
        for key in prop_names:
            print('%20s : %s' %(key,self.properties[key][0]))
