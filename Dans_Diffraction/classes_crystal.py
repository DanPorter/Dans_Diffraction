"""
classes_crystal.py
A crystal object that reads crystallogaphic data from cif files and
can generate useful information such as reflection intensities and 
two-theta angles.

E.G. 
    f = 'Folder/Diamond.cif'
    xtl = Crystal(f)
    xtl.info() >> print information about the crystal
    
    Crystal properties are stored within lower classes:
        xtl.Cell     >> lattice parameters
        xtl.Symmetry >> Symmetry operations
        xtl.Atoms    >> Symmetric atomic positions
        xtl.Structure>> All atomic positions within the cell
    
    Additional calculations can be made within additional classes:
        xtl.Properties >> Calculate and display useful properties
        xtl.Plot       >> Plot Crystal structures and simulate diffraction patterns
        xtl.Scatter    >> Simulate diffraction intensities of different types

By Dan Porter, PhD
Diamond
2017

Version 3.3.1
Last updated: 06/04/25

Version History:
27/07/17 1.0    Version History started.
30/10/17 1.1    Many minor updates.
06/01/18 2.0    Name change and other updates
13/02/18 2.1    Move scattering commands to xtl.Scatter
05/04/18 2.2    Magnetic symmetry automatically inverted for odd time
04/06/18 2.3    removeatom added to Atom class
31/10/18 2.3    Update Symmetry.symmetric_coordinates funcitons, add ability to view all or only non-identical
09/03/19 2.4    Add print functions to Symmetry
12/08/19 2.5    self.info outputs string, __repr__ methods added
12/12/19 2.6    Added Symmetry.is_symmetric_reflection(ref1, ref2), added multiple scattering code
30/03/20 2.7    Moved Multicrystal class to separate file, other minor tweaks
19/04/20 2.8    Added update_cif and write_cif funcitons
12/05/20 2.9    Updated Atom.from_cif to be more reliable
27/05/20 3.0    Updated write_cif for magnetic moments - now writes simple mcif structures
09/06/20 3.0.1  Updated code for changes to fc.gen_sym_mat
10/06/20 3.1    Updated Symmetry to include time operators
02/09/20 3.2.0  Added Cell.reflection_hkl and transmission_hkl, added __str__ methods
22/10/20 3.2.1  Added Cell.moment, updated Cell.latt()
15/11/21 3.2.2  Added Cell.orientation, updated Cell.UV()
12/01/21 3.2.3  Added Symmetry.axial_vector
22/05/23 3.2.4  Added Symmetry.wyckoff_label(), Symmetry.spacegroup_dict
06/05/24 3.3.0  Symmetry.from_cif now loads operations from find_spacegroup if not already loaded
06/04/25 3.3.1  scale parameter of superlattice improved
15/09/25 3.3.2  Atoms.type changed to always be array type

@author: DGPorter
"""

import numpy as np
from warnings import warn

# Internal functions
from . import functions_general as fg
from . import functions_crystallography as fc
from . import functions_lattice as fl
from .classes_orientation import Orientation
from .classes_properties import Properties
# from .classes_orbitals import CrystalOrbitals
from .classes_scattering import Scattering
from .classes_multicrystal import MultiCrystal
from .classes_plotting import Plotting, PlottingSuperstructure

__version__ = '3.3.1'


class Crystal:
    """
    Reads the structure information from a cif file and generates the full structure.
    Allows the adjustment of the structure through the lattice parameters, symmetry 
    or atomic displacement.
    Can calculate reflection intensities and two-theta values.
    
    E.G.
      xtl = Crystal('Diamond.cif')
      xtl.Cell.lp() >> give the lattice parameters
      xtl.Atoms.uvw() >> give the symmetric atomic positions
      xtl.Symmetry.symmetry_operations >> give symmetry operations
      xtl.Structure.uvw() >> give the full, unsymmetrised structure
      xtl.Scatter.hkl([1,0,0],8.00) >> prints the intensity and two-theta of this reflection at this energy
      xtl.Scatter.print_all_reflections(8.00) >> print all allowed reflections, with intensities, at this energy
      xtl.write_cif('Diamond2.cif') >> write updated structure to file
    
    To create your own crystal (BCC example):
      xtl = Crystal()
      xtl.new_latt([2.866])
      xtl.new_atoms(u=[0,0.5],
                    v=[0,0.5],
                    w=[0,0.5],
                    type=['Fe','Fe'])
      xtl.hkl([[1,0,0],[1,1,0],[1,1,1],[2,0,0]])
    
    Also, see:
        help(xtl.Cell)
        help(xtl.Atoms)
        help(xtl.Symmetry)
        help(xtl.Scatter)
    """

    # Defaults
    filename = ''
    name = 'Crystal'
    cif = {}
    scale = 1.0

    def __init__(self, filename=None):

        # Instatiate crystal attributes
        self.Cell = Cell()
        self.Symmetry = Symmetry()
        self.Atoms = Atoms()
        self.Structure = Atoms()

        # Get data from cif file
        if filename is not None:
            # Read cif file
            cifvals = fc.readcif(filename)

            # Standard parameters
            self.filename = filename
            self.name = cifvals['FileTitle']
            self.cif = cifvals
            self.scale = 1.0

            # Generate attributes from cif file
            self.Cell.fromcif(cifvals)
            self.Symmetry.fromcif(cifvals)
            self.Atoms.fromcif(cifvals)

        # Generate the full crystal structure (apply symmetry operations to atomic positions)
        self.generate_structure()  # Creates self.Structure, an instance of Atoms

        # Add exta functions
        self.Plot = Plotting(self)
        self.Scatter = Scattering(self)
        self.Properties = Properties(self)
        # self.Orbitals = CrystalOrbitals(self) # slows down cif parsing a lot!
        # self.Fdmnes = Fdmnes(self)

    def generate_structure(self):
        """
        Combines the atomic positions with symmetry operations, returning the full structure as an Atoms class
        :return: None
        """

        uvw, atom_type, label, occ, uiso, mxmymz = self.Atoms.get()

        # Shortcut if only 1 symmetry operation
        if len(self.Symmetry.symmetry_operations) == 1:
            self.Structure = Atoms(uvw[:, 0], uvw[:, 1], uvw[:, 2], atom_type, label, occ, uiso, mxmymz)
            return

        new_uvw = np.empty([0, 3])
        new_type = []
        new_label = []
        new_occupancy = []
        new_uiso = []
        new_mxmymz = np.empty([0, 3])
        for n in range(len(uvw)):
            sympos, symmag = self.Symmetry.symmetric_coordinates(uvw[n], mxmymz[n])
            Nsympos = len(sympos)
            # symmag = self.MagSymmetry.symmetric_coordinates(mxmymz[n])
            # symmag = np.tile(mxmymz[n],Nsympos).reshape([-1,3])

            # Append to P1 atoms arrays
            new_uvw = np.append(new_uvw, sympos, axis=0)
            new_mxmymz = np.append(new_mxmymz, symmag, axis=0)
            new_type += [atom_type[n]] * Nsympos
            new_label += [label[n]] * Nsympos
            new_occupancy += [occ[n]] * Nsympos
            new_uiso += [uiso[n]] * Nsympos

        u = new_uvw[:, 0]
        v = new_uvw[:, 1]
        w = new_uvw[:, 2]
        new_type = np.array(new_type)
        new_label = np.array(new_label)
        new_occupancy = np.array(new_occupancy)
        new_uiso = np.array(new_uiso)

        self.Structure = Atoms(u, v, w, new_type, new_label, new_occupancy, new_uiso, new_mxmymz)

    def update_cif(self, cifvals=None):
        """
        Update self.cif dict with current values
        :param cifvals: cif dict from readcif (None to use self.cif)
        :return: cifvals
        """

        if cifvals is None:
            cifvals = self.cif

        # Update name
        cifvals['_pd_phase_name'] = self.name
        cifvals = self.Cell.update_cif(cifvals)
        cifvals = self.Symmetry.update_cif(cifvals)
        cifvals = self.Atoms.update_cif(cifvals)
        cifvals = self.Properties.update_cif(cifvals)

        self.cif = cifvals
        return cifvals

    def write_cif(self, filename=None, comments=None):
        """
        Write crystal structure to CIF (Crystallographic Information File)
         Only basic information is saved to the file, but enough to open in VESTA etc.
         If magnetic ions are defined, a magnetic cif (*.mcif) will be produce
        :param filename: name to write too, if None, use writes to self.name (.cif/.mcif)
        :param comments: str comments to add to file header
        :return: None
        """

        cifvals = self.update_cif()

        if filename is None:
            filename = '%s' % fg.saveable(self.name)
        cifvals['FileTitle'] = filename

        if self.Atoms.ismagnetic():
            fc.write_mcif(cifvals, filename, comments)
        else:
            fc.write_cif(cifvals, filename, comments)

    def new_cell(self, *lattice_parameters, **kwargs):
        """
        Replace the lattice parameters
        :param lattice_parameters: [a,b,c,alpha,beta,gamma]
        :return: None
        """
        self.Cell.latt(*lattice_parameters, **kwargs)

    def new_atoms(self, u=[0], v=[0], w=[0], type=None,
                  label=None, occupancy=None, uiso=None, mxmymz=None):
        """
        Replace current atomic positions with new ones and regenerate structure
        :param u: array : atomic positions u
        :param v: array : atomic positions v
        :param w:  array : atomic positions w
        :param type:  array : atomic types
        :param label: array : atomic labels
        :param occupancy: array : atomic occupation
        :param uiso: array : atomic isotropic thermal parameters
        :param mxmymz: array : atomic magnetic vectors [mu,mv,mw]
        :return: None
        """

        self.Atoms = Atoms(u, v, w, type, label, occupancy, uiso, mxmymz)
        self.generate_structure()

    def generate_lattice(self, U=1, V=1, W=0):
        """
        Generate a repeated lattice of the crystal structure
            latt = xtl.generate_lattice(2,0,0)
        :param U: Repeat of the cell along the a axis
        :param V: Repeat of the cell along the b axis
        :param W: Repeat of the cell along the c axis
        :return: Crystal object
        """

        uvw, type, label, occ, uiso, mxmymz = self.Structure.generate_lattice(U, V, W, centred=False)
        R = self.Cell.calculateR(uvw)
        lp = self.Cell.generate_lattice(U + 1, V + 1, W + 1)

        latt = Crystal()
        latt.new_cell(lp)

        UV = latt.Cell.UV()
        uvw = latt.Cell.indexR(R)
        latt.new_atoms(u=uvw[:, 0], v=uvw[:, 1], w=uvw[:, 2],
                       type=type, label=label, occupancy=occ, uiso=uiso)
        return latt

    def generate_superstructure(self, P):
        """
        Generate a superstructure of the current cell
            a' = n1a + n2b + n3c
            b' = m1a + m2b + m3c
            c' = o1a + o2b + o3c
                    OR
            [a',b',c'] = P[a,b,c]
                    Where
            P = [[n1,n2,n3],
                 [m1,m2,m3],
                 [o1,o2,o3]]
        Returns a superstructure Crystal class:
            su = xtl.generate_superstructrue([[2,0,0],[0,2,0],[0,0,1]])
        
        Superstructure Crystal classes have additional attributes:
            su.P = P as given
            su.Parent = the parent Crystal Class
        
        Use >>hasattr(su,'Parent') to check if the current object is a
        superstructure Crystal class
        """

        super = Superstructure(self, P)
        super.generate_super_positions()
        return super

    def transform(self, P):
        """
        Transform the current cell
            a' = n1a + n2b + n3c
            b' = m1a + m2b + m3c
            c' = o1a + o2b + o3c
                    OR
            [a',b',c'] = P[a,b,c]
                    Where
            P = [[n1,n2,n3],
                 [m1,m2,m3],
                 [o1,o2,o3]]
        Returns a superstructure Crystal class:
            su = xtl.transform([[0,1,0],[1,0,0],[0,0,1]])

        Superstructure Crystal classes have additional attributes:
            su.P = P as given
            su.Parent = the parent Crystal Class

        Use >>hasattr(su,'Parent') to check if the current object is a
        superstructure Crystal class
        """

        super = Superstructure(self, P)
        super.generate_super_positions()
        return super

    def invert_structure(self):
        """
        Convert handedness of structure, transform from left-handed to right handed, or visa-versa
        Equivlent to xtl.transform([[-1,0,0], [0,-1,0], [0,0,-1]])
        :return: Superstructure Crystal class
        """
        return self.transform([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])

    def add_parent(self, parent, P):
        """
        Add parent structure, returning Crystal as superstructure
            parent = Crystal(cif_parent)
            xtl = Crystal(cif_superstructure)
            su = xtl.add_parent(parent, [[1,1,0],[0,2,0],[0,0,1]])
        """

        uvw, atom_type, label, occupancy, uiso, mxmymz = self.Structure.get()
        latt = self.Cell.lp()

        super = Superstructure(parent, P)
        super.name = self.name
        super.new_cell(latt)
        super.new_atoms(uvw[:, 0], uvw[:, 1], uvw[:, 2], atom_type, label, occupancy, uiso, mxmymz)
        return super

    def start_gui(self):
        """
        Start Crystal GUI
        :return: None
        """
        try:
            from .tkgui import CrystalGui
            CrystalGui(self)
        except ImportError:
            print('Sorry, you need to install tkinter!')

    def search_distances(self, min_d=0.65, max_d=3.20, c_ele=None, elems=None,
                         labels=None, simple=True):
        """
        Calculated atoms interatomic distances form each label.
        :param c_ele (list,string): only sites with noted elements
                                    if None all site
        :param elems (list,string): only distances with noted elements
                                    if None all site
        :param min_d: minimum distance 
        :param max_d: maximum distance
        :return dictionary: 

        """
        xyz = self.Cell.calculateR(self.Atoms.uvw())
        UVstar = self.Cell.UVstar()
        UV = self.Cell.UV()
        Lpar = self.Cell.lp()[:3]

        Lran = np.array([0, 0, 0, 0, 0, 0])

        for i in range(3):
            Lran[i] = min(fg.distance2plane([0, 0, 0], UVstar[i], xyz))
        for i in range(3):
            Lran[3 + i] = min(fg.distance2plane(UV[i], UVstar[i] + UV[i], xyz))
        for i, L in enumerate(Lran):
            Lran[i] = np.ceil(max_d / Lpar[i % 3]) if L < max_d else 0

        IntRes = self.Structure.generate_lattice(U=Lran[0] + Lran[3],
                                                 V=Lran[1] + Lran[4],
                                                 W=Lran[2] + Lran[5],
                                                 centred=False)[:3]
        B_uvw, B_Ele, B_label = IntRes
        B_label, B_Ele = np.asarray(B_label), np.asarray(B_Ele)

        B_uvw -= np.array(Lran[:3])
        B_xyz = self.Cell.calculateR(B_uvw)

        if c_ele is None:
            c_ele = set(self.Atoms.type)
        elif isinstance(c_ele, str):
            c_ele = [c_ele]

        if elems is None:
            elems = set(self.Atoms.type)
        elif isinstance(elems, str):
            elems = [elems]

        if labels is None:
            labels = self.Atoms.label
        elif isinstance(labels, str):
            labels = [labels]

        distances = {}
        for i, atom in enumerate(xyz):
            if not self.Atoms.type[i] in c_ele:
                continue
            if not self.Atoms.label[i] in labels:
                continue
            s_dist = {}
            vdist = (B_xyz - atom) ** 2
            vdist = np.sqrt(np.sum(vdist, axis=1))
            cond1 = (vdist > min_d) * (vdist < max_d)
            vlabel = B_label[cond1]
            vele = B_Ele[cond1]
            vdist = vdist[cond1]

            Ord = np.argsort(vdist)
            cond2 = [ele in elems for ele in vele[Ord]]
            s_dist = {'dist': vdist[Ord][cond2],
                      'label': list(vlabel[Ord][cond2]),
                      'type': list(vele[Ord][cond2])}

            distances[self.Atoms.label[i]] = s_dist

        if simple:
            # reduce the output for mixed site occupation
            lab = [i for i,j in zip(self.Atoms.label, self.Atoms.type) if j in c_ele]
            for i, site_i in enumerate(lab):
                for site_j in lab[i + 1:]:
                    uvw_i = self.Atoms[site_i].uvw()
                    uvw_j = self.Atoms[site_j].uvw()
                    if all(uvw_i == uvw_j):
                        lab.remove(site_j)
                        try:
                            del distances[site_j]
                        except KeyError:
                            pass

        return distances

    def __add__(self, other):
        return MultiCrystal([self, other])

    def info(self):
        """
        Returns information about the crystal structure
        :return: str
        """
        out = '\n###########################################\n'
        out += '{}\n'.format(self.name)
        out += 'Formula: {}\n'.format(self.Properties.molname())
        out += 'Magnetic: {}\n'.format(self.Structure.ismagnetic())
        out += 'Spacegroup: %s\n' % repr(self.Symmetry)
        out += self.Cell.info()
        out += 'Density: %6.3f g/cm\n\n' % self.Properties.density()
        out += self.Structure.info()
        out += '\n'
        # print "To see the full list of structure positions, type Crystal.Structure.info()"
        return out

    def __repr__(self):
        if self.Structure.ismagnetic():
            nmom = np.sum(np.any(np.abs(self.Structure.mxmymz()) > 0, axis=1))
            fmt = "%s with %d atomic positions (%d magnetic), %d symmetries"
            return fmt % (self.name, len(self.Atoms.type), nmom, len(self.Symmetry.symmetry_operations))
        fmt = "%s with %d atomic positions, %d symmetries"
        return fmt % (self.name, len(self.Atoms.type), len(self.Symmetry.symmetry_operations))

    def __str__(self):
        return self.info()


class Cell:
    """
    Contains lattice parameters and unit cell
    Provides tools to convert between orthogonal and lattice bases in real and reciprocal space.
    
    E.G.
        UC = Cell() # instantiate the Cell object
        UC.latt([2.85,2.85,10.8,90,90,120]) # Define the lattice parameters from a list
        UC.tth([0,0,12],energy_kev=8.0) # Calculate the two-theta of a reflection
        UC.lp() # Returns the current lattice parameters
        UC.orientation # class to chanage cell orientation in space
    """
    _required_cif = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma"
    ]

    def __init__(self, a=1.0, b=1.0, c=1.0, alpha=90., beta=90.0, gamma=90.0):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._basis_function = fl.basis_3
        self.orientation = Orientation()

    def latt(self, *lattice_parameters, **kwargs):
        """ 
        Generate lattice parameters with list
          latt(1) -> a=b=c=1,alpha=beta=gamma=90
          latt([1,2,3]) -> a=1,b=2,c=3,alpha=beta=gamma=90
          latt([1,2,3,120]) -> a=1,b=2,c=3,alpha=beta=90,gamma=120
          latt([1,2,3,10,20,30]) -> a=1,b=2,c=3,alpha=10,beta=20,gamma=30
          latt(1,2,3,10,20,30) -> a=1,b=2,c=3,alpha=10,beta=20,gamma=30
          latt(a=1,b=2,c=3,alpha=10,beta=20,gamma=30]) -> a=1,b=2,c=3,alpha=10,beta=20,gamma=30
        """

        lp = fc.gen_lattice_parameters(*lattice_parameters, **kwargs)
        self.a = lp[0]
        self.b = lp[1]
        self.c = lp[2]
        self.alpha = lp[3]
        self.beta = lp[4]
        self.gamma = lp[5]

    def fromcif(self, cifvals):
        """
        Import lattice parameters from a cif dictionary
        Required CIF keys:
            _cell_length_a
            _cell_length_b
            _cell_length_c
            _cell_angle_alpha
            _cell_angle_beta
            _cell_angle_gamma
        :param cifvals: dict from readcif
        :return: None
        """
        if not fc.cif_check(cifvals, self._required_cif):
            warn('Lattice parameters cannot be read from cif')
            return

        a, da = fg.readstfm(cifvals['_cell_length_a'])
        b, db = fg.readstfm(cifvals['_cell_length_b'])
        c, dc = fg.readstfm(cifvals['_cell_length_c'])
        alpha, dalpha = fg.readstfm(cifvals['_cell_angle_alpha'])
        beta, dbeta = fg.readstfm(cifvals['_cell_angle_beta'])
        gamma, dgamma = fg.readstfm(cifvals['_cell_angle_gamma'])

        self.latt(a, b, c, alpha, beta, gamma)

    def update_cif(self, cifvals):
        """
        Update cif dict with current values
        :param cifvals: dict from readcif
        :return: cifvals
        """

        cifvals['_cell_length_a'] = '%1.6f' % self.a
        cifvals['_cell_length_b'] = '%1.6f' % self.b
        cifvals['_cell_length_c'] = '%1.6f' % self.c
        cifvals['_cell_angle_alpha'] = '%1.4g' % self.alpha
        cifvals['_cell_angle_beta'] = '%1.4g' % self.beta
        cifvals['_cell_angle_gamma'] = '%1.4g' % self.gamma

        cifvals['_cell_volume'] = '%1.4f' % self.volume()
        return cifvals

    def lp(self):
        """
        Returns the lattice parameters
        :return: a,b,c,alpha,beta,gamma
        """
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def choose_basis(self, option='default'):
        """
        Choose the basis function
        Options:
            1. c || z, b* || y - basis choice of Materials Project
            2. a || x, c* || z - basis choice of Vesta
            3. c || z, a* || x - basis choice of Busing & Levy (Default)
        :param option: name or number of basis
        """
        self._basis_function = fl.choose_basis(option)

    def UV(self):
        """
         Returns the unit cell as a [3x3] array, [A,B,C]
         The vector A is directed along the x-axis
        """
        return self.orientation(self._basis_function(*self.lp()))

    def UVstar(self):
        """
        Returns the reciprocal unit cell as a [3x3] array, [A*,B*,C*]
        :return: [a*;b*;c*]
        """
        return fl.reciprocal_basis(self.UV())

    def volume(self):
        """
        Returns the volume of the unit cell, in A^3
        :return: volume
        """
        return fl.lattice_volume(*self.lp())

    def Bmatrix(self):
        """
        Calculate the Busing and Levy B matrix from a real space UV
         "choose the x-axis parallel to a*, the y-axis in the plane of a* and b*, and the z-axis perpendicular to
         that plane"
        W. R. Busing & H. A. Levy, Acta  Cryst.  (1967). 22,  457
        """
        return fl.busingandlevy(*self.lp())

    def info(self):
        """
         Prints the lattice parameters and cell volume"
        :return: str
        """
        out = 'a = %6.3f A,  b = %6.3f A,  c = %6.3f A\nA = %8.2f,  B = %8.2f,  G = %8.2f' % self.lp()
        out += '\nVolume: %6.2f A^3\n' % self.volume()
        return out

    def __repr__(self):
        return 'Cell(a=%1.5g, b=%1.5f, c=%1.5g, alpha=%1.5g, beta=%1.5g, gamma=%1.5g)' % self.lp()

    def __str__(self):
        return self.info()

    def generate_lattice(self, U, V, W):
        """
         Returns the lattice parameters of a larger lattice
        """
        return U * self.a, V * self.b, W * self.c, self.alpha, self.beta, self.gamma

    def calculateQ(self, HKL):
        """
        Convert coordinates [h,k,l], in the basis of the reciprocal lattice, to
        coordinates [x,y,z], in an orthogonal basis, in units of A-1
                Q(x,y,z) = hA* + kB* + lC*
        
        E.G.
            Q = Cell.calculateQ([1,0,0]) # for a hexagonal system, a = 2.85
            > Q = array([[2.2046264, 1.2728417, 0.0000000]])
        """
        HKL = np.reshape(np.asarray(HKL, dtype=float), [-1, 3])
        return np.dot(HKL, self.UVstar())

    def indexQ(self, Q):
        """
        Convert coordinates [x,y,z], in an orthogonal basis, to
        coordinates [h,k,l], in the basis of the reciprocal lattice
                    H(h,k,l) = Q(x,y,z) / [A*,B*,C*]

        E.G.
            HKL = indexQ([2.2046264, 1.2728417, 0.0000000]) # for a hexagonal system, a = 2.85
            > HKL = [1,0,0]
        """
        Q = np.reshape(np.asarray(Q, dtype=float), [-1, 3])
        return fl.index_lattice(Q, self.UVstar())

    def calculateR(self, UVW):
        """
        Convert coordinates [u,v,w], in the basis of the unit cell, to
        coordinates [x,y,z], in an orthogonal basis, in units of A
                    R(x,y,z) = uA + vB + wC
        E.G.
            R = Cell.calculateR([0.1,0,0]) # for a hexagonal system, a = 2.85
            > R = array([[0.285, 0, 0]])
        """
        UVW = np.reshape(np.asarray(UVW, dtype=float), [-1, 3])
        return np.dot(UVW, self.UV())

    def indexR(self, R):
        """
        Convert coordinates [x,y,z], in an orthogonal basis, to
        coordinates [u,v,w], in the basis of the unit cell
                U(u,v,w) = R(x,y,z) / [A,B,C]
        
        E.G.
            UVW = indexR([0.285, 0, 0]) # for a hexagonal system, a = 2.85
            > UVW = [0.1,0,0]
        """
        R = np.reshape(np.asarray(R, dtype=float), [-1, 3])
        return fl.index_lattice(R, self.UV())

    def moment(self, mxmymz):
        """Calcualte moment from value stored in cif"""
        momentmag = fg.mag(mxmymz).reshape([-1, 1])
        momentxyz = self.calculateR(mxmymz)
        mom = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
        mom[np.isnan(mom)] = 0.
        return mom

    def Qmag(self, HKL):
        """
        Returns the magnitude of wave-vector transfer of [h,k,l], in A-1
        :param HKL: list of hkl reflections
        :return: list of Q values
        """
        Q = self.calculateQ(HKL)
        return fg.mag(Q)

    def tth(self, HKL, energy_kev=8.048, wavelength_a=None):
        """
        Returns the two-theta angle, in deg, of [h,k,l] at specified energy in keV
        :param HKL: list of hkl reflections
        :param energy_kev: photon energy in keV
        :param wavelength_a: wavelength in Angstroms
        :return: two-theta angles
        """
        Qmag = self.Qmag(HKL)
        return fc.cal2theta(Qmag, energy_kev, wavelength_a)

    def angle(self, hkl1, hkl2):
        """
        Return the angle between two reflections
        :param hkl1: [h,k,l] reflection 1
        :param hkl2: [h,k,l] reflection 2
        :return: angle in degrees
        """
        q1 = self.calculateQ(hkl1)
        q2 = self.calculateQ(hkl2)
        return np.abs(fg.ang(q1, q2, deg=True))

    def theta_reflection(self, HKL, energy_kev=8.048, specular=[0, 0, 1], theta_offset=0):
        """
        Calculate the sample angle for diffraction in reflection geometry given a particular specular direction
        """

        Q = self.calculateQ(HKL)
        tth = self.tth(HKL, energy_kev)
        angle = np.zeros(Q.shape[0])
        for n in range(Q.shape[0]):
            angle[n] = np.rad2deg(fg.ang(Q[n], specular)) + theta_offset
        return (tth / 2) + angle

    def theta_transmission(self, HKL, energy_kev=8.048, parallel=[0, 0, 1], theta_offset=0):
        """
        Calculate the sample angle for diffraction in transmission geometry given
        a particular direction parallel to the beam
        """

        Q = self.calculateQ(HKL)
        tth = self.tth(HKL, energy_kev)
        angle = np.zeros(Q.shape[0])
        for n in range(Q.shape[0]):
            angle[n] = np.rad2deg(fg.ang(Q[n], parallel)) - 90 + theta_offset
        return (tth / 2) + angle

    def dspace(self, hkl):
        """
        Calculate the d-spacing in A
        :param hkl: array : list of reflections
        :return: d-spacing
        """
        Qmag = self.Qmag(hkl)
        return fc.q2dspace(Qmag)

    def max_hkl(self, energy_kev=8.048, max_angle=180.0, wavelength_a=None, maxq=None):
        """
        Returns the maximum index of h, k and l for a given energy
        :param energy_kev: energy in keV
        :param max_angle: maximum two-theta at this energy
        :param wavelength_a: wavelength in A
        :param maxq: maximum wavevetor transfere in A-1 (suplants all above)
        :return: maxh, maxk, maxl
        """
        if maxq is None:
            maxq = fc.calqmag(max_angle, energy_kev, wavelength_a)
        Qpos = [[maxq, maxq, maxq],
                [-maxq, maxq, maxq],
                [maxq, -maxq, maxq],
                [-maxq, -maxq, maxq],
                [maxq, maxq, -maxq],
                [-maxq, maxq, -maxq],
                [maxq, -maxq, -maxq],
                [-maxq, -maxq, -maxq]]
        hkl = self.indexQ(Qpos)
        return np.ceil(np.abs(hkl).max(axis=0)).astype(int)

    def all_hkl(self, energy_kev=8.048, max_angle=180.0, wavelength_a=None, maxq=None):
        """
        Returns an array of all (h,k,l) reflections at this energy
        :param energy_kev: energy in keV
        :param max_angle: max two-theta angle
        :param wavelength_a: wavelength in A
        :param maxq: maximum wavevetor transfere in A-1 (suplants all above)
        :return: array hkl[:,3]
        """
        # Find the largest indices
        hmax, kmax, lmax = self.max_hkl(energy_kev, max_angle, wavelength_a, maxq)
        # Generate the grid
        HKL = fc.genHKL([hmax, -hmax], [kmax, -kmax], [lmax, -lmax])
        # Some will be above the threshold
        Qm = self.Qmag(HKL)
        if maxq is None:
            maxq = fc.calqmag(max_angle, energy_kev, wavelength_a)
        return HKL[Qm <= maxq, :]

    def reflection_hkl(self, energy_kev=8.048, max_angle=180.0,
                       specular=(0, 0, 1), theta_offset=0, min_theta=0, max_theta=180.):
        """
        Returns an array of all (h,k,l) reflections in reflection geometry
        :param energy_kev: energy in keV
        :param max_angle: max two-theta angle
        :param specular: (h,k,l) of direction normal to surface and the incident beam
        :param theta_offset: float : angle (deg) of surface relative to specular normal
        :param min_theta: float : cut hkl reflections with reflection-theta lower than min_theta
        :param max_theta: flaot : cut hkl reflections with reflection-theta greater than max_theta
        :return: array of hkl
        """
        hkl = self.all_hkl(energy_kev, max_angle)
        tth = self.tth(hkl, energy_kev)
        theta = self.theta_reflection(hkl, energy_kev, specular, theta_offset)

        p1 = (theta > min_theta) * (theta < max_theta)
        p2 = (tth > (theta + min_theta)) * (tth < (theta + max_theta))
        return hkl[p1 * p2, :]

    def transmission_hkl(self, energy_kev=8.048, max_angle=180.0,
                         parallel=(0, 0, 1), theta_offset=0, min_theta=0, max_theta=180.):
        """
        Returns an array of all (h,k,l) reflections in reflection geometry
        :param energy_kev: energy in keV
        :param max_angle: max two-theta angle
        :param parallel: (h,k,l) of direction normal to surface, parallel to the incident beam
        :param theta_offset: float : angle (deg) of surface relative to specular normal
        :param min_theta: float : cut hkl reflections with reflection-theta lower than min_theta
        :param max_theta: flaot : cut hkl reflections with reflection-theta greater than max_theta
        :return: array of hkl
        """
        hkl = self.all_hkl(energy_kev, max_angle)
        tth = self.tth(hkl, energy_kev)
        theta = self.theta_transmission(hkl, energy_kev, parallel, theta_offset)

        p1 = (theta > min_theta) * (theta < max_theta)
        p2 = (tth > (theta + min_theta)) * (tth < (theta + max_theta))
        return hkl[p1 * p2, :]

    def sort_hkl(self, hkl, ascend=True):
        """
        Returns array of (h,k,l) sorted by two-theta
        :param hkl: array : list of [h,k,l] values
        :param ascend: True*/False : if False, lowest two-theta
        :return: HKL[sorted,:]
        """

        hkl = np.reshape(np.asarray(hkl, dtype=float), [-1, 3])
        Qm = self.Qmag(hkl)
        idx = np.argsort(Qm)
        return hkl[idx, :]

    def powder_average(self, hkl):
        """
        Returns the powder average correction for the given hkl
        :param hkl: array : list of reflections
        :return: correction
        """
        q = self.Qmag(hkl)
        return 1 / (q + 0.001) ** 2

    def find_close_reflections(self, hkl, energy_kev, max_twotheta=2, max_angle=10):
        """
        Find reflections near to given HKL for a given two-theta or reflection angle
        :param hkl: [h,k,l] indices of reflection to start from
        :param energy_kev: energy in keV
        :param max_twotheta: matches reflections within two-theta of hkl
        :param max_angle: matches reflections within max_angle of hkl
        :return: list of matching [[h,k,l]] reflections
        """

        if max_twotheta is None:
            max_twotheta = 180.
        if max_angle is None:
            max_angle = 180.

        all_hkl = self.all_hkl(energy_kev, 180.)
        all_hkl = self.sort_hkl(all_hkl)

        all_tth = self.tth(all_hkl, energy_kev)
        tth1 = self.tth(hkl, energy_kev)
        tth_dif = np.abs(all_tth - tth1)

        all_Q = self.calculateQ(all_hkl)
        Q1 = self.calculateQ(hkl)
        all_angles = np.abs([fg.ang(Q1, Q2, 'deg') for Q2 in all_Q])

        selected = (tth_dif < max_twotheta) * (all_angles < max_angle)
        return all_hkl[selected, :]

    def reciprocal_space_plane(self, x_axis=(1, 0, 0), y_axis=(0, 1, 0), centre=(0, 0, 0), q_max=4.0, cut_width=0.05):
        """
        Returns positions within a reciprocal space plane
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
        
        Returns: X,Y,HKL
          Qx = [nx1] array of x positions for each HKL in the plane
          Qy = [nx1] array of y positions for each HKL in the plane
          HKL= [nx1] array of each HKL in the plane
        """

        # Determine the directions in cartesian space
        x_cart = self.calculateQ(x_axis)
        y_cart = self.calculateQ(y_axis)
        x_cart, y_cart, z_cart = fc.orthogonal_axes(x_cart, y_cart)
        c_cart = self.calculateQ(centre)

        # Generate lattice of reciprocal space points
        maxq = np.sqrt(q_max ** 2 + q_max ** 2)
        hmax, kmax, lmax = fc.maxHKL(maxq, self.UVstar())
        HKL = fc.genHKL([hmax, -hmax], [kmax, -kmax], [lmax, -lmax])
        HKL = np.ceil(HKL + centre)  # reflection about central reflection
        Q = self.calculateQ(HKL)

        # generate box in reciprocal space
        CELL = np.array([2 * q_max * x_cart, -2 * q_max * y_cart, cut_width * z_cart])

        box_coord = fg.index_coordinates(Q - c_cart, CELL)
        incell = np.all(np.abs(box_coord) <= 0.5, axis=1)
        if np.sum(incell) < 1:
            return np.zeros((0,)), np.zeros((0,)), np.zeros((0, 3))
        plane_coord = 2 * q_max * box_coord[incell, :]
        return plane_coord[:, 0], plane_coord[:, 1], HKL[incell, :]

    def ubmatrix(self):
        """Return UB matrix from Busing & Levy in the diffractometer frame"""
        return fc.ubmatrix(self.UV(), self.orientation.umatrix)

    def labwavevector(self, hkl):
        """
        Calculate the lab wavevector using the unit-vector, oritenation matrix and rotation matrix
        Returns vectors in the lab coordinate system, by default defined like Diamond Light Source:
          x-axis : away from synchrotron ring, towards wall
          y-axis : towards ceiling
          z-axis : along beam direction
        :param hkl: [3xn] array of (h, k, l) reciprocal lattice vectors
        :return: [3xn] array of Q vectors in the lab coordinate system
        """
        uv = self.UV()
        u = self.orientation.umatrix
        r = self.orientation.rotation
        lab = self.orientation.labframe
        return fc.labwavevector(hkl, uv, U=u, R=r, LAB=lab)

    def diff6circle(self, delta=0, gamma=0, energy_kev=None, wavelength=1.0):
        """
        Calcualte wavevector in diffractometer axis using detector angles
        :param delta: float angle in degrees in vertical direction (about diff-z)
        :param gamma: float angle in degrees in horizontal direction (about diff-x)
        :param energy_kev: float energy in KeV
        :param wavelength: float wavelength in A
        :return: q[1*3], ki[1*3], kf[1*3]
        """
        return self.orientation.diff6circle(delta, gamma, energy_kev, wavelength)

    def diff6circle2hkl(self, phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0, energy_kev=None, wavelength=1.0):
        """
        Return [h,k,l] position of diffractometer axes at given energy
        :param phi: float sample angle in degrees
        :param chi: float sample angle in degrees
        :param eta: float sample angle in degrees
        :param mu: float sample angle in degrees
        :param delta: float detector angle in degrees
        :param gamma: float detector angle in degrees
        :param energy_kev: float energy in KeV
        :param wavelength: float wavelength in A
        :return: [h,k,l]
        """
        ub = self.ubmatrix()
        lab = self.orientation.labframe
        return fc.diff6circle2hkl(ub, phi, chi, eta, mu, delta, gamma, energy_kev, wavelength, lab)

    def diff6circle_match(self, phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0, energy_kev=None, wavelength=1.0, fwhm=0.5):
        """
        Return the closest hkl and intensity factor
        :param phi: float sample angle in degrees
        :param chi: float sample angle in degrees
        :param eta: float sample angle in degrees
        :param mu: float sample angle in degrees
        :param delta: float detector angle in degrees
        :param gamma: float detector angle in degrees
        :param energy_kev: float energy in KeV
        :param wavelength: float wavelength in A
        :param fwhm: float peak width in A-1
        :return: [h,k,l], If
        """
        hkl = self.diff6circle2hkl(phi, chi, eta, mu, delta, gamma, energy_kev, wavelength)
        close_hkl = np.round(hkl)
        dist = fg.mag(self.calculateQ(close_hkl) - self.calculateQ(hkl))
        return close_hkl, fg.gauss(dist, fwhm=fwhm)[0]


class Atom:
    """
    Atom class
    Contains site information
    """

    def __init__(self, u, v, w, atom_type, label, occupancy=1.0, uiso=0.001, mxmymz=None):
        self.u = u
        self.v = v
        self.w = w
        self.type = atom_type
        self.label = label
        self.occupancy = occupancy
        self.uiso = uiso
        if mxmymz is None:
            mxmymz = np.array([0., 0, 0])
        self.mxmymz = mxmymz

        props = fc.atom_properties(atom_type)  # returns a numpy structured array
        prop_names = props.dtype.names
        self.properties = props
        for key in prop_names:
            setattr(self, key, props[key][0])

    def __repr__(self):
        out = "Atom(%.3g, %.3g, %.3g, %s, %s)" % (self.u, self.v, self.w, self.type, self.label)
        return out

    def __str__(self):
        out = "%s, %s, u = %6.3f, v = %6.3f, w = %6.3f, occ = %4.2f, uiso = %5.3f, mxmymz = (%.3g,%.3g,%.3g)"
        mx, my, mz = self.mxmymz
        return out % (self.label, self.type, self.u, self.v, self.w, self.occupancy, self.uiso, mx, my, mz)

    def info(self):
        """Display atomic properties"""

        prop_names = self.properties.dtype.names
        out = ''
        for key in prop_names:
            out += '%20s : %s\n' % (key, self.properties[key][0])
        return out

    def uvw(self):
        """
        Returns a [1x3] array of current positions
        :return: np.array([1x3])
        """
        return np.asarray([self.u, self.v, self.w], dtype=float)

    def total_moment(self):
        """Return the total moment along a, b, c directions"""
        return np.sum(self.mxmymz)


class Atoms:
    """
    Contains properties of atoms within the crystal
    Each atom has properties:
        u,v,w >> atomic coordinates, in the basis of the unit cell
        type >> element species, given as element name, e.g. 'Fe'
        label >> Name of atomic position, e.g. 'Fe1'
        occupancy >> Occupancy of this atom at this atomic position
        uiso >> atomic displacement factor (ADP) <u^2>
        mxmymz >> magnetic moment direction [x,y,z]
    """

    _default_atom = 'Fe'
    _default_uiso = 1.0 / (8 * np.pi ** 2)  # B=1

    _required_cif = [
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    _type_str_fmt = '<U8'

    def __init__(self, u=[0], v=[0], w=[0], type=None,
                 label=None, occupancy=None, uiso=None, mxmymz=None):
        " Initialisation, defines Atoms defaults"
        self.u = np.asarray(u, dtype=float).reshape(-1)
        self.v = np.asarray(v, dtype=float).reshape(-1)
        self.w = np.asarray(w, dtype=float).reshape(-1)
        Natoms = len(u)

        # ---Defaults---
        # type
        if type is None:
            self.type = np.asarray([self._default_atom] * Natoms, dtype=self._type_str_fmt)
        else:
            self.type = np.asarray(type, dtype=self._type_str_fmt).reshape(-1)
        # label
        if label is None:
            self.label = self.type.copy()
        else:
            self.label = np.asarray(label, dtype=str).reshape(-1)
        # occupancy
        if occupancy is None:
            self.occupancy = np.ones(Natoms)
        else:
            self.occupancy = np.asarray(occupancy, dtype=float).reshape(-1)
        # Uiso
        if uiso is None:
            self.uiso = self._default_uiso * np.ones(Natoms)
        else:
            self.uiso = np.asarray(uiso, dtype=float).reshape(-1)
        # Mag vector mxmymz
        if mxmymz is None:
            self.mx = np.zeros(Natoms)
            self.my = np.zeros(Natoms)
            self.mz = np.zeros(Natoms)
        else:
            mpos = np.asarray(mxmymz, dtype=float).reshape(Natoms, 3)
            self.mx = mpos[:, 0]
            self.my = mpos[:, 1]
            self.mz = mpos[:, 2]

    def __call__(self, u=[0], v=[0], w=[0], type=None,
                 label=None, occupancy=None, uiso=None, mxmymz=None):
        """Re-initialises the class, generating new atomic positions"""
        self.__init__(u, v, w, type, label, occupancy, uiso, mxmymz=None)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = list(self.label).index(idx)
        return self.atom(idx)

    def fromcif(self, cifvals):
        """
        Import atom parameters from a cif dictionary
        Required cif keys:
            _atom_site_label
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
        Optional cif keys:
            _atom_site_type_symbol
            _atom_site_U_iso_or_equiv
            _atom_site_B_iso_or_equiv
            _atom_site_occupancy
            _atom_site_moment_label
            _atom_site_moment_crystalaxis_x
            _atom_site_moment_crystalaxis_y
            _atom_site_moment_crystalaxis_z
        :param cifvals: dict
        :return: none
        """
        if not fc.cif_check(cifvals, self._required_cif):
            warn('Atom site parameters cannot be read from cif')  # TODO: provide more details
            return

        keys = cifvals.keys()

        # Get atom names & labels
        label = cifvals['_atom_site_label']

        if '_atom_site_type_symbol' in keys:
            element = [fc.str2element(x) for x in cifvals['_atom_site_type_symbol']]
        else:
            element = [fc.str2element(x) for x in cifvals['_atom_site_label']]

        while False in element:
            fidx = element.index(False)
            warn('%s is not a valid element, replacing with H' % cifvals['_atom_site_label'][fidx])
            element[fidx] = 'H'

        # Replace Deuterium with Hydrogen
        if 'D' in element:
            warn('Replacing Deuterium ions with Hydrogen')
            element = ['H' if el == 'D' else el for el in element]

        # Thermal parameters
        if '_atom_site_U_iso_or_equiv' in keys:
            uiso = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_U_iso_or_equiv']])
        elif '_atom_site_B_iso_or_equiv' in keys:
            biso = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_B_iso_or_equiv']])
            uiso = fc.biso2uiso(biso)
        else:
            uiso = np.zeros(len(element))
        # Occupancy
        if '_atom_site_occupancy' in keys:
            occ = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_occupancy']])
        else:
            occ = np.ones(len(element))

        # Get coordinates
        u = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_x']], dtype=float)
        v = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_y']], dtype=float)
        w = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_z']], dtype=float)

        # Get magnetic vectors
        mx = np.zeros(len(u))
        my = np.zeros(len(u))
        mz = np.zeros(len(u))
        if '_atom_site_moment_label' in keys:
            mag_atoms = cifvals['_atom_site_moment_label']
            mxi = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_x']])
            myi = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_y']])
            mzi = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_z']])
            for n, ma in enumerate(mag_atoms):
                mag_idx = label.index(ma)
                mx[mag_idx] = mxi[n]
                my[mag_idx] = myi[n]
                mz[mag_idx] = mzi[n]

        # Assign attributes
        self.u = u
        self.v = v
        self.w = w
        self.type = np.array(element, dtype=self._type_str_fmt)
        self.label = np.array(label, dtype=self._type_str_fmt)
        self.occupancy = occ
        self.uiso = uiso
        self.mx = mx
        self.my = my
        self.mz = mz

    def update_cif(self, cifvals):
        """
        Update cif dict with stored values
        :param cifvals: cif dict from readcif
        :return: cifvals
        """

        keys = cifvals.keys()

        cifvals['_atom_site_label'] = self.label
        cifvals['_atom_site_type_symbol'] = self.type
        cifvals['_atom_site_U_iso_or_equiv'] = self.uiso
        cifvals['_atom_site_occupancy'] = self.occupancy
        cifvals['_atom_site_fract_x'] = self.u
        cifvals['_atom_site_fract_y'] = self.v
        cifvals['_atom_site_fract_z'] = self.w

        # Magnetic moments
        mag_label = []
        mag_x = []
        mag_y = []
        mag_z = []
        mag_sym = []
        for n in range(len(self.label)):
            if self.mx[n] ** 2 + self.my[n] ** 2 + self.mz[n] ** 2 > 0.01:
                mag_label += [self.label[n]]
                mag_x += [self.mx[n]]
                mag_y += [self.my[n]]
                mag_z += [self.mz[n]]
                mag_sym += ['mx,my,mz']

        cifvals['_atom_site_moment.label'] = mag_label
        cifvals['_atom_site_moment.crystalaxis_x'] = mag_x
        cifvals['_atom_site_moment.crystalaxis_y'] = mag_y
        cifvals['_atom_site_moment.crystalaxis_z'] = mag_z
        # cifvals['_atom_site_moment.symmform'] = mag_sym
        return cifvals

    def atom(self, idx):
        """Create Atom object for atom site"""
        idx = np.asarray(idx, dtype=int).reshape(-1)
        atoms = [Atom(self.u[i], self.v[i], self.w[i], self.type[i], self.label[i],
                      self.occupancy[i], self.uiso[i], self.mxmymz()[i]) for i in idx]
        if len(atoms) == 1:
            return atoms[0]
        return atoms

    def changeatom(self, idx, u=None, v=None, w=None, type=None,
                   label=None, occupancy=None, uiso=None, mxmymz=None):
        """
        Change an atom site's properties.
        If properties are given as None, they are not changed.

        :param idx: atom array index
        :param u: atomic position u in relative coordinates along basis vector a
        :param v: atomic position u in relative coordinates along basis vector b
        :param w: atomic position u in relative coordinates along basis vector c
        :param type: atomic element type
        :param label: atomic site label
        :param occupancy: atom site occupancy
        :param uiso: atom site isotropic thermal parameter
        :param mxmymz: atom site magnetic vector (mx, my, mz)
        :return: None
        """

        if u is not None:
            self.u[idx] = u

        if v is not None:
            self.v[idx] = v

        if w is not None:
            self.w[idx] = w

        if type is not None:
            self.type[idx] = type

        if label is not None:
            old_labels = list(self.label)
            old_labels[idx] = label
            self.label = np.array(old_labels, dtype=self._type_str_fmt)

        if occupancy is not None:
            self.occupancy[idx] = occupancy

        if uiso is not None:
            self.uiso[idx] = uiso

        if mxmymz is not None:
            mpos = np.asarray(mxmymz, dtype=float).reshape(-1, 3)
            self.mx[idx] = mpos[:, 0]
            self.my[idx] = mpos[:, 1]
            self.mz[idx] = mpos[:, 2]

    def findatom(self, u=None, v=None, w=None, type=None,
                 label=None, occupancy=None, uiso=None, mxmymz=None, tol=0.01):
        """
        Find atom using parameters, return idx
        :param u: float
        :param v: float
        :param w: float
        :param type: str
        :param label: str
        :param occupancy: float
        :param uiso: float
        :param mxmymz: [mx,my,mz]
        :param tol: float, tolerance to match value
        :return: array of indexes
        """
        all = np.ones(len(self.u))
        if u is not None:
            all *= np.abs(self.u - u) < tol
        if v is not None:
            all *= np.abs(self.v - v) < tol
        if w is not None:
            all *= np.abs(self.w - w) < tol
        if type is not None:
            all *= np.array(self.type) == type
        if label is not None:
            all *= np.array([True if label.lower() in a.lower() else False for a in self.label])
        if occupancy is not None:
            all *= np.abs(self.occupancy - occupancy) < tol
        if uiso is not None:
            all *= np.abs(self.uiso - uiso) < tol
        if mxmymz is not None:
            all *= fg.mag(self.mxmymz() - mxmymz) < tol
        return np.where(all)[0]

    def addatom(self, u=0, v=0, w=0, type=None, label=None, occupancy=None, uiso=None, mxmymz=None):
        """
        Adds a new atom
        :param u:
        :param v:
        :param w:
        :param type:
        :param label:
        :param occupancy:
        :param uiso:
        :param mxmymz:
        :return:
        """

        self.u = np.append(self.u, [u])
        self.v = np.append(self.v, [v])
        self.w = np.append(self.w, [w])

        # type
        if type is None:
            self.type = np.append(self.type, [self._default_atom])
        else:
            self.type = np.append(self.type, [type])

        # label
        if label is None:
            self.label = np.append(self.label, [self._default_atom])
        else:
            self.label = np.append(self.label, [label])

        # occupancy
        if occupancy is None:
            self.occupancy = np.append(self.occupancy, [1.0])
        else:
            self.occupancy = np.append(self.occupancy, [occupancy])

        # Uiso
        if uiso is None:
            self.uiso = np.append(self.uiso, [self._default_uiso])
        else:
            self.uiso = np.append(self.uiso, [uiso])

        if mxmymz is None:
            self.mx = np.append(self.mx, [0])
            self.my = np.append(self.my, [0])
            self.mz = np.append(self.mz, [0])
        else:
            self.mx = np.append(self.mx, mxmymz[0])
            self.my = np.append(self.my, mxmymz[1])
            self.mz = np.append(self.mz, mxmymz[2])

    def removeatom(self, idx):
        """
        Removes atom number idx from the list
        :param idx: int, atom index
        :return: None
        """

        self.u = np.delete(self.u, idx)
        self.v = np.delete(self.v, idx)
        self.w = np.delete(self.w, idx)
        self.type = np.delete(self.type, idx)
        self.label = np.delete(self.label, idx)
        self.occupancy = np.delete(self.occupancy, idx)
        self.uiso = np.delete(self.uiso, idx)
        self.mx = np.delete(self.mx, idx)
        self.my = np.delete(self.my, idx)
        self.mz = np.delete(self.mz, idx)

    def remove_duplicates(self, min_distance=0.01, all_types=False):
        """
        Remove atoms of the same type that are too close to each other
        :param min_distance: remove atoms within this distance, in fractional units
        :param all_types: if True, also remove atoms of different types
        :return: None
        """

        uvw = self.uvw()
        atom_type = self.type
        rem_atom_idx = []
        for n in range(0, len(atom_type) - 1):
            match_position = fg.mag(uvw[n, :] - uvw[n + 1:, :]) < min_distance
            match_type = atom_type[n] == atom_type[n + 1:]
            if all_types:
                rem_atom_idx += list(1 + n + np.where(match_position)[0])
            else:
                rem_atom_idx += list(1 + n + np.where(match_position * match_type)[0])
        # remove atoms
        print('Removing %d atoms' % len(rem_atom_idx))
        self.removeatom(rem_atom_idx)

    def check(self):
        """
        Checks the validity of the contained attributes
        :return: None
        """

        good = True

        # Check lengths
        Natoms = len(self.u)
        if len(self.v) != Natoms: good = False; print('Cell.v is the wrong length!')
        if len(self.w) != Natoms: good = False; print('Cell.w is the wrong length!')
        if len(self.type) != Natoms: good = False; print('Cell.type is the wrong length!')
        if len(self.label) != Natoms: good = False; print('Cell.label is the wrong length!')
        if len(self.occupancy) != Natoms: good = False; print('Cell.occupancy is the wrong length!')
        if len(self.uiso) != Natoms: good = False; print('Cell.uiso is the wrong length!')
        if len(self.mx) != Natoms: good = False; print('Cell.mx is the wrong length!')
        if len(self.my) != Natoms: good = False; print('Cell.my is the wrong length!')
        if len(self.mz) != Natoms: good = False; print('Cell.mz is the wrong length!')

        # Check atom types
        atoms = fc.atom_properties(fields='Element')
        for atom in self.type:
            if atom not in atoms:
                good = False
                print(atom, ' has no properties assigned.')

        # Check occupancy
        for n in range(Natoms):
            if self.occupancy[n] > 1:
                good = False
                print('Atom %d has occupancy greater than 1!' % n)
            elif self.occupancy[n] < 0:
                good = False
                print('Atom %d has occupancy less than 0!' % n)

        # Check uiso
        for n in range(Natoms):
            if self.uiso[n] < 0:
                good = False
                print('Atom %d has uiso less than 0!' % n)
            elif self.uiso[n] > 0.1:
                good = False
                print('Atom %d has uiso greater than 0.1!' % n)

        if good:
            print("Atoms look good.")

    def fitincell(self):
        """Adjust all atom positions to fit within unit cell"""
        uvw = fc.fitincell(self.uvw())
        self.u, self.v, self.w = uvw.T

    def uvw(self):
        """
        Returns a [nx3] array of current positions
        :return: np.array([nx3])
        """
        return np.asarray([self.u, self.v, self.w], dtype=float).T

    def mxmymz(self):
        """
        Returns a [nx3] array of magnetic vectors
        :return: np.array([nx3])
        """
        return np.asarray([self.mx, self.my, self.mz], dtype=float).T

    def total_moment(self):
        """Return the total moment along a, b, c directions"""
        return np.sum(self.mxmymz(), axis=0)

    def ismagnetic(self):
        """ Returns True if any ions have magnetic moments assigned"""
        return np.any(np.abs(self.mxmymz()) > 0)

    def get(self):
        """
        Returns the structure arrays
         uvw, type, label, occupancy, uiso, mxmymz = Atoms.get()
        """
        return self.uvw(), np.asarray(self.type), np.asarray(self.label), np.asarray(self.occupancy), np.asarray(
            self.uiso), self.mxmymz()

    def generate_lattice(self, U=1, V=1, W=0, centred=True):
        """
        Expand the atomic positions beyond the unit cell, creating a lattice
            uvw,type,label,occ,uiso,mxmymz = self.generate_lattice(U,V,W,centred)
              U,V,W = maximum lattice index to loop to
              centred = if True, positions will loop from e.g. -U to U,
                        otherwise, will loop from e.g. 0 to U
              uvw,type,label,occ,uiso,mxmymz = standard array outputs of Atoms
        """

        uvw, type, label, occ, uiso, mxmymz = self.get()

        new_uvw = np.ndarray([0, 3])
        new_type = np.ndarray([0])
        new_label = np.ndarray([0])
        new_occupancy = np.ndarray([0])
        new_uiso = np.ndarray([0])
        new_mxmymz = np.ndarray([0, 3])
        if centred:
            urange = range(-U, U + 1)
            vrange = range(-V, V + 1)
            wrange = range(-W, W + 1)
        else:
            urange = range(0, U + 1)
            vrange = range(0, V + 1)
            wrange = range(0, W + 1)

        for uval in urange:
            for vval in vrange:
                for wval in wrange:
                    new_uvw = np.vstack([new_uvw, uvw + [uval, vval, wval]])
                    new_type = np.hstack([new_type, type])
                    new_label = np.hstack([new_label, label])
                    new_occupancy = np.hstack([new_occupancy, occ])
                    new_uiso = np.hstack([new_uiso, uiso])
                    new_mxmymz = np.vstack([new_mxmymz, mxmymz])

        return new_uvw, new_type, new_label, new_occupancy, new_uiso, new_mxmymz

    def weight(self):
        """
        Calculate the molecular weight in g/mol of all the atoms
        :return: float
        """
        weights = fc.atom_properties(self.type, 'Weight')
        total_weight = sum(weights * self.occupancy)  # g per mol per unit cell
        return total_weight

    def mass_fraction(self):
        """
        Return the mass fraction per element
        :return: float
        """

        weights = fc.atom_properties(self.type, 'Weight') * self.occupancy
        total_weight = sum(weights)

        return weights / total_weight

    def scattering_factor_coefficients(self, table='itc'):
        """Return scattering factor coefficients for the elements"""
        return fc.scattering_factor_coefficients(*self.type, table=table)

    def info(self, idx=None, type=None):
        """
        Prints properties of all atoms
        :param idx: None or array of atoms to display
        :param type: None or str type of atom to dispaly
        :return: str
        """

        if idx is not None:
            disprange = np.asarray(idx)
        elif type is not None:
            disprange = np.where(self.type == type)[0]
        else:
            disprange = range(0, len(self.u))

        out = ''
        if np.any(fg.mag(self.mxmymz()) > 0):
            out += '  n Label Atom  u       v       w        Occ  Uiso      mx      my      mz      \n'
            fmt = '%3.0f %5s %4s %7.4f %7.4f %7.4f   %4.2f %6.4f   %7.4f %7.4f %7.4f\n'
            for n in disprange:
                out += fmt % (
                    n, self.label[n], self.type[n], self.u[n], self.v[n], self.w[n],
                    self.occupancy[n], self.uiso[n],
                    self.mx[n], self.my[n], self.mz[n]
                )
        else:
            out += '  n Label Atom  u       v       w        Occ  Uiso\n'
            fmt = '%3.0f %5s %4s %7.4f %7.4f %7.4f   %4.2f %6.4f\n'
            for n in disprange:
                out += fmt % (
                    n, self.label[n], self.type[n], self.u[n], self.v[n], self.w[n],
                    self.occupancy[n], self.uiso[n]
                )
        return out

    def __repr__(self):
        return "%d sites, elements: %s" % (len(self.type), np.unique(self.type))

    def __str__(self):
        return self.info()


class Symmetry:
    """
    Contains symmetry information about the crystal, including the symmetry operations.
    """

    spacegroup = 'P1'
    spacegroup_number = 1
    symmetry_operations = ['x,y,z']
    symmetry_operations_magnetic = ['x,y,z']
    symmetry_operations_time = [1]
    symmetry_matrices = np.eye(4)
    spacegroup_dict = fc.spacegroup(1)

    def __init__(self, symmetry_operations=None, symmetry_operations_magnetic=None):
        """Initialises the symmetry group"""

        if symmetry_operations is not None:
            self.addsym(symmetry_operations, symmetry_operations_magnetic)
        else:
            self.generate_matrices()

    def fromcif(self, cifvals):
        """
        Import symmetry information from a cif dictionary
        Required cif keys:
            None
        Optional cif keys:
            _symmetry_equiv_pos_as_xyz
            _space_group_symop_operation_xyz
            _space_group_symop_magn_operation_xyz
            _space_group_symop_magn_operation_mxmymz
            _space_group_symop_magn_centering_xyz
            _space_group_symop_magn_centering_mxmymz
            _symmetry_space_group_name_H-M
            _space_group_name_H-M_alt
            _space_group_magn_name_BNS
            _symmetry_Int_Tables_number
            _space_group_IT_number
            _space_group_magn_number_BNS
        :param cifvals: dict of values from cif
        :return:
        """

        keys = cifvals.keys()

        sym, mag, tim = fc.cif_symmetry(cifvals)
        self.symmetry_operations = sym
        self.symmetry_operations_magnetic = mag
        self.symmetry_operations_time = tim

        # Get space group
        if '_symmetry_space_group_name_H-M' in keys:
            spacegroup = cifvals['_symmetry_space_group_name_H-M']
        elif '_space_group_name_H-M_alt' in keys:
            spacegroup = cifvals['_space_group_name_H-M_alt']
        elif '_space_group_magn_name_BNS' in keys:
            spacegroup = cifvals['_space_group_magn_name_BNS']
        elif len(sym) == 1:
            spacegroup = 'P1'
        else:
            spacegroup = 'unknown'
        self.spacegroup = spacegroup

        if '_symmetry_Int_Tables_number' in keys:
            sgn = cifvals['_symmetry_Int_Tables_number']
        elif '_space_group_IT_number' in keys:
            sgn = cifvals['_space_group_IT_number']
        elif '_space_group_magn_number_BNS' in keys:
            sgn = cifvals['_space_group_magn_number_BNS'].strip('\'"')
        elif spacegroup == 'P1':
            sgn = '1'
        else:
            sgn = '0'
        try:
            self.spacegroup_number = float(sgn)
        except ValueError:
            self.spacegroup_number = 0
        try:
            if '.' in sgn:
                self.spacegroup_dict = fc.spacegroup_magnetic(sgn)
            else:
                self.spacegroup_dict = fc.spacegroup(sgn)
        except KeyError:
            # Find from spacegroup
            check = fc.find_spacegroup(spacegroup)
            if check:
                self.spacegroup_dict = check
                self.spacegroup_number = check['space group number']
            else:
                self.spacegroup_dict = fc.spacegroup(1)

        if len(self.symmetry_operations) == 1:
            # use found spacegroup as none provided by CIF
            self.load_spacegroup(sg_dict=self.spacegroup_dict)
        else:
            self.generate_matrices()

    def update_cif(self, cifvals):
        """
        Update cifvals dict with current symmetry operations
        :param cifvals: cif dict from functions_crystallography.readcif
        :return: cifvals
        """

        cifvals['_symmetry_equiv_pos_as_xyz'] = self.symmetry_operations
        cifvals['_space_group_symop_operation_xyz'] = self.symmetry_operations

        # newer versions of mcif don't use magn_operation.mxmymz but use symform in atom spec
        # . may be in the wrong place for some program
        # Add time to symmetry operations
        time_ops = []
        for n in range(len(self.symmetry_operations)):
            ops = self.symmetry_operations[n].split(',')
            ops += ['%+g' % self.symmetry_operations_time[n]]
            time_ops += [','.join(ops[:4])]  # use the current value first
        cifvals['_space_group_symop_magn_operation.id'] = range(1, len(self.symmetry_operations) + 1)
        cifvals['_space_group_symop_magn_operation.xyz'] = time_ops
        cifvals['_space_group_symop_magn_operation.mxmymz'] = self.symmetry_operations_magnetic
        cifvals['_space_group_symop_magn_centering.id'] = ['1']
        cifvals['_space_group_symop_magn_centering.xyz'] = ['x,y,z,+1']
        cifvals['_space_group_symop_magn_centering.mxmymz'] = ['mx,my,mz']

        # Get space group
        cifvals['_symmetry_space_group_name_H-M'] = self.spacegroup
        cifvals['_space_group_name_H-M_alt'] = self.spacegroup
        cifvals['_space_group_magn.name_BNS'] = self.spacegroup

        cifvals['_symmetry_Int_Tables_number'] = self.spacegroup_number
        cifvals['_space_group_IT_number'] = self.spacegroup_number
        cifvals['_space_group_magn.number_BNS'] = self.spacegroup_number
        return cifvals

    def load_spacegroup(self, sg_number=None, sg_dict=None):
        """
        Load symmetry operations from a spacegroup from the International Tables of Crystallogrphy
        See functions_crystallography.spacegroup for more details
        :param sg_number: space group number (1-230)
        :param sg_dict: alternative input: spacegroup dict from fc.spacegroup
        :return: None
        """
        if sg_dict is None:
            sg_dict = fc.spacegroup(sg_number)
        if 'positions magnetic' in sg_dict:
            self.load_magnetic_spacegroup(sg_dict=sg_dict)
            return
        self.spacegroup_number = int(sg_dict['space group number'])
        self.spacegroup = sg_dict['space group name']
        self.spacegroup_dict = sg_dict

        symops = sg_dict['general positions']
        self.symmetry_operations = symops
        self.symmetry_operations_magnetic = fc.symmetry_ops2magnetic(symops)
        self.symmetry_operations_time = [1] * len(symops)

        self.generate_matrices()

    def load_magnetic_spacegroup(self, msg_number=None, sg_dict=None):
        """
        Load symmetry operations from a magnetic spacegroup from Bilbao crystallographic server
        Replaces the current symmetry operators and the magnetic symmetry operators.
        See functions_crystallography.spacegroup_magnetic for more details
        :param msg_number: magnetic space group number e.g. 61.433
        :param sg_dict: alternative inuput: spacegroup dict from fc.spacegroup_magnetic
        :return: None
        """
        if sg_dict is None:
            maggroup = fc.spacegroup_magnetic(msg_number)
        else:
            maggroup = sg_dict
        self.spacegroup_number = maggroup['space group number']
        self.spacegroup = maggroup['space group name']
        self.spacegroup_dict = maggroup
        symops = maggroup['positions general']
        symmag = maggroup['positions magnetic']
        self.symmetry_operations = symops
        self.symmetry_operations_magnetic = symmag
        self.symmetry_operations_time = fc.sym_op_time(symops)

        self.generate_matrices()

    def changesym(self, idx, operation):
        """
        Change a symmetry operation
        :param idx: symmetry index
        :param operation: str e.g. 'x,-y,z'
        """

        self.symmetry_operations[idx] = operation
        self.generate_matrices()

    def invert_magsym(self, idx):
        """
        Invert the time symmetry of a magnetic symmetry
        :param idx: symmetry index, 0:Nsym
        :return:
        """

        ID = np.asarray(idx).reshape(-1)
        for idx in ID:
            old_op = self.symmetry_operations_magnetic[idx]
            new_op = fc.invert_sym(old_op)
            self.symmetry_operations_magnetic[idx] = new_op

    def addsym(self, operations, mag_operations=None):
        """
        Add symmetry operations
            Symmetry.addsym('x,y,z+1/2') >> adds single symmetry operation, magnetic operation is infered
            Symmetry.addsym(['x,y,z+1/2','z,x,y']) >> adds multiple symmetry operation, magnetic operations are infered
            Symmetry.addsym('x,y,z+1/2','x,y,-z') >> adds single symmetry operation + magnetic operation
        """

        operations = np.array(operations).reshape(-1)
        mag_operations = np.array(mag_operations).reshape(-1)
        tim_operations = fc.sym_op_time(operations)

        if mag_operations[0] is None:
            mag_operations = fc.symmetry_ops2magnetic(operations)

        for op, mag_op, tim_op in zip(operations, mag_operations, tim_operations):
            if op not in self.symmetry_operations:
                self.symmetry_operations += [op]
                self.symmetry_operations_magnetic += [mag_op]
                self.symmetry_operations_time += [tim_op]
        self.generate_matrices()

    def addcen(self, operations, mag_operations=None):
        """
        Apply centring operations to current symmetry operations
        """

        operations = np.array(operations).reshape(-1)
        mag_operations = np.array(mag_operations).reshape(-1)

        if mag_operations[0] is None:
            mag_operations = fc.symmetry_ops2magnetic(operations)

        self.symmetry_operations = fc.gen_symcen_ops(self.symmetry_operations, operations)
        self.symmetry_operations_magnetic = fc.gen_symcen_ops(self.symmetry_operations_magnetic, mag_operations)
        self.symmetry_operations_time = fc.sym_op_time(self.symmetry_operations)
        self.generate_matrices()

    def generate_matrices(self):
        """
        Generates the symmetry matrices from string symmetry operations
        """

        self.symmetry_matrices = fc.gen_sym_mat(self.symmetry_operations)

    def print_subgroups(self):
        """
        Return str of subgroups of this spacegroup
        :return: str
        """
        return fc.spacegroup_subgroups_list(sg_dict=self.spacegroup_dict)

    def print_magnetic_spacegroups(self):
        """
        Return str of available magnetic spacegroups for this spacegroup
        :return: str
        """
        return fc.spacegroup_magnetic_list(sg_dict=self.spacegroup_dict)

    def symmetric_coordinates(self, UVW, MXMYMZ=None, remove_identical=True):
        """
        Returns array of symmetric coordinates
        Uses fc.gen_sym_pos
        Returns coordinates wrapped within the unit cell, with identical positions removed.
        All positions returned if remove_identical=False
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.symmetric_coordinates([0.1,0,0])
            >  array([[0.1, 0.0, 0.0],
                      [0.9, 0.0, 0.0],
                      [0.0, 0.1, 0.0]])
        """

        # Apply symmetry operations to atomic positions
        u, v, w = UVW
        sym_uvw = fc.gen_sym_pos(self.symmetry_operations, u, v, w)
        sym_uvw = fc.fitincell(sym_uvw)

        # Remove identical positions
        if remove_identical:
            unique_uvw, uniqueidx, matchidx = fg.unique_vector(sym_uvw, tol=0.01)
            sym_uvw = unique_uvw

        if MXMYMZ is None:
            return sym_uvw

        # Apply magnetic symmetry operations to magnetic vectors
        mx, my, mz = MXMYMZ
        sym_xyz = fc.gen_sym_pos(self.symmetry_operations_magnetic, mx, my, mz)

        # Remove positions that were identical
        if remove_identical:
            sym_xyz = sym_xyz[uniqueidx]
        return sym_uvw, sym_xyz

    def symmetric_coordinate_operations(self, UVW, MXMYMZ=None):
        """
        Returns array of symmetric operations for given position
        Uses fc.gen_sym_pos
        Returns list of identical symmetry operations, with identical positions removed.
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.symmetric_coordinate_operations([0.1,0,0])
            >  array(['x,y,z', '-x,-y,-z', 'y,x,z'])
        """

        # Apply symmetry operations to atomic positions
        u, v, w = UVW
        sym_uvw = fc.gen_sym_pos(self.symmetry_operations, u, v, w)
        sym_uvw = fc.fitincell(sym_uvw)

        # Remove identical positions
        unique_uvw, uniqueidx, matchidx = fg.unique_vector(sym_uvw, tol=0.01)

        if MXMYMZ is None:
            return np.array(self.symmetry_operations)[uniqueidx]

        return np.array(self.symmetry_operations)[uniqueidx], np.array(self.symmetry_operations_magnetic)[uniqueidx]

    def print_symmetric_coordinate_operations(self, UVW, remove_identical=True):
        """
        Returns array of symmetric operations for given position
        Uses fc.gen_sym_pos
        Returns list of identical symmetry operations, with identical positions removed
        All positions returned if remove_identical=False
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.print_symmetric_coordinate_operations([0.1,0,0])
            > n  u       v       w                   Symmetry    Magnetic Symmetry
              0   0.1000  0.0000  0.0000                x,y,z                x,y,z
              1   0.9000  0.0000  0.0000             -x,-y,-z             -x,-y,-z
              2   0.0000  0.1000  0.0000                y,x,z                y,x,z
        """

        UVW = np.asarray(UVW, dtype=float).reshape([-1, 3])

        out = ''
        for u, v, w in UVW:
            # Apply symmetry operations to atomic positions
            sym_uvw = fc.gen_sym_pos(self.symmetry_operations, u, v, w)
            sym_uvw = fc.fitincell(sym_uvw)

            # Remove identical positions
            unique_uvw, uniqueidx, matchidx = fg.unique_vector(sym_uvw, tol=0.01)

            out += ' (%1.3g, %1.3g, %1.3g)\n' % (u, v, w)
            out += '  n       u       v       w   Symmetry                        Magnetic Symmetry\n'
            if remove_identical:
                # Only display distict positions
                for n in range(len(unique_uvw)):
                    out += '%3d %7.4f %7.4f %7.4f %30s %20s\n' % (
                        uniqueidx[n], unique_uvw[n, 0], unique_uvw[n, 1], unique_uvw[n, 2],
                        self.symmetry_operations[uniqueidx[n]], self.symmetry_operations_magnetic[uniqueidx[n]])
            else:
                for n in range(len(sym_uvw)):
                    out += '%3d %7.4f %7.4f %7.4f %20s %20s\n' % (
                        n, sym_uvw[n, 0], sym_uvw[n, 1], sym_uvw[n, 2], self.symmetry_operations[n],
                        self.symmetry_operations_magnetic[n])
        return out

    def symmetric_reflections(self, HKL):
        """
        Returns array of symmetric reflection indices
        Uses fc.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            Symmetry.symmetric_reflections([1,0,0])
            >  array([[1,  0, 0],
                        [-1, 0, 0],
                        [0, -1, 0]])
        """

        NsymOps = len(self.symmetry_operations)

        HKL = np.asarray(HKL, dtype=float).reshape((-1, 3))

        symHKL = np.zeros([len(HKL) * NsymOps, 3])
        ii = 0
        for n in range(len(HKL)):
            for m in range(NsymOps):
                # multiply only by the rotational part
                opHKL = np.dot(HKL[n], self.symmetry_matrices[m][:3, :3])
                symHKL[ii, :] = opHKL
                ii += 1
        return symHKL

    def symmetric_reflections_unique(self, HKL):
        """
        Returns array of symmetric reflection indices, with identical reflections removed
        Uses fc.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            Symmetry.symmetric_reflections([1,1,0])
            >  array([[1,  1, 0],
                        [-1,-1, 0]])
        """

        symHKL = self.symmetric_reflections(HKL)

        # Remove identical positions
        unique_hkl, uniqueidx, matchidx = fg.unique_vector(symHKL, tol=0.01)
        return unique_hkl

    def symmetric_reflections_count(self, HKL):
        """
        Returns array of symmetric reflection indices,
        identical reflections are removed, plus the counted sum of each reflection
        Uses fc.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            HKL, count = Symmetry.symmetric_reflections([1,1,0])
            >  HKL = array([[1,  1, 0],
                              [-1,-1, 0]])
            > count = array([2,1])
                        
        """

        symHKL = self.symmetric_reflections(HKL)

        # Remove identical positions
        unique_hkl, uniqueidx, matchidx = fg.unique_vector(symHKL, tol=0.01)

        # Count reflections at identical positions
        count = np.zeros(len(unique_hkl))
        for n in range(len(unique_hkl)):
            count[n] = matchidx.count(n)
        return unique_hkl, count

    def symmetric_intensity(self, HKL, I, dI=None):
        """
        Returns symmetric reflections with summed intensities of repeated reflections
        Assumes HKL reflections are unique, repeated reflections will be incorrectly added together.
        Uses fc.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            HKL, I = Symmetry.symmetric_intensity([1,1,0],10)
            >  HKL = array([[1,  1, 0],
                              [-1,-1, 0]])
            > I = array([20,10])
            OR
            HKL, I, dI = Symmetry.symmetric_intensity([1,1,0],10,1)
            >  HKL = array([[1,  1, 0],
                              [-1,-1, 0]])
            > I = array([20,10])
            > dI = array([2,1])
        """

        HKL = np.asarray(HKL, dtype=float).reshape((-1, 3))
        I = np.asarray(I, dtype=float)

        NsymOps = len(self.symmetry_operations)

        symHKL = self.symmetric_reflections(HKL)
        symI = np.repeat(I, NsymOps)  # repeats each element of I

        # Remove identical positions
        unique_hkl, uniqueidx, matchidx = fg.unique_vector(symHKL, tol=0.01)
        unique_I = symI[uniqueidx]

        # Sum intensity at duplicate positions
        count = np.zeros(len(unique_hkl))
        for n in range(len(unique_hkl)):
            count[n] = matchidx.count(n)

        sum_I = unique_I * count

        if dI is None:
            return unique_hkl, sum_I

        dI = np.asarray(dI, dtype=float)
        sym_dI = np.repeat(dI, NsymOps)  # repeats each element of I
        unique_dI = sym_dI[uniqueidx]
        sum_dI = unique_dI * count  # *np.sqrt(count)?
        return unique_hkl, sum_I, sum_dI

    def is_symmetric_reflection(self, hkl1, hkl2, tolerance=0.01):
        """
        Check if reflection 1 is a symmetric equivalent of reflection 2
        :param hkl1: [h,k,l] reflection 1
        :param hkl2: [h,k,l] reflection 2
        :param tolerance: tolerance for matching reflections
        :return: True/ False
        """
        symm_hkl = self.symmetric_reflections(hkl2)
        difference = fg.mag(hkl1 - symm_hkl)
        return np.any(difference < tolerance)

    def remove_symmetric_reflections(self, hkl_list, tolerance=0.01):
        """
        Return a list of reflections with symmetric reflections removed
        :param hkl_list: list of [h,k,l] reflections
        :param tolerance: tolerance for matching reflections
        :return: array of [h,k,l]
        """
        hkl_list = np.asarray(hkl_list).reshape(-1, 3)
        symmetric_idx = np.zeros(len(hkl_list), dtype=bool)
        symmetric_hkl = np.empty([0, 3])
        for n in range(len(hkl_list)):
            difference = fg.mag(hkl_list[n] - symmetric_hkl)
            if np.any(difference < tolerance): continue

            symmetric_hkl = np.vstack([symmetric_hkl, self.symmetric_reflections(hkl_list[n])])
            symmetric_idx[n] = True
        return hkl_list[symmetric_idx]

    def average_symmetric_intensity(self, hkl_list, intensity_list, tolerance=0.01):
        """
        Return a list of reflections with symmetric reflections removed, matching reflections will be averaged
        :param hkl_list: list of [h,k,l] reflections
        :param intensity_list: list of intensities
        :param tolerance: tolerance for matching reflections
        :return: array of [h,k,l]
        """
        hkl_list = np.asarray(hkl_list).reshape(-1, 3)
        intensity_list = np.asarray(intensity_list).reshape(-1)
        intensity_tot = np.ones(len(intensity_list))
        symmetric_idx = np.zeros(len(hkl_list), dtype=bool)
        symmetric_hkl = np.empty([0, 3])
        hkl_idx = np.empty([0], dtype=int)
        for n in range(len(hkl_list)):
            difference = fg.mag(hkl_list[n] - symmetric_hkl)
            close = difference < tolerance
            if np.any(close):
                intensity_list[hkl_idx[close]] += intensity_list[n]
                intensity_tot[hkl_idx[close]] += 1
                continue
            symref = self.symmetric_reflections(hkl_list[n])
            symmetric_hkl = np.vstack([symmetric_hkl, symref])
            hkl_idx = np.append(hkl_idx, [n] * len(symref))
            symmetric_idx[n] = True
        return hkl_list[symmetric_idx], intensity_list[symmetric_idx] / intensity_tot[symmetric_idx]

    def print_symmetric_vectors(self, HKL):
        """
        Print symmetric vectors
        :param HKL: [h,k,l] reflection
        :return: str
        """

        HKL = np.asarray(HKL, dtype=float).reshape((-1, 3))
        ops = np.array(self.symmetry_operations)
        out = ''
        for hkl in HKL:
            symHKL = self.symmetric_reflections(hkl)
            unique_hkl, uniqueidx, matchidx = fg.unique_vector(symHKL, tol=0.01)
            out += '--%s--\n' % fc.hkl2str(hkl)
            for n in range(len(unique_hkl)):
                symops = ops[np.array(matchidx) == n]
                symstr = '   '.join(symops)
                out += '  %2d %16s  : %2d : %s\n' % (n, fc.hkl2str(unique_hkl[n]), len(symops), symstr)
        return out

    def symmetric_magnetic_vectors(self, MXMYMZ):
        """
        NOT COMPLETE
        """

        r"""
        for n, (mx, my, mz) in enumerate(MXMYMZ):
            # Apply symmetry constraints
            if len(contraints) > 0:
                C = constraints
                C = C.replace('m','')
                #print('Constraint: {:3s} {:s}'.format(label[n],C))
                # convert '2x' to 'x'
                old = re.findall('\d[xyz]',C) # find number before x or y or z, e.g. 2x
                new = [s.replace('x','*x').replace('y','*y').replace('z','*z') for s in old]
                for s in range(len(old)):
                    C = C.replace(old[s],new[s])
                C = C.split(',')
                # Apply constraint to symmetry arguments
                S = [s.replace('x','a').replace('y','b').replace('z','c') for s in symmag]
                S = [s.replace('a',C[0]).replace('b',C[1]).replace('c',C[2]) for s in S]
            else:
                S = symmag
        """

        mx, my, mz = MXMYMZ
        sym_xyz = fc.gen_sym_pos(self.symmetry_operations_magnetic, mx, my, mz)
        return sym_xyz

    def axial_vector(self, uvw, remove_identical=True):
        """
        Perform symmetry operations on an axial vector uvw
        :param uvw: 3 element array/ list in cell coordinates
        :param remove_identical: True/ False, if True, identical operations are removed
        :return: [S*3]  array of transformed coordinates
        """

        u, v, w = uvw
        sym_uvw = fc.gen_sym_axial_vector(self.symmetry_operations, u, v, w)
        sym_uvw = fc.fitincell(sym_uvw)

        # Remove identical positions
        unique_uvw, uniqueidx, matchidx = fg.unique_vector(sym_uvw, tol=0.01)

        if remove_identical is None:
            return np.array(self.symmetry_operations)[uniqueidx]

    def reflection_multiplyer(self, HKL):
        """
        Returns the number of symmetric reflections for each hkl
        :param HKL: [nx3] array of [h,k,l]
        :return: [n] array of multiplyers
        """
        HKL = np.asarray(HKL, dtype=float).reshape((-1, 3))
        multiplyers = np.zeros(len(HKL))
        for n, hkl in enumerate(HKL):
            sym_hkl = self.symmetric_reflections_unique(hkl)
            multiplyers[n] = len(sym_hkl)
        if len(multiplyers) == 1:
            return multiplyers[0]
        return multiplyers

    def spacegroup_name(self):
        """Return the spacegroup name and number as str"""
        spg = self.spacegroup.replace(' ', '')
        spn = self.spacegroup_number
        return "{} ({})".format(spg, spn)

    def parity_time_info(self):
        """
        Returns string of parity and time operations for symmetry operations
        :return: str
        """
        # From functions_crystallography.symmetry_ops2magnetic
        operations = self.symmetry_operations
        magnetic_ops = self.symmetry_operations_magnetic
        # Convert operations to matrices
        mat_ops = fc.gen_sym_mat(operations)
        tim_ops = fc.sym_op_time(operations)

        out = 'Spacegoup: %s (%s)\n' % (self.spacegroup, self.spacegroup_number)
        out += '  n, Symmetry operations,   Time,    Parity,    T*P*M,  Magnetic Operation\n'
        for n, mat in enumerate(mat_ops):
            # Get time operation
            t = tim_ops[n]
            # Only use rotational part
            m = mat[:3, :3]
            # Get parity
            p = np.linalg.det(m)
            # Generate string
            mag_str = fc.sym_mat2str(t * p * m)
            out += "%3d, %19s, %6s, %9s, %8s, %19s\n" % (n, operations[n], t, p, mag_str, magnetic_ops[n])
        return out

    def wyckoff_labels(self, UVW):
        """
        Return Wyckoff site for position
        :param UVW: (u,v,w) or None to use xtl.Atoms.uvw()
        :return: list of Wyckoff site letters
        """
        return fc.wyckoff_labels(self.spacegroup_dict, UVW)

    def print_wyckoff_sites(self):
        """Return info str about Wycoff positions for this spacegroup"""

        spg = self.spacegroup_dict
        out = 'Spacegoup: %s (%s)\n' % (spg['space group name'], spg['space group number'])
        if 'parent number' in spg:
            spg = fc.spacegroup(self.spacegroup_dict['parent number'])
            out += ' Parent: %s (%s)\n' % (spg['space group name'], spg['space group number'])
        centring = spg['positions centring']
        out += '\nCentring operations: %2d : \n     ' % len(centring)
        out += '\n     '.join(centring)
        out += '\n\n'
        out += 'Wyckoff Sites\n'
        coordinates = spg['positions coordinates']
        multiplicity = spg['positions multiplicity']
        symmetry = spg['positions symmetry']
        wyckoff_letter = spg['positions wyckoff letter']
        for n in range(len(coordinates)):
            out += '%3s : %10s : %2s\n     ' % (wyckoff_letter[n], symmetry[n], multiplicity[n])
            out += '\n     '.join(coordinates[n])
            out += '\n\n'
        return out

    def info(self):
        """
        Prints the symmetry information
        :return: str
        """

        out = 'Spacegoup: %s (%s)\n' % (self.spacegroup, self.spacegroup_number)

        out += 'Symmetry operations:\n'
        for n in range(len(self.symmetry_operations)):
            x1 = self.symmetry_operations[n].strip('\"\'')
            x2 = self.symmetry_operations_magnetic[n].strip('\"\'')
            out += '%2d %25s %25s\n' % (n, x1, x2)
        # print 'Centring operations:'
        # for n in range(len(self.centring_operations)):
        #    x1 = self.centring_operations[n].strip('\"\'')
        #    x2 = self.centring_operations_magnetic[n].strip('\"\'').replace('x','mx').replace('y','my').replace('z','mz')
        #    print '%2d %25s %25s' %(n,x1,x2)
        out += '-----------------------------------------------------\n\n'
        return out

    def __repr__(self):
        return "%s, %d symmetry operations" % (self.spacegroup_name(), len(self.symmetry_operations))

    def __str__(self):
        return self.info()


class Superstructure(Crystal):
    """
    Generate a superstructure of the current cell
        a' = n1a + n2b + n3c
        b' = m1a + m2b + m3c
        c' = o1a + o2b + o3c
                OR
        [a',b',c'] = P[a,b,c]
                Where
        P = [[n1,n2,n3],
             [m1,m2,m3],
             [o1,o2,o3]]
    Returns a superstructure Crystal class:
        xtl = Crystal()
        su = Superstructure(xtl,[[2,0,0],[0,2,0],[0,0,1]])

    Superstructure Crystal classes have additional attributes compared with Crystal classes:
        su.P = P as given
        su.Parent = the parent Crystal Class
    And additional functions:
        su.calculateQ_parent >> indexes (h,k,l) coordinates in the frame same cartesian frame as the Parent structure
        su.superhkl2parent >> indexes (h,k,l) coordinates in the frame of the Parent structure
        su.parenthkl2super >> indexes parent (h,k,l) coordinates in the frame of superstructure

    Use >>hasattr(su,'Parent') to check is the current object is a
    superstructure Crystal class
    """

    # Parent = Crystal()
    # P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def __init__(self, Parent, P):
        """Initialise"""

        # Instatiate crystal attributes
        self.Cell = Cell()
        self.Symmetry = Symmetry()
        self.Atoms = Atoms()
        self.Structure = Atoms()

        self.name = Parent.name + ' supercell'
        self.P = np.asarray(P, dtype=float)
        self.Parent = Parent
        newUV = Parent.Cell.calculateR(P)
        self.new_cell(fl.basis2latpar(newUV))
        parent_cells_in_supercell = fc.calc_vol(P)
        self.scale = Parent.scale * parent_cells_in_supercell

        # Add exta functions
        self.Plot = PlottingSuperstructure(self)
        self.Scatter = Scattering(self)
        self.Properties = Properties(self)

    def generate_super_positions(self):
        """
        Generate the supercell and superstructure based on P and parent structure
        :return: None, set new atom positions
        """

        # Build lattice of points
        UVW = fc.genHKL(2 * np.max(self.P))
        real_lattice = self.Parent.Cell.calculateR(UVW)

        # Determine lattice points within the cell
        newUV = self.Parent.Cell.calculateR(self.P)
        indx_lattice = fc.indx(real_lattice, newUV)
        isincell = np.all(indx_lattice <= 0.99, axis=1) * np.all(indx_lattice > -0.01, axis=1)
        # for n in range(len(real_lattice)):
        #    print UVW[n,:],real_lattice[n,:],indx_lattice[n,:],isincell[n]
        UVW = UVW[isincell, :]
        # print UVW
        Ncell = len(UVW)

        # Increase scale to match size of cell
        self.scale = self.Parent.scale * Ncell

        # Get Parent atoms
        uvw, atom_type, label, occ, uiso, mxmymz = self.Parent.Structure.get()

        # Generate all atomic positions
        new_uvw = np.empty([0, 3])
        new_mxmymz = np.empty([0, 3])
        new_type = np.tile(atom_type, Ncell)
        new_label = np.tile(label, Ncell)
        new_occ = np.tile(occ, Ncell)
        new_uiso = np.tile(uiso, Ncell)
        for n in range(Ncell):
            latpos = np.array([UVW[n, 0] + uvw[:, 0], UVW[n, 1] + uvw[:, 1], UVW[n, 2] + uvw[:, 2]]).T
            latR = self.Parent.Cell.calculateR(latpos)
            latuvw = fc.indx(latR, newUV)

            # Append to P1 atoms arrays
            new_uvw = np.append(new_uvw, latuvw, axis=0)
            new_mxmymz = np.append(new_mxmymz, mxmymz, axis=0)

        self.new_atoms(u=new_uvw[:, 0], v=new_uvw[:, 1], w=new_uvw[:, 2],
                       type=new_type, label=new_label, occupancy=new_occ, uiso=new_uiso,
                       mxmymz=new_mxmymz)

    def set_scale(self):
        """
        Set scale parameter automatically
        Based on ratio of parent - to superstructure volume
        """
        vol_parent = self.Parent.Cell.volume()
        vol_super = self.Cell.volume()
        self.scale = self.Parent.scale * vol_super / vol_parent

    def superUV(self):
        """
        Returns the supercell unit vectors defined relative to the Parent cell
        """
        return self.Parent.Cell.calculateR(self.P)

    def superUVstar(self):
        """
        Returns the reciprocal supercell unit vectors defined relative to the Parent cell
        """
        return fc.RcSp(self.superUV())

    def parentUV(self):
        """
        Returns the parent unit vectors defined relative to the supercell
        """
        return np.dot(np.linalg.inv(self.P), self.Cell.UV())

    def parentUVstar(self):
        """
        Returns the parent reciprocal cell unit vectors defined relative to the supercell
        """
        return fc.RcSp(self.parentUV())

    def calculateQ_parent(self, super_hkl):
        """
        Indexes (h,k,l) coordinates in the frame same cartesian frame as the Parent structure
            Q = h'*a'* + k'*b'* + l'*c'*
        Where a'*, b'*, c'* are defined relative to the parent lattice, a*,b*,c*

            [qx,qy,qz] = calculateQ_parent([h',k',l'])
        """
        return np.dot(super_hkl, self.superUVstar())

    def superhkl2parent(self, super_HKL):
        """
        Indexes (h,k,l) coordinates in the frame of the Parent structure
            Q = h*a* + k*b* + l*c* = h'*a'* + k'*b'* + l'*c'*
            [h',k',l'] = Q/[a'*,b'*,c'*]

            [h,k,l] = superhkl2parent([h',k',l'])
        """

        Q = self.calculateQ_parent(super_HKL)
        return fc.indx(Q, self.Parent.Cell.UVstar())

    def parenthkl2super(self, parent_HKL):
        """
        Indexes (h,k,l) coordinates in the frame of the Parent structure
            Q = h*a* + k*b* + l*c* = h'*a'* + k'*b'* + l'*c'*
            [h',k',l'] = Q/[a'*,b'*,c'*]

            [h',k',l'] = parenthkl2super([h,k,l])
        """

        Q = self.Parent.Cell.calculateQ(parent_HKL)
        return fc.indx(Q, self.superUVstar())


if __name__ == '__main__':
    xtl = Crystal()
