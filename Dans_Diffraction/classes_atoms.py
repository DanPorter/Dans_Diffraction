"""

"""
import os
from warnings import warn
import numpy as np
from . import functions_general as fg
from . import functions_crystallography as fc


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

props = fc.atom_properties()
ATOM_PROPERTIES = {
    e: {
        p: props[n][p] for p in props.dtype.names
    } for n, e in enumerate(props['Element'])
}

def get_atom_properties(symbol):
    """Return all atom properties"""

    name, occ, charge = fc.split_element_symbol(symbol)
    fc.atom_properties(symbol)

    data = np.loadtxt(WAASKIRF_FILE)
    # get names
    with open(WAASKIRF_FILE) as f:
        lines = re.findall(r'#S\s+\d+\s+[A-Z].*?\n', f.read())
        table_names = [line[7:].strip() for line in lines]
    idx = [table_names.index(el) for el in element]
    coef = data[idx, :]

    xma_data = np.loadtxt(XMAFILE)
    xma_data[:, z]

    asf = np.load(ASFFILE, allow_pickle=True)
    asf = asf.item()
    z = asf[el]['Z']
    energy = np.array(asf[el]['energy']) / 1000.  # eV -> keV
    f1 = np.array(asf[el]['f1'])
    f2 = np.array(asf[el]['f2'])
    f1[f1 < -1000] = np.nan
    f2[f2 < -1000] = np.nan


    try:
        data = np.genfromtxt(PENGFILE, skip_header=0, dtype=None, names=True, encoding='ascii', delimiter=',')
    except TypeError:
        # Numpy version < 1.14
        data = np.genfromtxt(PENGFILE, skip_header=0, dtype=None, names=True, delimiter=',')
    # elements must be a list e.g. ['Co','O']
    elements_l = np.asarray(element).reshape(-1)
    all_elements = [el for el in data['Element']]

    nsl = read_neutron_scattering_lengths(table)  # 'sears', 'itc'
    nsl[element]


class AtomType:
    """
    Defines the element type and symbol, stores atom properties
    """
    Z: int
    Element: str
    Name: str
    Group: int
    Period: int
    Block: str
    ValenceE: int = 1
    Config: str = '1s1'
    Radii: float = 25
    Weight: float = 1
    Coh_b: float = 0
    Inc_b: float = 0
    Nabs: float = 0
    Nscat: float = 0
    xray_scattering_factor_pars: tuple = (1, 0)
    electron_scattering_factor_pars: tuple = (1, 0)
    magnetic_form_factor_pars: tuple = (0, 0)
    K: float = 0
    L1: float = 0
    L2: float = 0
    L3: float = 0
    M1: float = 0
    M2: float = 0
    M3: float = 0
    M4: float = 0
    M5: float = 0
    N1: float = 0
    N2: float = 0
    N3: float = 0

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        try:
            symbol_name, occupancy, charge = fc.split_element_symbol(symbol)
            properties = ATOM_PROPERTIES.get(symbol_name, {})
            properties.update(kwargs)
            self.__dict__.update(properties)
        except IndexError:
            pass

    def __repr__(self):
        return f"AtomType('{self.symbol}')"

    def __str__(self):
        out = f"{repr(self)}\n"
        out += '\n'.join(f"{prop:>12}: {val}" for prop, val in self.__dict__.items())
        out += '\n'
        return out

    def _load_xray_scattering_factors(self):
        symbol_name, occupancy, charge = fc.split_element_symbol(self.symbol)


    def _load_properties(self, **kwargs):
        symbol_name, occupancy, charge = fc.split_element_symbol(self.symbol)
        properties = ATOM_PROPERTIES.get(symbol_name, {}).copy()
        properties.update(kwargs)

        self.Z = properties['Z']
        self.Element = properties['Element']
        self.Name = properties['Name']
        self.Group = properties['Group']
        self.Period = properties['Period']
        self.Block = properties['Block']
        self.ValenceE = properties['ValenceE']
        self.Config = properties['Config']
        self.Radii = properties['Radii']
        self.Weight = properties['Weight']
        self.Coh_b = properties['Coh_b']
        self.Inc_b = properties['Inc_b']
        self.Nabs = properties['Nabs']
        self.Nscat = properties['Nscat']
        self.K = properties['K']
        self.L1 = properties['L1']
        self.L2 = properties['L2']
        self.L3 = properties['L3']
        self.M1 = properties['M1']
        self.M2 = properties['M2']
        self.M3 = properties['M3']
        self.M4 = properties['M4']
        self.M5 = properties['M5']
        self.N1 = properties['N1']
        self.N2 = properties['N2']
        self.N3 = properties['N3']


class AtomSite:
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
            self.type = np.asarray([self._default_atom] * Natoms)
        else:
            self.type = np.asarray(type, dtype=str).reshape(-1)
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
            idx = self.label.index(idx)
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
        self.type = element
        self.label = label
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

    def changeatom(self, idx=None, u=None, v=None, w=None, type=None,
                   label=None, occupancy=None, uiso=None, mxmymz=None):
        """
        Change an atoms properties
        :param idx:
        :param u:
        :param v:
        :param w:
        :param type:
        :param label:
        :param occupancy:
        :param uiso:
        :param mxmymz:
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
            self.label = np.array(old_labels)

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
        type = self.type
        rem_atom_idx = []
        for n in range(0, len(type) - 1):
            match_position = fg.mag(uvw[n, :] - uvw[n + 1:, :]) < min_distance
            match_type = type[n] == type[n + 1:]
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

