"""
Crystal Orientation Class

By Dan Porter, PhD
Diamond
2021

Version 1.0.0
Last updated: 27/09/21

Version History:
26/09/21 0.1.0  Version History started.
27/09/21 1.0.0  Initial version finished

@author: DGPorter
"""

import numpy as np

from . import functions_lattice as fl
from . import functions_crystallography as fc

__version__ = '1.0.0'


def string_vector(v):
    """Convert [3] array to string"""
    return '[%5.3f,%5.3f,%5.3f]' % (v[0], v[1], v[2])


def string_matrix(m):
    """Convert [3*3] array to string"""
    x = string_vector(m[0])
    y = string_vector(m[1])
    z = string_vector(m[2])
    return '[%s, %s, %s]' % (x, y, z)


class Orientation:
    """
    CrystalOrientation Class
    Contains [3*3] matrices to control the orientation of a crystal in space
        The crystal is initially defined in the "Diffractometer Frame"
        "umatrix" controls the initial alignment of the cell with respect to this frame
        "rotation" rotates the cell as if on a diffractometer
        "labframe" is a transformation matrix from the "Diffractometer Frame" to a frame of your choice

    E.G.
        o = Orientation()
        o.orient(a_axis=[1,0,0], c_axis=[0,0,1])
        o.rotate_6circle(chi=90, eta=20)
        o.set_lab_i16()
        o([0,0,1])
        returns > [ 0.    ,  0.9397, -0.342 ]

    --- The Diffractometer Frame ---
    Diffractometer frame according to Fig. 1, H. You, J. Appl. Cryst 32 (1999), 614-623
      z-axis : axis parallel to the phi rotation axis when all angles 0 (towards wall (+x) in lab frame)
      x-axis : vector normal to phi axis where phi=0 (toward ceiling (+y) in lab frame)
      y-axis : vector normal to x,z axes (parallel to beam (+z) in lab frame)
    """

    def __init__(self, umatrix=None, rotation=None, labframe=None):
        # Default orientation matrix in diffractometer frame
        if umatrix is None:
            self.umatrix = np.eye(3)
        else:
            self.set_u(umatrix)
        # Default rotation matrix in diffractometer frame
        if rotation is None:
            self.rotation = np.eye(3)
        else:
            self.set_r(rotation)
        # Default lab-frame in units of diffractometer-frame
        if labframe is None:
            self.labframe = np.eye(3)
        else:
            self.set_lab(labframe)

    def set_u(self, umatrix):
        """Set oritenation matrix in diffractometer frame"""
        self.umatrix = np.asarray(umatrix, dtype=float).reshape(3, 3)

    def orient(self, a_axis=None, b_axis=None, c_axis=None):
        """Set orientation matrix from directions of crystal axes"""
        self.umatrix = fc.umatrix(a_axis, b_axis, c_axis)

    def random_orientation(self, a_axis=None, b_axis=None, c_axis=None):
        """Set a random orientation matrix"""
        ax1 = np.random.rand(3)
        try:
            if a_axis is not None:
                self.umatrix = fc.umatrix(a_axis=a_axis, c_axis=ax1)
            elif b_axis is not None:
                self.umatrix = fc.umatrix(a_axis=ax1, b_axis=b_axis)
            elif c_axis is not None:
                self.umatrix = fc.umatrix(a_axis=ax1, c_axis=c_axis)
            else:
                ax2 = np.random.rand(3)
                self.umatrix = fc.umatrix(a_axis=ax1, b_axis=ax2)
        except Exception:
            # Catch parallel vectors
            self.random_orientation(a_axis, b_axis, c_axis)

    def set_r(self, rotation):
        """Set rotation matrix in diffractometer frame"""
        self.rotation = np.asarray(rotation, dtype=float).reshape(3, 3)

    def rotate_6circle(self, phi=0, chi=0, eta=0, mu=0):
        """Set rotation matrix using 6-circle diffractometer axes"""
        self.rotation = fc.diffractometer_rotation(phi, chi, eta, mu)

    def set_lab(self, lab):
        """Set transformation matrix between diffractometer and lab"""
        self.labframe = np.asarray(lab, dtype=float).reshape(3, 3)

    def set_lab_i16(self):
        """Set lab transformation matrix for beamline I16 at Diamond Light Source"""
        self.set_lab([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    def clear(self):
        """Clear orientation"""
        self.umatrix = np.eye(3)
        self.rotation = np.eye(3)
        self.labframe = np.eye(3)

    def vector(self, vec):
        """Return vector transformed by orientation, rotation and lab transformation"""
        return fc.labvector(vec, self.umatrix, self.rotation, self.labframe)

    def diff6circle(self, delta=0, gamma=0, energy_kev=None, wavelength=1.0):
        """
        Calcualte wavevector in diffractometer axis using detector angles
        :param delta: float angle in degrees in vertical direction (about diff-z)
        :param gamma: float angle in degrees in horizontal direction (about diff-x)
        :param energy_kev: float energy in KeV
        :param wavelength: float wavelength in A
        :param lab: [3*3] lab transformation matrix
        :return: q[1*3], ki[1*3], kf[1*3]
        """
        q = fc.diff6circleq(delta, gamma, energy_kev, wavelength, lab=self.labframe)
        ki, kf = fc.diff6circlek(delta, gamma, energy_kev, wavelength, lab=self.labframe)
        return q, ki, kf

    def __call__(self, vec):
        return self.vector(vec)

    def __repr__(self):
        umatrix = string_matrix(self.umatrix)
        rot = string_matrix(self.rotation)
        lab = string_matrix(self.labframe)
        return "Orientation(umatrix=%s, rotation=%s, labframe=%s)" % (umatrix, rot, lab)

    def __str__(self):
        umatrix = string_matrix(self.umatrix)
        rot = string_matrix(self.rotation)
        lab = string_matrix(self.labframe)
        s = "Orientation(\n"
        s += "    umatrix=%s\n" % umatrix
        s += "    rotation=%s\n" % rot
        s += "    labframe=%s\n)" % lab
        return s


class CrystalOrientation:
    """
    CrystalOrientation Class

    Define an orientation matrix in the diffractometer frame
    Diffractometer frame according to Fig. 1, H. You, J. Appl. Cryst 32 (1999), 614-623
      z-axis : axis parallel to the phi rotation axis when all angles 0 (towards wall (+x) in lab frame)
      x-axis : vector normal to phi axis where phi=0 (toward ceiling (+y) in lab frame)
      y-axis : vector normal to x,z axes (parallel to beam (+z) in lab frame)
    """
    def __init__(self, *lattice_parameters, **kwargs):
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = fc.gen_lattice_parameters(
            *lattice_parameters,
            **kwargs
        )
        self.orientation = Orientation()

    def __repr__(self):
        return 'CrystalOrientation%s' % str(self.lp())

    def __str__(self):
        a, b, c = self.unit_vectors()
        astar, bstar, cstar = self.reciprocal_unit_vectors()
        s = '  a = %s\n' % string_vector(a)
        s += '  b = %s\n' % string_vector(b)
        s += '  c = %s\n\n' % string_vector(c)

        s += '  a* = %s\n' % string_vector(astar)
        s += '  b* = %s\n' % string_vector(bstar)
        s += '  c* = %s\n\n' % string_vector(cstar)
        return s

    def set_latt(self, a=None, b=None, c=None, alpha=None, beta=None, gamma=None):
        """Set or change lattice parameters"""
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

    def lp(self):
        """Return tuple of lattice parameters in angstroms and degrees"""
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def lp_star(self):
        """Return tuple of reciprocal lattice parameters in inverse-angstroms and degrees"""
        return fl.reciprocal_lattice_parameters(*self.lp())

    def _uv(self):
        """Return unit vectors [a,b,c] in default frame (a* along X)"""
        return fl.basis_3(*self.lp())

    def _uvstar(self):
        """Return unit vectors in recirpocal space in default frame (a* along X)"""
        return fl.reciprocal_basis(self._uv())

    def unit_vectors(self):
        """Return real space unit vectors [a, b, c]"""
        return self.orientation(self._uv())

    def reciprocal_unit_vectors(self):
        """Return unit vectors in recirpocal space [a*, b*, c*]"""
        return self.orientation(self._uvstar())

    def bmatrix(self):
        """Return the B matrix from Busing & Levy in the diffractometer frame"""
        return fl.busingandlevy(*self.lp())

    def ubmatrix(self):
        """Return UB matrix from Busing & Levy in the diffractometer frame"""
        return fc.ubmatrix(self._uv(), self.orientation.umatrix)

    def realspace(self, uvw):
        """Generate vector in real space from uvw = [u*a, v*b, w*c]"""
        uvw = np.reshape(np.asarray(uvw, dtype=float), [-1, 3])
        uv = self._uv()
        r = np.dot(uvw, uv)
        return self.orientation(r)

    def recspace(self, hkl):
        """Generate vector in reciprocal space from hkl = [h*a*, k*b*, l*c*]"""
        hkl = np.reshape(np.asarray(hkl, dtype=float), [-1, 3])
        uvs = self._uvstar()
        q = np.dot(hkl, uvs)
        return self.orientation(q)

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

