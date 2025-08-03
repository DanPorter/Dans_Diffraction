"""
Module: functions_lattice.py

By Dan Porter, PhD
Diamond
2024


Version 1.0
Last updated: 17/10/24

Version History:
17/10/24 1.0.0  Created module extracting lattice functions from functions_crystallography

Acknoledgements:
    October 2024    Thanks to Lee Richter for pointing out the error in triclinic angles!

@author: DGPorter
"""

import numpy as np

__version__ = '1.0.0'


def choose_basis(name):
    """
    Return a basis function based on a name

    Available options:
        1. 'MaterialsProject': c || z, b* || y
        2. 'Vesta': a || x, c* || z
        3. 'BusingandLevy': c || z, a* || x, 'default'

    :param name: str name of basis
    :return: function
    """
    name = str(name).lower().replace('-', '').replace(' ', '')
    if name.lower() in ['1', 'mp', 'materialsproject', 'cz']:
        return basis_1
    if name.lower() in ['2', 'vesta', 'ax']:
        return basis_2
    if name.lower() in ['3', 'default', 'bl', 'busingandlevy', 'businglevy', 'by']:
        return basis_3
    raise KeyError("basis %s not recognised" % name)


def gen_lattice_parameters(*lattice_parameters, **kwargs):
    """
    Generate list of lattice parameters:
     a,b,c,alpha,beta,gamma = gen_lattice_parameters(*args)
    args:
      1 -> a=b=c=1,alpha=beta=gamma=90
      [1,2,3] -> a=1,b=2,c=3,alpha=beta=gamma=90
      [1,2,3,120] -> a=1,b=2,c=3,alpha=beta=90,gamma=120
      [1,2,3,10,20,30] -> a=1,b=2,c=3,alpha=10,beta=20,gamma=30
      1,2,3,10,20,30 -> a=1,b=2,c=3,alpha=10,beta=20,gamma=30
      a=1,b=2,c=3,alpha=10,beta=20,gamma=30 -> a=1,b=2,c=3,alpha=10,beta=20,gamma=30

    :param lattice_parameters: float or list in Angstroms & degrees
    :param kwargs: lattice parameters
    :return: a,b,c,alpha,beta,gamma
    """
    if lattice_parameters and np.size(lattice_parameters[0]) > 1:
        lattice_parameters = lattice_parameters[0]
    if len(lattice_parameters) not in [0, 1, 3, 4, 6]:
        raise Exception('Incorrect number of lattice parameters')

    defaults = dict(a=1.0, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0)
    defaults.update(kwargs)
    if defaults['b'] is None:
        defaults['b'] = defaults['a']
    if defaults['c'] is None:
        defaults['c'] = defaults['a']
    a, b, c, alpha, beta, gamma = defaults.values()

    if len(lattice_parameters) > 0:
        a = lattice_parameters[0]
        b = 1.0 * a
        c = 1.0 * a

    if len(lattice_parameters) > 1:
        b = lattice_parameters[1]
        c = lattice_parameters[2]

    if len(lattice_parameters) == 4:
        gamma = lattice_parameters[3]

    if len(lattice_parameters) == 6:
        alpha = lattice_parameters[3]
        beta = lattice_parameters[4]
        gamma = lattice_parameters[5]
    return a, b, c, alpha, beta, gamma


def basis_1(*lattice_parameters, **kwargs):
    """
    Generate direct-space basis-vectors [a, b, c] from lattice parameters
    Basis choice equivalent to that of materials project:
    https://github.com/materialsproject/pymatgen/blob/v2024.10.3/src/pymatgen/core/lattice.py#L39-L1702

        vector c || z-axis
        vector a rotated by beta about y-axis from +ve x-axis
        vector b* || y-axis

    Calculate the lattice positions:

        [[x, y, z]] = dot([[u, v, w]], [vec_a, vec_b, vec_c])

    :param lattice_parameters: float or list in Angstroms & degrees, see gen_lattice_parameters()
    :param kwargs: lattice parameters
    :returns: [3x3] array, as [vec_a, vec_b, vec_c] in Angstroms
    """
    a, b, c, alpha, beta, gamma = gen_lattice_parameters(*lattice_parameters, **kwargs)
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = np.clip(val, -1, 1)  # rounding errors may cause values slightly > 1
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]

    return np.array([vector_a, vector_b, vector_c])


def basis_2(*lattice_parameters, **kwargs):
    """
    Generate direct-space basis-vectors [a, b, c] from lattice parameters
    Basis choice equivalent to that of Vesta:
    https://github.com/materialsproject/pymatgen/blob/v2024.10.3/src/pymatgen/core/lattice.py#L39-L1702

        vector a || x-axis
        vector b rotated by gamma about z-axis from +ve y-axis
        vector c* || z-axis

    Calculate the lattice positions:

        [[x, y, z]] = dot([[u, v, w]], [vec_a, vec_b, vec_c])

    :param lattice_parameters: float or list in Angstroms & degrees, see gen_lattice_parameters()
    :param kwargs: lattice parameters
    :returns: [3x3] array, as [vec_a, vec_b, vec_c] in Angstroms
    """
    a, b, c, alpha, beta, gamma = gen_lattice_parameters(*lattice_parameters, **kwargs)
    """
    https://github.com/materialsproject/pymatgen/blob/v2024.10.3/src/pymatgen/core/lattice.py#L39-L1702
    Vesta - a along x, c* along z
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)
    c1 = c * cos_beta
    c2 = (c * (cos_alpha - (cos_beta * cos_gamma))) / sin_gamma

    vector_a = [float(a), 0.0, 0.0]
    vector_b = [b * cos_gamma, b * sin_gamma, 0]
    vector_c = [c1, c2, np.sqrt(c ** 2 - c1 ** 2 - c2 ** 2)]
    return np.array([vector_a, vector_b, vector_c])


def basis_3(*lattice_parameters, **kwargs):
    """
    Generate direct-space basis-vectors [a, b, c] from lattice parameters
    Basis choice equivalent to the inverse of the Bmatrix of W. R. Busing and H. A. Levy, Acta Cryst. (1967)
    https://docs.mantidproject.org/nightly/concepts/Lattice.html

        vector a* || x-axis
        vector b rotated by alpha about x-axis from +ve z-axis
        vector c || z-axis

    Calculate the lattice positions:

        [[x, y, z]] = dot([[u, v, w]], [vec_a, vec_b, vec_c])

    :param lattice_parameters: float or list in Angstroms & degrees, see gen_lattice_parameters()
    :param kwargs: lattice parameters
    :returns: [3x3] array, as [vec_a, vec_b, vec_c] in Angstroms
    """
    a, b, c, alpha, beta, gamma = gen_lattice_parameters(*lattice_parameters, **kwargs)

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    sqrt = np.sqrt
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)

    # Analytical solution of inv(Bmatrix) by SymPy
    basis = np.array([
        [
            a*sqrt(1 - ca**2*cb**2/(sa**2*sb**2) + 2*ca*cb*cg/(sa**2*sb**2) - cg**2/(sa**2*sb**2))*sb,
            (-a*ca*cb + a*cg)/sa,
            (-a*sqrt(
                1 - ca**2*cb**2/(sa**2*sb**2) + 2*ca*cb*cg/(sa**2*sb**2) - cg**2/(sa**2*sb**2)
            )*sb*ca*cg + a*sqrt(
                1 - ca**2*cb**2/(sa**2*sb**2) + 2*ca*cb*cg/(sa**2*sb**2) - cg**2/(sa**2*sb**2)
            )*sb*cb - a*sqrt(
                1 - ca**2*cg**2/(sa**2*sg**2) + 2*ca*cb*cg/(sa**2*sg**2) - cb**2/(sa**2*sg**2)
            )*sg*ca**2*cb + a*sqrt(
                1 - ca**2*cg**2/(sa**2*sg**2) + 2*ca*cb*cg/(sa**2*sg**2) - cb**2/(sa**2*sg**2)
            )*sg*ca*cg)/(sqrt(
                1 - ca**2*cg**2/(sa**2*sg**2) + 2*ca*cb*cg/(sa**2*sg**2) - cb**2/(sa**2*sg**2)
            )*sa**2*sg)
        ],
        [
            0,
            b*sqrt(
                1 - ca**2/(sb**2*sg**2) + 2*ca*cb*cg/(sb**2*sg**2) - cb**2*cg**2/(sb**2*sg**2)
            )*sg/sqrt(
                1 - ca**2*cb**2/(sa**2*sb**2) + 2*ca*cb*cg/(sa**2*sb**2) - cg**2/(sa**2*sb**2)
            ),
            b*sqrt(
                1 - ca**2/(sb**2*sg**2) + 2*ca*cb*cg/(sb**2*sg**2) - cb**2*cg**2/(sb**2*sg**2)
            )*sg*ca/(sqrt(
                1 - ca**2*cb**2/(sa**2*sb**2) + 2*ca*cb*cg/(sa**2*sb**2) - cg**2/(sa**2*sb**2)
            )*sa)
        ],
        [0, 0, c]
    ])
    return basis


def busingandlevy(*lattice_parameters, **kwargs):
    """
    Calculate the Busing and Levy B-matrix from lattice parameters
        "we choose the x-axis parallel to a*, the y-axis in the plane of a* and b*,
        and the z-axis perpendicular to that plane"
    From: W. R. Busing and H. A. Levy, Acta Cryst. (1967). 22, 457-464
        "Angle calculations for 3- and 4-circle X-ray and neutron diffractometers"
    See also: https://docs.mantidproject.org/nightly/concepts/Lattice.html

    Creates a matrix to transform (hkl) into a cartesian basis:
        (qx,qy,qz)' = B.(h,k,l)'       (where ' indicates a column vector)

    The B matrix is related to the reciprocal basis vectors:
        (astar, bstar, cstar) = 2 * np.pi * B.T
    Where cstar is defined along the z-axis

    :param lattice_parameters: float or list in Angstroms & degrees, see gen_lattice_parameters()
    :param kwargs: lattice parameters
    :returns: [3x3] array B matrix in inverse-Angstroms (no 2pi)
    """
    a, b, c, alpha, beta, gamma = gen_lattice_parameters(*lattice_parameters, **kwargs)

    alpha1 = np.deg2rad(alpha)
    alpha2 = np.deg2rad(beta)
    alpha3 = np.deg2rad(gamma)

    beta1 = np.arccos((np.cos(alpha2) * np.cos(alpha3) - np.cos(alpha1)) / (np.sin(alpha2) * np.sin(alpha3)))
    beta2 = np.arccos((np.cos(alpha1) * np.cos(alpha3) - np.cos(alpha2)) / (np.sin(alpha1) * np.sin(alpha3)))
    beta3 = np.arccos((np.cos(alpha1) * np.cos(alpha2) - np.cos(alpha3)) / (np.sin(alpha1) * np.sin(alpha2)))

    b1 = 1 / (a * np.sin(alpha2) * np.sin(beta3))
    b2 = 1 / (b * np.sin(alpha3) * np.sin(beta1))
    b3 = 1 / (c * np.sin(alpha1) * np.sin(beta2))

    # c1 = b1 * b2 * np.cos(beta3)
    # c2 = b1 * b3 * np.cos(beta2)
    # c3 = b2 * b3 * np.cos(beta1)

    bmatrix = np.array([
        [b1, b2 * np.cos(beta3), b3 * np.cos(beta2)],
        [0, b2 * np.sin(beta3), -b3 * np.sin(beta2) * np.cos(alpha1)],
        [0, 0, 1 / c]
    ])
    return bmatrix


def basis2bandl(basis):
    """
    Calculate the Busing and Levy B matrix from a real space UV
    "choose the x-axis parallel to a*, the y-axis in the plane of a* and b*, and the z-axis perpendicular to that plane"
    From: W. R. Busing and H. A. Levy, Acta Cryst. (1967). 22, 457-464
    "Angle calculations for 3- and 4-circle X-ray and neutron diffractometers"
    See also: https://docs.mantidproject.org/nightly/concepts/Lattice.html

    B = [[b1, b2 * cos(beta3), b3 * cos(beta2)],
        [0, b2 * sin(beta3), -b3 * sin(beta2) * cos(alpha1)],
        [0, 0, 1 / a3]]
    return 2pi * B  # equivalent to transpose([a*, b*, c*])
    """
    return reciprocal_basis(basis).T


def angles_allowed(alpha=90, beta=90, gamma=90):
    """
    Determine if lattice angles are suitable for basis vectors

    https://journals.iucr.org/a/issues/2011/01/00/au5114/index.html
    As reported in International Tables for Crystallography:
        Donnay & Donnay, 1959 International Tables for X-ray Crystallography, Vol. II.
        Koch, 2004 International Tables for Crystallography, Vol. C.

    :param alpha: angle in degrees
    :param beta: angle in degrees
    :param gamma: angle in degrees
    :return: bool, True if angles are suitable for creation of basis
    """
    a = np.radians(alpha)
    b = np.radians(beta)
    g = np.radians(gamma)
    p = 4 * np.sin((a + b + g)/2) * np.sin((a + b - g)/2) * np.sin((a + g - b)/2) * np.sin((b + g - a)/2)
    return p > 0


def random_lattice(symmetry='triclinic'):
    """
    Return a random set of real lattice parameters
    :param symmetry: string 'cubic', 'tetragona', 'rhobohedral', 'monoclinic-a/b/c', 'triclinic'
    :return: (a, b, c, alpha, beta, gamma) lattice parameters in Angstroms/ degrees
    """
    def r_len():
        return np.random.normal(5., 2.0)

    def r_ang():
        return np.random.normal(90., 20.)

    if symmetry == 'cubic':
        latt = 3 * [r_len()] + [90, 90, 90]
    elif symmetry == 'tetragonal':
        latt = 2 * [r_len()] + [r_len()] + [90, 90, 90]
    elif symmetry == 'hexagonal':
        latt = 2 * [r_len()] + [r_len()] + [90, 90, 120]
    elif symmetry == 'rhobohedral':
        latt = 3 * [r_len()] + 3 * [r_ang()]
    elif symmetry == 'monoclinic-a':
        latt = [r_len(), r_len(), r_len(), r_ang(), 90, 90]
    elif symmetry == 'monoclinic-b':
        latt = [r_len(), r_len(), r_len(), 90, r_ang(), 90]
    elif symmetry == 'monoclinic-c':
        latt = [r_len(), r_len(), r_len(), 90, 90, r_ang()]
    else:
        triclinic_angles = [r_ang(), r_ang(), r_ang()]
        while not angles_allowed(*triclinic_angles):
            triclinic_angles = [r_ang(), r_ang(), r_ang()]
        latt = [r_len(), r_len(), r_len()] + triclinic_angles
    return tuple(latt)


def random_basis(symmetry='triclinic', basis_option='default'):
    """
    Generate a random basis of unit vectors from a real set of lattice parameters
    :param symmetry: string 'cubic', 'tetragona', 'rhobohedral', 'monoclinic-a/b/c', 'triclinic'
    :param basis_option: str name of basis, 'materialsproject', 'vesta', 'busingandlevy'
    :return: [3x3] array, as [vec_a, vec_b, vec_c] in Angstroms
    """
    basis_function = choose_basis(basis_option)
    latt = random_lattice(symmetry)
    return basis_function(*latt)


def lattice_volume(*lattice_parameters, **kwargs):
    """
    Calculate basis volume from lattice parameters

        volume = vec_a . (vec_b X vec_c)

    :param lattice_parameters: float or list in Angstroms & degrees, see gen_lattice_parameters()
    :param kwargs: lattice parameters
    :returns: float in Angstroms cubed
    """
    a, b, c, alpha, beta, gamma = gen_lattice_parameters(*lattice_parameters, **kwargs)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    p = (4 *
         np.sin((alpha + beta + gamma) / 2) *
         np.sin((alpha + beta - gamma) / 2) *
         np.sin((alpha + gamma - beta) / 2) *
         np.sin((beta + gamma - alpha) / 2)
         )
    return np.sqrt(a**2 * b**2 * c**2 * p)


def dspacing(h, k, l, *lattice_parameters, **kwargs):
    """
    Calculate the lattice d-spacing for Bragg reflection (h,k,l)

        d = lambda / 2 * sin(theta) = 2*pi / |h.a* + k.b* + l.c*|

    :param h, k, l: Miller-indices of reflection
    :param lattice_parameters: float or list in Angstroms & degrees, see gen_lattice_parameters()
    :param kwargs: lattice parameters
    :return: float in Angstroms
    """
    a, b, c, alpha, beta, gamma = gen_lattice_parameters(*lattice_parameters, **kwargs)
    v = lattice_volume(*lattice_parameters, **kwargs)
    s2a = np.sin(np.radians(alpha)) ** 2
    s2b = np.sin(np.radians(beta)) ** 2
    s2g = np.sin(np.radians(gamma)) ** 2
    ca = np.cos(np.radians(alpha))
    cb = np.cos(np.radians(beta))
    cg = np.cos(np.radians(gamma))

    s11 = (b ** 2 * c ** 2 * s2a) * h ** 2
    s22 = (a ** 2 * c ** 2 * s2b) * k ** 2
    s33 = (a ** 2 * b ** 2 * s2g) * l ** 2
    s12 = (a * b * c * c * (ca * cb - cg)) * 2 * h * k
    s23 = (a * a * b * c * (cb * cg - ca)) * 2 * k * l
    s13 = (a * b * b * c * (cg * ca - cb)) * 2 * h * l
    return np.sqrt(v ** 2 / (s11 + s22 + s33 + s12 + s23 + s13))


def reciprocal_basis(basis_vectors):
    """
    Return the reciprocal basis vectors
        [a*, b*, c*] = 2*pi*inv([a, b, c]).T

    :param basis_vectors: [3*3] basis vectors array [a[3], b[3], c[3]]
    :return: [3*3] array of reciprocal vectors [a*[3], b*[3], c*[3]]
    """
    return 2 * np.pi * np.linalg.inv(basis_vectors).T


def reciprocal_lattice_parameters(*lattice_parameters, **kwargs):
    """
    Return the reciprocal lattice parameters in inverse-angstroms and degrees
    :param lattice_parameters: float or list in Angstroms & degrees, see gen_lattice_parameters()
    :param kwargs: lattice parameters
    :return: a*, b*, c*, alpha*, beta*, gamma*
    """
    a, b, c, alpha, beta, gamma = gen_lattice_parameters(*lattice_parameters, **kwargs)

    alpha1 = np.deg2rad(alpha)
    alpha2 = np.deg2rad(beta)
    alpha3 = np.deg2rad(gamma)

    beta1 = np.arccos((np.cos(alpha2) * np.cos(alpha3) - np.cos(alpha1)) / (np.sin(alpha2) * np.sin(alpha3)))
    beta2 = np.arccos((np.cos(alpha1) * np.cos(alpha3) - np.cos(alpha2)) / (np.sin(alpha1) * np.sin(alpha3)))
    beta3 = np.arccos((np.cos(alpha1) * np.cos(alpha2) - np.cos(alpha3)) / (np.sin(alpha1) * np.sin(alpha2)))

    b1 = 1 / (a * np.sin(alpha2) * np.sin(beta3))
    b2 = 1 / (b * np.sin(alpha3) * np.sin(beta1))
    b3 = 1 / (c * np.sin(alpha1) * np.sin(beta2))
    return b1, b2, b3, np.rad2deg(beta1), np.rad2deg(beta2), np.rad2deg(beta3)


def index_lattice(coords, basis_vectors):
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
    return np.dot(coords, np.linalg.inv(basis_vectors))


def basis2latpar(basis_vectors):
    """
    Convert UV=[a,b,c] to a,b,c,alpha,beta,gamma
     a,b,c,alpha,beta,gamma = UV2latpar(UV)

    :param basis_vectors: [3*3] basis vectors array [a[3], b[3], c[3]]
    """
    av, bv, cv = basis_vectors
    a = np.sqrt(np.sum(np.square(av)))
    b = np.sqrt(np.sum(np.square(bv)))
    c = np.sqrt(np.sum(np.square(cv)))

    def cal_angle(vec1, vec2):
        crs = np.sqrt(np.sum(np.square(np.cross(vec1, vec2))))
        dt = np.dot(vec1, vec2)
        return np.rad2deg(np.arctan2(crs, dt))

    alpha = cal_angle(bv, cv)
    beta = cal_angle(av, cv)
    gamma = cal_angle(av, bv)
    return a, b, c, alpha, beta, gamma
