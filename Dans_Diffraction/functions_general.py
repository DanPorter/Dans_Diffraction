# -*- coding: utf-8 -*-
"""
Module: Generally useful functions "functions_general.py"

Contains various useful shortcuts for manipulating strings and arrays,
making use of numpy and re.

By Dan Porter, PhD
Diamond
2021

Version 2.0.0
Last updated: 02/02/21

Version History:
06/01/18 1.0    Program created from DansGeneralProgs.py V2.3
02/05/18 1.1    Added find_vector
24/05/18 1.2    Corrected 'quad' for the case (1,-2,1)=1
31/10/18 1.3    Added complex2str
20/08/19 1.4    Added search_dict_lists
27/03/20 1.5    Corrected error in gauss for 1d case when centre /= 0
05/05/20 1.6    New version of readstfm, allows E powers and handles non-numbers.
12/05/20 1.7    Added sph2cart, replace_bracket_multiple
20/07/20 1.8    Added vector_inersection and plane_intersection, updated findranges, added whererun
01/12/20 1.8.1  Added get_methods function
26/01/21 1.8.2  Added shortstr and squaredata
02/02/21 2.0.0  Merged changes in other versions, added vector_intersection and you_normal_vector

@author: DGPorter
"""

import sys, os, re
import numpy as np
import inspect

__version__ = '2.0.0'
__date__ = '02/Feb/2021'

# File directory
directory = os.path.abspath(os.path.dirname(__file__))

# Constants
pi = np.pi  # mmmm tasty Pi
e = 1.6021733E-19  # C  electron charge
h = 6.62606868E-34  # Js  Plank consant
c = 299792458  # m/s   Speed of light
u0 = 4 * pi * 1e-7  # H m-1 Magnetic permeability of free space
me = 9.109e-31  # kg Electron rest mass
Na = 6.022e23  # Avagadro's No
A = 1e-10  # m Angstrom
r0 = 2.8179403227e-15  # m classical electron radius = e^2/(4pi*e0*me*c^2)
Cu = 8.048  # Cu-Ka emission energy, keV
Mo = 17.4808  # Mo-Ka emission energy, keV
# Mo = 17.4447 # Mo emission energy, keV

'--------------------------Misc General Programs------------------------'

'---------------------------Vector manipulation-------------------------'


def mag(A):
    """
    Returns the magnitude of vector A
    If A has 2 dimensions, returns an array of magnitudes

    E.G.
     mag([1,1,1]) = 1.732
     mag(array([[1,1,1],[2,2,2]]) = [1.732, 3.464]
    """
    A = np.asarray(A, dtype=np.float)
    return np.sqrt(np.sum(A ** 2, axis=len(A.shape) - 1))


def norm(A):
    """
    Returns normalised vector A
    If A has 2 dimensions, returns an array of normalised vectors
    The returned array will be the same shape as the given array.

    E.G.
     norm([1,1,1]) = [1,1,1]/1.732 = [ 0.57735,  0.57735,  0.57735]
     norm(array([[1,1,1],[2,2,2]]) = [ [ 0.57735,  0.57735,  0.57735] , [ 0.57735,  0.57735,  0.57735] ]
     """

    A = np.asarray(A, dtype=np.float).reshape((-1, np.shape(A)[-1]))
    mag = np.sqrt(np.sum(A ** 2, axis=A.ndim - 1)).reshape((-1, 1))
    mag[mag == 0] = 1  # stop warning errors
    N = A / mag
    if A.shape[0] == 1:
        N = N.reshape(-1)
    return N


def quad(A):
    """
    Returns +/-1 depending on quadrant of 3D vector A
    i.e.:    A      Returns
        [ 1, 1, 1]    1
        [-1, 1, 1]    1
        [ 1,-1, 1]    1
        [ 1, 1,-1]    1
        [-1,-1, 1]   -1
        [-1, 1,-1]   -1
        [ 1,-1,-1]   -1
        [-1,-1,-1]   -1
    """

    A = np.asarray(A, dtype=np.float).reshape((-1, 3))
    if A.size == 1:
        # return (np.sum(A,axis=1)>0)[0]*2 - 1
        return (np.sum(A >= 0, axis=1) > 1)[0] * 2 - 1
    else:
        # return (np.sum(A,axis=1)>0)*2 - 1
        return (np.sum(A >= 0, axis=1) > 1) * 2 - 1


def quadmag(A):
    """
    Returns +/- mag depending on quadrant of a 3D vector A
    i.e.:    A      Returns
        [ 1, 1, 1]    1.732
        [-1, 1, 1]    1.732
        [ 1,-1, 1]    1.732
        [ 1, 1,-1]    1.732
        [-1,-1, 1]   -1.732
        [-1, 1,-1]   -1.732
        [ 1,-1,-1]   -1.732
        [-1,-1,-1]   -1.732
    """

    A = np.asarray(A, dtype=np.float).reshape((-1, 3))
    mag = np.sqrt(np.sum(A ** 2, axis=len(A.shape) - 1)).reshape(-1)
    quad = (np.sum(A, len(A.shape) - 1) > 0) * 2 - 1
    if len(A) == 1:
        return quad[0] * mag[0]
    else:
        return quad * mag


def ang(a, b, deg=False):
    """
    Returns the angle, in Radians between vectors a and b
    E.G.
    ang([1,0,0],[0,1,0]) >> 1.571
    ang([1,0,0],[0,1,0],'deg') >> 90
    """
    a = np.asarray(a, dtype=float).reshape([-1])
    b = np.asarray(b, dtype=float).reshape([-1])

    cosang = np.dot(a, b)
    sinang = quadmag(np.cross(a, b))
    angle = np.arctan2(sinang, cosang)
    if deg:
        return np.rad2deg(angle)
    return angle


def cart2sph(xyz, deg=False):
    """
    Convert coordinates in cartesian to coordinates in spherical
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    ISO convention used.
        theta = angle from Z-axis to X-axis
          phi = angle from X-axis to component in XY plane
    :param xyz: [n*3] array of [x,y,z] coordinates
    :param deg: if True, returns angles in degrees
    :return: [r, theta, phi]
    """
    xyz = np.asarray(xyz).reshape(-1, 3)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    r = mag(xyz)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # theta = np.arctan2(xyz[:,2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    if deg:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
    return np.vstack((r, theta, phi)).T


def rot3D(A, alpha=0., beta=0., gamma=0.):
    """Rotate 3D vector A by euler angles
        A = rot3D(A,alpha=0.,beta=0.,gamma=0.)
       where alpha = angle from X axis to Y axis (Yaw)
             beta  = angle from Z axis to X axis (Pitch)
             gamma = angle from Y axis to Z axis (Roll)
       angles in degrees
       In a right handed coordinate system.
           Z
          /|\
           |
           |________\Y
           \        /
            \
            _\/X
    """

    A = np.asarray(A, dtype=np.float).reshape((-1, 3))

    # Convert to radians
    alpha = alpha * np.pi / 180.
    beta = beta * np.pi / 180.
    gamma = gamma * np.pi / 180.

    # Define 3D rotation matrix
    Rx = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0., np.sin(gamma), np.cos(gamma)]])
    Ry = np.array([[np.cos(beta), 0., np.sin(beta)], [0., 1., 0.], [-np.sin(beta), 0., np.cos(beta)]])
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0.], [np.sin(alpha), np.cos(alpha), 0.], [0., 0., 1.]])
    R = np.dot(np.dot(Rx, Ry), Rz)

    # Rotate coordinates
    return np.dot(R, A.T).T


def rotmat(a, b):
    """
    Determine rotation matrix from a to b
    From Kuba Ober's answer on stackexchange:
    http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Usage:
     a = [1,0,0]
     b = [0,1,0]
     U = rotmat(a,b)
     np.dot(U,a)
     >> [0,1,0]
    """

    a = norm(a)
    b = norm(b)

    d = np.dot(a, b)
    c = mag(np.cross(a, b))
    G = np.array([[d, -c, 0], [c, d, 0], [0, 0, 1]])

    u = a
    v = norm(b - np.dot(a, b) * a)
    w = np.cross(b, a)

    Fm1 = np.array([u, v, w]).T
    F = np.linalg.inv(Fm1)
    U = np.dot(np.dot(Fm1, G), F)

    # print('a = {}\nb = {}'.format(str(a),str(b)))
    # print('G = {}\nFm1 = {}\nF = {}\nU = {}'.format(str(G),str(Fm1),str(F),str(U)))
    return U


def rotate_about_axis(point, axis, angle):
    """
    Rotate vector A about vector Axis by angle
    Using Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    :param point: [x,y,z] coordinate to rotate
    :param axis: [dx,dy,dz] vector about which to rotate
    :param angle: angle to rotate in deg
    :return: [x,y,z] rotated point
    """
    point = np.asarray(point, dtype=np.float)
    axis = np.asarray(axis, dtype=np.float) / np.sqrt(np.sum(np.square(axis)))
    rad = np.deg2rad(angle)
    cs = np.cos(rad)
    sn = np.sin(rad)
    return point * cs + np.cross(axis, point) * sn + axis * np.dot(axis, point) * (1 - cs)


def you_normal_vector(eta=0, chi=90, mu=0):
    """
    Determine the normal vector using the You diffractometer angles
      you_normal_vector(0, 0, 0) = [1, 0, 0]
      you_normal_vector(0, 90, 0) = [0, 1, 0]
      you_normal_vector(90, 90, 0) = [0, 0, -1]
      you_normal_vector(0, 0, 90) = [0, 0, -1]
    :param eta: angle (deg) along the x-axis
    :param mu: angle (deg) about the z-axis
    :param chi: angle deg) a
    :return: array
    """
    eta = np.deg2rad(eta)
    chi = np.deg2rad(chi)
    mu = np.deg2rad(mu)
    normal = np.array([np.sin(mu) * np.sin(eta) * np.sin(chi) + np.cos(mu) * np.cos(chi),
                       np.cos(eta) * np.sin(chi),
                       -np.cos(mu) * np.sin(eta) * np.sin(chi) - np.sin(mu) * np.cos(chi)])
    return normal


def group(A, tolerance=0.0001):
    """
    Group similear values in an array, returning the group and indexes
    array will be sorted so lowest numbers are grouped first
      group_index, group_values, group_counts = group([2.1, 3.0, 3.1, 1.00], 0.1)
    group_values = [1., 2., 3.] array of grouped numbers (rounded)
    array_index = [1, 2, 2, 0] array matching A values to groups, such that A ~ group_values[group_index]
    group_index = [3, 0, 1] array matching group values to A, such that group_values ~ A[group_index]
    group_counts = [1, 1, 2] number of iterations of each item in group_values
    :param A: list or numpy array of numbers
    :param tolerance: values within this number will be grouped
    :return: group_values, array_index, group_index, group_counts
    """
    A = np.asarray(A, dtype=np.float).reshape(-1)
    idx = np.argsort(A)
    rtn_idx = np.argsort(idx)
    A2 = np.round(A / tolerance) * tolerance
    groups, indices, inverse, counts = np.unique(A2[idx], return_index=True, return_inverse=True, return_counts=True)
    # groups = A[idx][indices]  # return original, not rounded values
    array_index = inverse[rtn_idx]
    group_index = idx[indices]
    return groups, array_index, group_index, counts


def unique_vector(vecarray, tol=0.05):
    """
    Find unique vectors in an array within a certain tolerance

    newarray,uniqueidx,matchidx = unique_vector(vecarray,tol=0.05)
       vecarray = np.array([[:]])
       newarray = vecarray with duplicate vectors removed
       uniqueidx= index vectors of newarray such that vecarray[uniqueidx]=newarray
       matchidx = index vectors of vecarray such that newarray[matchidx]=vecarray
       tol = tolerance of vector difference to define unique vectors

    E.G.
    vecarray = [[1,1,1],[2,2,2],[3,3,3],[1,1,1],[4,4,4],[2,2,2]]
    newarray,uniqueidx,matchidx = unique_vector(vecarray,tol=0.05)
      newarray:        uniqueidx:
     [[1, 1, 1],           [0,
      [2, 2, 2],            1,
      [3, 3, 3],            2,
      [4, 4, 4]]            4]

    matchidx: [0, 1, 2, 0, 3, 1]

    E.G.
    newarray,uniqueidx,matchidx = unique_vector([[1,0,0],[1,0,0],[0,1,0]],tol=0.05)
    >> newarray = [[1,0,0],[0,1,0]]
    >> uniqueidx= [0,2]
    >> matchidx = [0,0,1]
    """

    vecarray = np.asarray(vecarray).reshape([len(vecarray), -1])
    matchidx = -np.ones(len(vecarray), dtype=int)
    Nunique = 0
    for n in range(len(vecarray)):
        if matchidx[n] > -1: continue
        diff = mag(vecarray - vecarray[n])
        matchidx[diff < tol] = Nunique
        Nunique += 1
    matchidx = matchidx.tolist()
    uniqueidx = [matchidx.index(n) for n in range(np.max(matchidx) + 1)]
    newarray = vecarray[uniqueidx, :]
    return newarray, uniqueidx, matchidx


def distance2line(line_start, line_end, point):
    """
    Calculate distance from a line between the start and end to an arbitary point in space
    :param line_start: array, position of the start of the line
    :param line_end:  array, position of the end of the line
    :param point: array, arbitary position in space
    :return: float
    """
    line_start = np.asarray(line_start)
    line_end = np.asarray(line_end)
    point = np.asarray(point)

    line_diff = line_end - line_start
    unit_line = line_diff / np.sqrt(np.sum(line_diff ** 2))

    vec_arb = (line_start - point) - np.dot((line_start - point), unit_line) * unit_line
    return np.sqrt(np.sum(vec_arb ** 2))


def vector_intersection(point1, direction1, point2, direction2):
    """
    Calculate the point in 2D where two lines cross.
    If lines are parallel, return nan
    For derivation, see: http://paulbourke.net/geometry/pointlineplane/
    :param point1: [x,y] some coordinate on line 1
    :param direction1: [dx, dy] the direction of line 1
    :param point2: [x,y] some coordinate on line 2
    :param direction2: [dx, dy] the direction of line 2
    :return: [x, y]
    """

    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    direction1 = np.asarray(direction1) / np.sqrt(np.sum(np.square(direction1)))
    direction2 = np.asarray(direction2) / np.sqrt(np.sum(np.square(direction2)))

    mat = np.array([direction1, -direction2])
    try:
        inv = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        print('Vectors are parallel')
        return np.array([np.nan, np.nan])
    ua, ub = np.dot(point2 - point1, inv)
    intersect = point1 + ua*direction1
    return intersect


def vector_intersection3d(point1, direction1, point2, direction2):
    """
    Calculate the point in 3D where two lines cross.
    If lines are parallel, return nan
    For derivation, see: https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines/271366
    :param point1: [x,y,z] some coordinate on line 1
    :param direction1: [dx, dy, dz] the direction of line 1
    :param point2: [x,y,z] some coordinate on line 2
    :param direction2: [dx, dy, dz] the direction of line 2
    :return: [x, y, z]
    """

    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    direction1 = np.asarray(direction1)/np.sqrt(np.sum(np.square(direction1)))
    direction2 = np.asarray(direction2)/np.sqrt(np.sum(np.square(direction2)))

    line = point2 - point1
    c1 = np.cross(direction2, line)
    c2 = np.cross(direction2, direction1)
    h = mag(c1)
    k = mag(c2)
    a = ang(c1, c2)
    if h == 0 or k == 0:
        print('Lines parallel')
        return np.array([np.nan, np.nan, np.nan])
    v = (h/k) * direction1
    if np.abs(a) < np.pi/2:
        return point1 + v
    return point1 - v


def plane_intersection(line_point, line_direction, plane_point, plane_normal):
    """
    Calculate the point at which a line intersects a plane
    :param line_point: [x,y],z] some coordinate on line
    :param line_direction: [dx,dy],dz] the direction of line
    :param plane_point:  [x,y],z] some coordinate on the plane
    :param plane_normal: [dx,dy],dz] the normal vector of the plane
    :return: [x,y],z]
    """

    line_point = np.asarray(line_point)
    plane_point = np.asarray(plane_point)
    line_direction = np.asarray(line_direction) / np.sqrt(np.sum(np.square(line_direction)))
    plane_normal = np.asarray(plane_normal) / np.sqrt(np.sum(np.square(plane_normal)))

    u1 = np.dot(plane_normal, plane_point - line_point)
    u2 = np.dot(plane_normal, line_direction)

    if u2 == 0:
        print('Plane is parallel to line')
        return None
    u = u1 / u2
    intersect = line_point + u*line_direction
    return intersect


def find_index(A, value):
    """
    Return the index of the closest number in array A to value

    E.G.
    A = [1,2,3,4,5]
    find_index(A,3.5)
     >> 2
    A = [[1,2],[3,4]]
    find_index(A,3.2)
     >> (1,0)

     For multiple values, use:
     B = [find_index(A,x) for x in values]
    """
    A = np.asarray(A, dtype=np.float)
    idx = np.argmin(np.abs(A - value))
    if len(A.shape) < 2:
        return idx
    else:
        return np.unravel_index(idx, A.shape)


def find_vector(A, V, difference=0.01):
    """
    Return the index(s) of vectors in array A within difference of vector V
    Comparison is based on vector difference A[n,:]-V

    Returns None if no matching vector is present.

    E.G.
    A = [[1,2,3],
         [4,5,6],
         [7,8,9]]
    find_index(A,[4,5,6])
     >> 1

    A = [[0, 2.5, 0],
         [1, 0.1, 0],
         [1,-0.1, 0],
         [0, 0.5, ]]
    find_vector(A, [1,0,0], difference=0.1)
     >> [1, 2]
    """
    A = np.asarray(A).reshape((-1, np.shape(A)[-1]))
    V = np.asarray(V).reshape(-1)
    M = mag(A - V)
    idx = np.where(M < difference)[0]
    if len(idx) == 1:
        return idx[0]
    elif len(idx) == 0:
        return None
    else:
        return idx


def search_dict_lists(d, **kwargs):
    """
    Search equal length lists in a dictionary for specific values
    e.g.
        idx = search_dict_lists(cif, _geom_bond_atom_site_label_1='Ru1',
                                     _geom_bond_atom_site_label_2='O1',
                                     _geom_bond_site_symmetry_2='.')
    :param d: dict
    :param kwargs: keyword=item
    :return: boolean array index
    """
    search = [np.array(d[key]) == value for key, value in kwargs.items()]
    bool = np.product(search, axis=0)
    return np.argwhere(bool).reshape(-1)


def detail(A):
    """
    Prints details about the given vector,
    including length, min, max, number of elements
    """
    tp = type(A)
    print('Type: {}'.format(tp))
    A = np.asarray(A, dtype=np.float)
    size = ' x '.join([str(x) for x in A.shape])
    mem = '%1.3g MB' % (A.nbytes * 1e-6)
    print('Size: {} (elements={}, {})'.format(size, A.size, mem))
    print('Min: {}, Max: {}, Mean: {}, NaNs: {}'.format(np.nanmin(A), np.nanmax(A), np.nanmean(A), np.sum(np.isnan(A))))


def inline_help(func):
    """Return function spec and first line of help in line"""
    fun_name = '%s%s' % (func.__name__, inspect.signature(func))
    fun_doc = func.__doc__.strip().split('\n')[0] if func.__doc__ else ""
    return "%s\n\t%s" % (fun_name, fun_doc)


def array_str(A):
    """
    Returns a short string with array information
    :param A: np array
    :return: str
    """
    shape = np.shape(A)
    try:
        amax = np.max(A)
        amin = np.min(A)
        amean = np.mean(A)
        out_str = "%s max: %4.5g, min: %4.5g, mean: %4.5g"
        return out_str % (shape, amax, amin, amean)
    except TypeError:
        # list of str
        array = np.asarray(A).reshape(-1)
        array_start = array[0]
        array_end = array[-1]
        out_str = "%s [%s, ..., %s]"
        return out_str % (shape, array_start, array_end)


def print_arrays(arrays=[]):
    """
    Prints values from several arrays with nice formatting
    e.g. dgp.print_arrays(1e6*np.random.rand(5,5))
         0   6.04e+03   2.29e+05   7.19e+05   4.74e+05   9.59e+05
         1   9.16e+05   6.33e+05    1.6e+05   3.75e+05   3.01e+05
         2   4.21e+04   1.78e+05   4.78e+05   7.56e+05   2.47e+05
         3   9.26e+05   6.39e+05   3.96e+05   4.42e+05   4.52e+04
         4   3.38e+05   4.54e+05   4.29e+05   6.34e+04   6.06e+04
    """

    # Combine all arrays into BigArray
    BigArray = np.arange(len(arrays[0])).reshape([len(arrays[0]), -1])
    for A in arrays:
        BigArray = np.hstack((BigArray, np.asarray(A).reshape([len(A), -1])))

    # Print BigArray
    fmt = '{:10.3g} ' * BigArray.shape[1]
    for n in range(len(BigArray)):
        print(fmt.format(*BigArray[n, :]))


def cell(lengths=[1, 1, 1], rotation=[0, 0, 0]):
    """
    Returns a unit CELL with vectors defined by length and rotation (Deg)
    :param lengths:
    :param rotation:
    :return:
    """
    CELL = np.eye(3) * lengths
    CELL = rot3D(CELL, *rotation)
    return CELL


def index_coordinates(A, CELL):
    """
    Index A(x1,y1,1) on new coordinate system B(x2,y2,z2), defined by unit vectors CELL = [A,B,C]
      A = [[nx3]] array of vectors
      CELL = [[3x3]] matrix of vectors [a,b,c]
    """

    A = np.asarray(A, dtype=np.float).reshape((-1, 3))
    B = np.dot(A, np.linalg.inv(CELL))
    return B


def isincell(A, cell_centre=[0, 0, 0], CELL=cell()):
    """
     Return boolean of whether vector xyx is inside a cell defined by position, size and rotation
       A = [x,y,z] or [[x1,y1,z1]] array of positions
       cell_centre = [x,y,z] cell centre
       CELL = [[3x3]] matrix of vectors [a,b,c]

    E.G.
       xyz = [[1,1,0],[0,0,0.5],[0.1,0,5]]
       centre = [0,0,0]
       CELL = cell([1,1,10],[0,0,0])
       isinbox(xyz,centre,CELL)
       >>> [False,True,True]
    """

    A = np.asarray(A, dtype=np.float).reshape((-1, 3))
    A = A - cell_centre
    idx = index_coordinates(A, CELL)
    return np.all(np.abs(idx) <= 0.5, axis=1)


def sphere_array(A, max_angle1=90, max_angle2=90, step1=1, step2=1):
    """
    Rotate points in array A by multiple angles
     B = sphere_array(A, max_angle1, max_angle2, step1, step2)

      A = [nx3] array of 3D coordinates
      max_angle1 = max rotation angle
      max_angle1 = max rotation angle
      step1 = angular step size
      step2 = angular step size

    Each coordinate in A will be rotated by angles:
        angles = np.arange(0, max_angle+step, step)
    The output B will have size:
        [len(A) * len(angles1) * len(angles2), 3]
    """

    A = np.asarray(A, dtype=np.float).reshape((-1, 3))

    angles1 = np.arange(0, max_angle1 + step1, step1)
    angles2 = np.arange(0, max_angle2 + step2, step2)
    len1 = len(angles1)
    len2 = len(angles2)
    len3 = len(A)
    tot_size = len1 * len2 * len3
    OUT = np.zeros([tot_size, 3])
    for n in range(len(angles1)):
        for m in range(len(angles2)):
            B = rot3D(A, 0, angles1[n], angles2[m])
            st = n * len2 + m * len3
            nd = n * len2 + (m + 1) * len3
            OUT[st:nd, :] = B
    return OUT


'---------------------------String manipulation-------------------------'


def stfm(val, err):
    """
    Create standard form string from value and uncertainty"
     str = stfm(val,err)
     Examples:
          '35.25 (1)' = stfm(35.25,0.01)
          '110 (5)' = stfm(110.25,5)
          '0.0015300 (5)' = stfm(0.00153,0.0000005)
          '1.56(2)E+6' = stfm(1.5632e6,1.53e4)

    Notes:
     - Errors less than 0.01% of values will be given as 0
     - The maximum length of string is 13 characters
     - Errors greater then 10x the value will cause the value to be rounded to zero
    """

    # Determine the number of significant figures from the error
    if err == 0. or val / float(err) >= 1E5:
        # Zero error - give value to 4 sig. fig.
        out = '{:1.5G}'.format(val)
        if 'E' in out:
            out = '{}(0)E{}'.format(*out.split('E'))
        else:
            out = out + ' (0)'
        return out
    elif np.log10(np.abs(err)) > 0.:
        # Error > 0
        sigfig = np.ceil(np.log10(np.abs(err))) - 1
        dec = 0.
    elif np.isnan(err):
        # nan error
        return '{} (-)'.format(val)
    else:
        # error < 0
        sigfig = np.floor(np.log10(np.abs(err)) + 0.025)
        dec = -sigfig

    # Round value and error to the number of significant figures
    rval = round(val / (10. ** sigfig)) * (10. ** sigfig)
    rerr = round(err / (10. ** sigfig)) * (10. ** sigfig)
    # size of value and error
    pw = np.floor(np.log10(np.abs(rval)))
    pwr = np.floor(np.log10(np.abs(rerr)))

    max_pw = max(pw, pwr)
    ln = max_pw - sigfig  # power difference

    if np.log10(np.abs(err)) < 0:
        rerr = err / (10. ** sigfig)

    # Small numbers - exponential notation
    if max_pw < -3.:
        rval = rval / (10. ** max_pw)
        fmt = '{' + '0:1.{:1.0f}f'.format(ln) + '}({1:1.0f})E{2:1.0f}'
        return fmt.format(rval, rerr, max_pw)

    # Large numbers - exponential notation
    if max_pw >= 4.:
        rval = rval / (10. ** max_pw)
        rerr = rerr / (10. ** sigfig)
        fmt = '{' + '0:1.{:1.0f}f'.format(ln) + '}({1:1.0f})E+{2:1.0f}'
        return fmt.format(rval, rerr, max_pw)

    fmt = '{' + '0:0.{:1.0f}f'.format(dec + 0) + '} ({1:1.0f})'
    return fmt.format(rval, rerr)


def readstfm(string):
    """
    Read numbers written in standard form: 0.01(2), return value and error
    Read numbers from string with form 0.01(2), returns floats 0.01 and 0.02
    Errors and values will return 0 if not given.

    E.G.
    readstfm('0.01(2)') = (0.01, 0.02)
    readstfm('1000(300)') = (1000.,300.)
    readstfm('1.23(3)E4') = (12300.0, 300.0)
    """

    values = re.findall('[-0-9.]+|\([-0-9.]+\)', string)
    if len(values) > 0 and '(' not in values[0] and values[0] != '.':
        value = values[0]
    else:
        value = '0'

    # Determine number of decimal places for error
    idx = value.find('.')  # returns -1 if . not found
    if idx > -1:
        pp = idx - len(value) + 1
    else:
        pp = 0
    value = float(value)

    error = re.findall('\([-0-9.]+\)', string)
    if len(error) > 0:
        error = abs(float(error[0].strip('()')))
        error = error * 10 ** pp
    else:
        error = 0.

    power = re.findall('(?:[eE]|x10\^|\*10\^|\*10\*\*)([+-]?\d*\.?\d+)', string)
    if len(power) > 0:
        power = float(power[0])
        value = value * 10 ** power
        error = error * 10 ** power
    return value, error


def saveable(string):
    """
    Returns a string without special charaters.
    Removes bad characters from a string, so it can be used as a filename
    E.G.
    saveable('Test#1<froot@5!') = 'TestS1_froot_5'
    """
    # Special - replace # with S for scans
    string = string.replace('#', 'S')
    # Replace some characters with underscores
    for char in '#%{}\/<>@|':
        string = string.replace(char, '_')
        # Replace other characters with nothing
    for char in '*$&^?!':
        string = string.replace(char, '')
    # Remove non-ascii characters
    string = ''.join(i for i in string if ord(i) < 128)
    # string = string.decode('unicode_escape').encode('ascii','ignore')
    return string


def findranges(scannos, sep=':'):
    """
    Convert a list of numbers to a simple string
    E.G.
    findranges([1,2,3,4,5]) = '1:5'
    findranges([1,2,3,4,5,10,12,14,16]) = '1:5,10:2:16'
    """

    scannos = np.sort(scannos).astype(int)

    dif = np.diff(scannos)

    stt, stp, rng = [scannos[0]], [dif[0]], [1]
    for n in range(1, len(dif)):
        if scannos[n + 1] != scannos[n] + dif[n - 1]:
            stt += [scannos[n]]
            stp += [dif[n]]
            rng += [1]
        else:
            rng[-1] += 1
    stt += [scannos[-1]]
    rng += [1]

    out = []
    x = 0
    while x < len(stt):
        if rng[x] == 1:
            out += ['{}'.format(stt[x])]
            x += 1
        elif stp[x] == 1:
            out += ['{}{}{}'.format(stt[x], sep, stt[x + 1])]
            x += 2
        else:
            out += ['{}{}{}{}{}'.format(stt[x], sep, stp[x], sep, stt[x + 1])]
            x += 2
    return ','.join(out)


def numbers2string(scannos, sep=':'):
    """
    Convert a list of numbers to a simple string
    E.G.
    numbers2string([50001,50002,50003]) = '5000[1:3]'
    numbers2string([51020,51030,51040]) = '510[20:10:40]'
    """

    if type(scannos) is str or type(scannos) is int or len(scannos) == 1:
        return str(scannos)

    scannos = np.sort(scannos).astype(str)

    n = len(scannos[0])
    while np.all([scannos[0][:-n] == x[:-n] for x in scannos]):
        n -= 1

    if n == len(scannos[0]):
        return '{}-{}'.format(scannos[0], scannos[-1])

    inistr = scannos[0][:-(n + 1)]
    strc = [i[-(n + 1):] for i in scannos]
    liststr = findranges(strc, sep=sep)
    return '{}[{}]'.format(inistr, liststr)


def complex2str(val, fmt='6.1f'):
    """
    Convert complex number to string
    """

    rl = np.real(val)
    im = np.imag(val)
    fmt1 = '%' + fmt
    fmt2 = '%-' + fmt
    if im >= 0:
        return (fmt1 + ' + i' + fmt2) % (rl, im)
    elif im < 0:
        return (fmt1 + ' - i' + fmt2) % (rl, np.abs(im))


def multi_replace(string, old=[], new=[]):
    """
    Replace multiple strings at once
    :param string:
    :param old:
    :param new:
    :return:
    """

    if type(new) is str:
        new = [new]
    if type(old) is str:
        old = [old]
    if len(new) == 1:
        new = new * len(old)
    for i, j in zip(old, new):
        string = string.replace(i, j)
    return string


def replace_bracket_multiple(name):
    """
    Replace any numbers in brackets with numbers multipled by bracket multiplyer
    Assumes bracket multiplier is on the left side
    e.g.
        replace_bracket_multiple('Mn0.3(Fe3.6(Co1.2)2)4(Mo0.7Pr44)3')
        >> 'Mn0.3Fe14.4Co9.6Mo2.1Pr132'
    :param name: str
    :return: str
    """
    """
    To do:
     - Multiply by fraction (regex for '/number')
     - Multiple on right hand side
    """
    # Regex:
    regex_num = re.compile('[\d\.]+')
    regex_bracket_n = re.compile('\)[\d\.]+')

    # Find outside brackets
    bracket = []
    start_idx = []
    level = 0
    for n, s in enumerate(name):
        if s in ['(', '[', '{']:
            start_idx += [n]
            level += 1
        elif s in [')', ']', '}']:
            level -= 1
            if level == 0:
                num = regex_bracket_n.findall(name[n:])
                if len(num) > 0:
                    bracket_end = n + len(num[0])
                    num = float(num[0][1:])
                else:
                    bracket_end = n + 1
                    num = 1.0
                bracket += [(
                    name[start_idx[0] + 1:n],  # insde brackets
                    name[start_idx[0]:bracket_end],  # str to replace
                    num  # multiplication appending bracket
                )]
                start_idx = []

    for numstr, repstr, num in bracket:
        # Run recursivley to get inner brackets
        numstr = replace_bracket_multiple(numstr)
        # Replace each number by it's multiple
        for oldnum in regex_num.findall(numstr):
            numstr = numstr.replace(oldnum, '%0.3g' % (float(oldnum) * num))
        # Replace in original string
        name = name.replace(repstr, numstr)
    return name


def shortstr(string):
    """
    Shorten string by removing long floats
    :param string: string, e.g. '#810002 scan eta 74.89533603616637 76.49533603616636 0.02 pil3_100k 1 roi2'
    :return: shorter string, e.g. '#810002 scan eta 74.895 76.495 0.02 pil3_100k 1 roi2'
    """
    #return re.sub(r'(\d\d\d)\d{4,}', r'\1', string)
    def subfun(m):
        return str(round(float(m.group()), 3))
    return re.sub(r'\d+\.\d{5,}', subfun, string)


def nice_print(precision=4, linewidth=300):
    """
    Sets default printing of arrays to a nicer format
    """
    np.set_printoptions(precision=precision, suppress=True, linewidth=linewidth)


'----------------------------------Others-------------------------------'


def gauss(x, y=None, height=1, cen=0, fwhm=0.5, bkg=0):
    """
    Define Gaussian distribution in 1 or 2 dimensions
    From http://fityk.nieto.pl/model.html
        x = [1xn] array of values, defines size of gaussian in dimension 1
        y = None* or [1xm] array of values, defines size of gaussian in dimension 2
        height = peak height
        cen = peak centre
        fwhm = peak full width at half-max
        bkg = background
    """

    if y is None:
        y = cen

    x = np.asarray(x, dtype=np.float).reshape([-1])
    y = np.asarray(y, dtype=np.float).reshape([-1])
    X, Y = np.meshgrid(x, y)
    g = height * np.exp(-np.log(2) * (((X - cen) ** 2 + (Y - cen) ** 2) / (fwhm / 2) ** 2)) + bkg

    if len(y) == 1:
        g = g.reshape([-1])
    return g


def frange(start, stop=None, step=1):
    """
    Returns a list of floats from start to stop in step increments
    Like np.arange but ends at stop, rather than at stop-step
    E.G.
    A = frange(0,5,1) = [0.,1.,2.,3.,4.,5.]
    """
    if stop is None:
        stop = start
        start = 0

    return list(np.arange(start, stop + 0.00001, step, dtype=np.float))


def squaredata(xdata, ydata, data, repeat=None):
    """
    Generate square arrays from 1D data, automatically determinging the repeat value
    :param xdata: [n] array
    :param ydata: [n] array
    :param data: [n] array
    :param repeat: int m or None to deteremine m from differences in xdata and y data
    :return: X, Y, D [n//m, m] arrays
    """

    if repeat is None:
        # Determine the repeat length of the scans
        delta_x = np.abs(np.diff(xdata))
        ch_idx_x = np.where(delta_x > delta_x.max() * 0.9)  # find biggest changes
        ch_delta_x = np.diff(ch_idx_x)
        rep_len_x = np.round(np.mean(ch_delta_x))
        delta_y = np.abs(np.diff(ydata))
        ch_idx_y = np.where(delta_y > delta_y.max() * 0.9)  # find biggest changes
        ch_delta_y = np.diff(ch_idx_y)
        rep_len_y = np.round(np.mean(ch_delta_y))
        repeat = int(max(rep_len_x, rep_len_y))
    xsquare = xdata[:repeat * (len(xdata) // repeat)].reshape(-1, repeat)
    ysquare = ydata[:repeat * (len(ydata) // repeat)].reshape(-1, repeat)
    dsquare = data[:repeat * (len(data) // repeat)].reshape(-1, repeat)
    return xsquare, ysquare, dsquare


def grid_intensity(points, values, resolution=0.01, peak_width=0.1, background=0):
    """
    Generates array of intensities along a spaced grid, equivalent to a powder pattern.
      grid, values = generate_powder(points, values, resolution=0.01, peak_width=0.1, background=0)
    :param points: [nx1] position of values to place on grid
    :param values: [nx1] values to place at points
    :param resolution: grid spacing size, with same units as points
    :param peak_width: width of convolved gaussian, with same units as points
    :param background: add a normal (random) background with width sqrt(background)
    :return: points, values
    """

    points = np.asarray(points, dtype=np.float)
    values = np.asarray(values, dtype=np.float)

    # create plotting mesh
    grid_points = np.arange(np.min(points) - 50 * resolution, np.max(points) + 50 * resolution, resolution)
    pixels = len(grid_points)
    grid_values = np.zeros([pixels])

    # add reflections to background
    pixel_size = (grid_points.max() - grid_points.min()) / pixels
    peak_width_pixels = peak_width / (1.0 * pixel_size)

    pixel_coord = (points - grid_points.min()) / (grid_points - grid_points.min()).max()
    pixel_coord = (pixel_coord * (pixels - 1)).astype(int)
    pixel_coord = pixel_coord.astype(int)

    for n in range(0, len(values)):
        grid_values[pixel_coord[n]] = grid_values[pixel_coord[n]] + values[n]

    # Convolve with a gaussian (if >0 or not None)
    if peak_width:
        gauss_x = np.arange(-3 * peak_width_pixels, 3 * peak_width_pixels + 1)  # gaussian width = 2*FWHM
        g = gauss(gauss_x, None, height=1, cen=0, fwhm=peak_width_pixels, bkg=0)
        grid_values = np.convolve(grid_values, g, mode='same')

    # Add background (if >0 or not None)
    if background:
        bkg = np.random.normal(background, np.sqrt(background), [pixels])
        grid_values = grid_values + bkg
    return grid_points, grid_values


def map2grid(grid, points, values, widths=None, background=0):
    """
    Generates array of intensities along a spaced grid, equivalent to a powder pattern.
      grid, values = generate_powder(points, values, resolution=0.01, peak_width=0.1, background=0)
    :param grid: [mx1] grid of positions
    :param points: [nx1] position of values to place on grid
    :param values: [nx1] values to place at points
    :param widths: width of convolved gaussian, with same units as points
    :param background: add a normal (random) background with width sqrt(background)
    :return: points, values
    """
    grid = np.asarray(grid, dtype=np.float)
    points = np.asarray(points, dtype=np.float)
    values = np.asarray(values, dtype=np.float)
    widths = np.asarray(widths, dtype=np.float)

    if widths.size == 1:
        widths = widths * np.ones(len(points))

    pixels = len(grid)
    grid_values = np.zeros([pixels])

    for point, value, width in zip(points, values, widths):
        g = gauss(grid, None, height=value, cen=point, fwhm=width, bkg=0)
        grid_values += g

    # Add background (if >0 or not None)
    if background:
        bkg = np.random.normal(background, np.sqrt(background), [pixels])
        grid_values = grid_values + bkg
    return grid_values


def whererun():
    """Returns the location where python was run"""
    return os.path.abspath('.')


def lastlines(filename, lines=1, max_line_length=255):
    """Returns the last n lines of a text file"""
    with open(filename, 'rb') as f:
        f.seek(-(lines+1)*max_line_length, os.SEEK_END)
        endlines = f.read().decode().split('\n')
    return endlines[-lines:]


def get_methods(object, include_special=True):
    """Returns a list of methods (functions) within object"""
    if include_special:
        return [method_name for method_name in dir(object) if callable(getattr(object, method_name))]
    return [method_name for method_name in dir(object) if callable(getattr(object, method_name)) and '__' not in method_name]


def list_methods(object, include_special=False):
    """Return list of methods (functions) in class object"""
    methods = get_methods(object, include_special)
    return '\n'.join([inline_help(getattr(object, fun)) for fun in methods])
