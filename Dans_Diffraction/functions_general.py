# -*- coding: utf-8 -*-
"""
Module: Generally useful functions "functions_general.py"

Contains various useful shortcuts for manipulating strings and arrays, 
making use of numpy and re.

By Dan Porter, PhD
Diamond
2018

Usage: 
    - Run this file in an interactive console
    OR
    - from Dans_Diffraction import functions_general as fg
    

Version 1.0
Last updated: 06/01/18

Version History:
06/01/18 1.0    Program created from DansGeneralProgs.py V2.3

@author: DGPorter
"""

import sys,os,re
import numpy as np

# File directory
directory = os.path.abspath(os.path.dirname(__file__))

# Constants
pi = np.pi          # mmmm tasty Pi
e = 1.6021733E-19   # C  electron charge
h = 6.62606868E-34  # Js  Plank consant
c = 299792458       # m/s   Speed of light
u0 = 4*pi*1e-7      # H m-1 Magnetic permeability of free space
me = 9.109e-31      # kg Electron rest mass
Na = 6.022e23       # Avagadro's No
A = 1e-10;          # m Angstrom
Cu = 8.048          # Cu-Ka emission energy, keV
Mo = 17.4808        # Mo-Ka emission energy, keV

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
    A = np.asarray(A,dtype=np.float)
    return np.sqrt(np.sum(A**2,axis=len(A.shape)-1))

def norm(A):
    """
    Returns normalised vector A
    If A has 2 dimensions, returns an array of normalised vectors
    The returned array will be the same shape as the given array.
    
    E.G.
     norm([1,1,1]) = [1,1,1]/1.732 = [ 0.57735,  0.57735,  0.57735]
     norm(array([[1,1,1],[2,2,2]]) = [ [ 0.57735,  0.57735,  0.57735] , [ 0.57735,  0.57735,  0.57735] ]
     """
     
    A = np.asarray(A,dtype=np.float).reshape((-1,np.shape(A)[-1]))
    mag = np.sqrt(np.sum(A**2,axis=A.ndim-1)).reshape((-1,1))
    mag[mag==0] = 1 # stop warning errors
    N = A/mag
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
    
    A = np.asarray(A,dtype=np.float).reshape((-1,3))
    if A.size == 1:
        return (np.sum(A,axis=1)>0)[0]*2 - 1
    else:
        return (np.sum(A,axis=1)>0)*2 - 1

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
    
    A = np.asarray(A,dtype=np.float).reshape((-1,3))
    mag = np.sqrt(np.sum(A**2,axis=len(A.shape)-1)).reshape(-1)
    quad = (np.sum(A,len(A.shape)-1)>0)*2 - 1
    if len(A) == 1:
        return quad[0]*mag[0]
    else:
        return quad*mag

def ang(a,b,deg=False):
    """
    Returns the angle, in Radians between vectors a and b
    E.G.
    ang([1,0,0],[0,1,0]) >> 1.571
    ang([1,0,0],[0,1,0],'deg') >> 90
    """
    a = np.asarray(a,dtype=float).reshape([-1])
    b = np.asarray(b,dtype=float).reshape([-1]) 
    
    cosang = np.dot(a, b)
    sinang = quadmag(np.cross(a, b))
    angle = np.arctan2(sinang, cosang)
    if deg:
        return np.rad2deg(angle)
    return angle

def rot3D(A,alpha=0.,beta=0.,gamma=0.):
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
    
    A = np.asarray(A,dtype=np.float).reshape((-1,3))
    
    # Convert to radians
    alpha = alpha*np.pi/180.
    beta  = beta*np.pi/180.
    gamma = gamma*np.pi/180.
    
    # Define 3D rotation matrix
    Rx = np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0.,np.sin(gamma),np.cos(gamma)]])
    Ry = np.array([[np.cos(beta),0.,np.sin(beta)],[0.,1.,0.],[-np.sin(beta), 0., np.cos(beta)]])
    Rz = np.array([[np.cos(alpha),-np.sin(alpha),0.],[np.sin(alpha),np.cos(alpha),0.],[0.,0.,1.]])
    R = np.dot(np.dot(Rx,Ry),Rz)
    
    # Rotate coordinates
    return np.dot(R,A.T).T

def rotmat(a,b):
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
    
    d = np.dot(a,b)
    c = mag(np.cross(a,b))
    G = np.array([[d,-c,0],[c,d,0],[0,0,1]])
    
    u = a
    v = norm(b-np.dot(a,b)*a)
    w = np.cross(b,a)
    
    Fm1 = np.array([u,v,w]).T
    F = np.linalg.inv( Fm1 )
    U = np.dot( np.dot( Fm1,G),F)
    
    #print('a = {}\nb = {}'.format(str(a),str(b)))
    #print('G = {}\nFm1 = {}\nF = {}\nU = {}'.format(str(G),str(Fm1),str(F),str(U)))
    return U

def unique_vector(vecarray,tol=0.05):
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
    
    vecarray = np.asarray(vecarray).reshape([len(vecarray),-1])
    matchidx = -np.ones(len(vecarray),dtype=int)
    Nunique = 0
    for n in range(len(vecarray)):
        if matchidx[n] > -1: continue
        diff = mag(vecarray-vecarray[n])
        matchidx[diff<tol] = Nunique
        Nunique += 1
    matchidx = matchidx.tolist()
    uniqueidx = [matchidx.index(n) for n in range(np.max(matchidx)+1)]
    newarray = vecarray[uniqueidx,:]
    return newarray,uniqueidx,matchidx

def find_index(A,value):
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
    A = np.asarray(A,dtype=np.float)
    idx = np.argmin(np.abs( A-value ))
    if len(A.shape) < 2:
        return idx
    else:
        return np.unravel_index(idx,A.shape)

def detail(A):
    """
    Prints details about the given vector, 
    including length, min, max, number of elements
    """
    tp = type(A)
    print('Type: {}'.format(tp))
    A = np.asarray(A,dtype=np.float)
    size = ' x '.join([str(x) for x in A.shape])
    mem = '%1.3g MB' % (A.nbytes*1e-6)
    print('Size: {} (elements={}, {})'.format(size,A.size,mem))
    print('Min: {}, Max: {}, Mean: {}, NaNs: {}'.format(np.nanmin(A),np.nanmax(A),np.nanmean(A),np.sum(np.isnan(A))))

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
    BigArray = np.arange(len(arrays[0])).reshape([len(arrays[0]),-1])
    for A in arrays:
        BigArray = np.hstack((BigArray,np.asarray(A).reshape([len(A),-1])))
    
    # Print BigArray
    fmt = '{:10.3g} '*BigArray.shape[1]
    for n in range(len(BigArray)):
        print(fmt.format(*BigArray[n,:]))

def cell(lengths=[1,1,1],rotation=[0,0,0]):
    " Returns a unit CELL with vectors defined by length and rotation (Deg)"
    CELL = np.eye(3)*lengths
    CELL = rot3D(CELL,*rotation)
    return CELL

def index_coordinates(A,CELL):
    """
    Index A(x1,y1,1) on new coordinate system B(x2,y2,z2), defined by unit vectors CELL = [A,B,C]
      A = [[nx3]] array of vectors
      CELL = [[3x3]] matrix of vectors [a,b,c]
    """
    
    A = np.asarray(A,dtype=np.float).reshape((-1,3))
    B = np.dot(A,np.linalg.inv(CELL))
    return B

def isincell(A,cell_centre=[0,0,0],CELL=cell()):
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
    
    A = np.asarray(A,dtype=np.float).reshape((-1,3))
    A = A - cell_centre
    idx = index_coordinates(A,CELL)
    return np.all(np.abs(idx)<=0.5,axis=1)

'---------------------------String manipulation-------------------------'

def stfm(val,err):
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
    if err == 0. or val/float(err) >= 1E5:
        # Zero error - give value to 4 sig. fig.
        out = '{:1.5G}'.format(val)
        if 'E' in out:
            out = '{}(0)E{}'.format(*out.split('E'))
        else:
            out = out + ' (0)'
        return out
    elif np.log10(np.abs(err)) > 0.:
        # Error > 0
        sigfig = np.ceil(np.log10(np.abs(err)))-1
        dec = 0.
    elif np.isnan(err):
        # nan error
        return '{} (-)'.format(val)
    else:
        # error < 0
        sigfig = np.floor(np.log10(np.abs(err))+0.025)
        dec = -sigfig
    
    # Round value and error to the number of significant figures
    rval = round(val/(10.**sigfig))*(10.**sigfig)
    rerr = round(err/(10.**sigfig))*(10.**sigfig)
    # size of value and error
    pw = np.floor(np.log10(np.abs(rval)))
    pwr = np.floor(np.log10(np.abs(rerr)))
    
    max_pw = max(pw,pwr)
    ln = max_pw - sigfig # power difference
    
    if np.log10(np.abs(err)) < 0:
        rerr = err/(10.**sigfig)
    
    # Small numbers - exponential notation
    if max_pw < -3.:
        rval = rval/(10.**max_pw)
        fmt = '{'+'0:1.{:1.0f}f'.format(ln)+'}({1:1.0f})E{2:1.0f}'
        return fmt.format(rval,rerr,max_pw)
    
    # Large numbers - exponential notation
    if max_pw >= 4.:
        rval = rval/(10.**max_pw)
        rerr = rerr/(10.**sigfig)
        fmt = '{'+'0:1.{:1.0f}f'.format(ln)+'}({1:1.0f})E+{2:1.0f}'
        return fmt.format(rval,rerr,max_pw)
    
    fmt = '{'+'0:0.{:1.0f}f'.format(dec+0)+'} ({1:1.0f})'
    return fmt.format(rval,rerr)

def readstfm(string):
    """
    Read numbers written in standard form: 0.01(2), return value and error
    Read numbers from string with form 0.01(2), returns floats 0.01 and 0.02
    
    E.G.
    readstfm('0.01(2)') = (0.01, 0.02)
    readstfm('1000(300)') = (1000.,300.)
    """
    
    values = re.findall('[-0-9.]+',string)
    
    if values[0] == '.': 
        values[0] = '0'
    value = float(values[0])
    error = 0.
    
    if len(values) > 1:
        error = float(values[1])
        
        # Determine decimal place
        idx = values[0].find('.') # returns -1 if no decimal
        if idx > -1:
            pp = idx - len(values[0]) + 1
            error = error*10**pp
    return value,error

def saveable(string):
    """
    Returns a string without special charaters.
    Removes bad characters from a string, so it can be used as a filename
    E.G.
    saveable('Test#1<froot@5!') = 'TestS1_froot_5'
    """
    # Special - replace # with S for scans
    string = string.replace('#','S')
    # Replace some characters with underscores
    for char in '#%{}\/<>@|':
        string = string.replace(char,'_') 
    # Replace other characters with nothing
    for char in '*$�!':
        string = string.replace(char,'') 
    return string

def findranges(scannos,sep=':'):
    """
    Convert a list of numbers to a simple string
    E.G.
    findranges([1,2,3,4,5]) = '1:5'
    findranges([1,2,3,4,5,10,12,14,16]) = '1:5,10:2:16'
    """
    
    scannos = np.sort(scannos).astype(int)
    
    dif = np.diff(scannos)
    
    stt,stp = [scannos[0]],[dif[0]]
    for n in range(1,len(dif)):
        if scannos[n+1] != scannos[n]+dif[n-1]:
            stt += [scannos[n]]
            stp += [dif[n]]
    stt += [scannos[-1]]
    
    out = []
    for x in range(0,len(stt),2):
        if stp[x] == 1:
            out += ['{}{}{}'.format(stt[x],sep,stt[x+1])]
        else:
            out += ['{}{}{}{}{}'.format(stt[x],sep,stp[x],sep,stt[x+1])]
    return ','.join(out)

def numbers2string(scannos,sep=':'):
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
        return '{}-{}'.format(scannos[0],scannos[-1])
    
    inistr = scannos[0][:-(n+1)]
    strc = [i[-(n+1):] for i in scannos]
    liststr = findranges(strc,sep=sep)
    return '{}[{}]'.format(inistr,liststr)

def multi_replace(string,old=[],new=[]):
    "Replace multiple strings at once"
    
    if type(new) is str:
        new = [new]
    if type(old) is str:
        old = [old]
    if len(new) == 1:
        new = new*len(old)
    for i,j in zip(old,new):
        string = string.replace(i,j)
    return string

def nice_print(precision=4,linewidth=300):
    """
    Sets default printing of arrays to a nicer format
    """
    np.set_printoptions(precision=precision,suppress=True,linewidth=linewidth)

'----------------------------------Others-------------------------------'

def gauss(x,y=0,height=1,cen=0,FWHM=0.5,bkg=0):
    """
    Define Gaussian distribution in 1 or 2 dimensions
    From http://fityk.nieto.pl/model.html
        x = [1xn] array of values, defines size of gaussian in dimension 1
        y = 0 or [1xm] array of values, defines size of gaussian in dimension 2
        height = peak height
        cen = peak centre
        FWHM = peak full width at half-max
        bkg = background
    """
    x = np.asarray(x,dtype=np.float).reshape([-1])
    y = np.asarray(y,dtype=np.float).reshape([-1])
    X,Y = np.meshgrid(x,y)
    gauss = height*np.exp(-np.log(2)*( ((X-cen)**2 + (Y-cen)**2) /(FWHM/2)**2 )) + bkg
    
    if len(y) == 1:
        gauss = gauss.reshape([-1])
    return gauss

def frange(start,stop=None,step=1):
    """
    Returns a list of floats from start to stop in step increments
    Like np.arange but ends at stop, rather than at stop-step
    E.G.
    A = frange(0,5,1) = [0.,1.,2.,3.,4.,5.]
    """
    if stop is None:
        stop = start
        start = 0
    
    return list(np.arange(start,stop+0.00001,step,dtype=np.float))
