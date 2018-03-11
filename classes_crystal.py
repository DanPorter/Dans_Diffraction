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
        xtl.Plotting   >> Plot Crystal structures and simulate diffraction patterns
        xtl.Scattering >> Simulate diffraction intensities of different types

By Dan Porter, PhD
Diamond
2017

Version 2.0
Last updated: 13/02/18

Version History:
27/07/17 1.0    Version History started.
30/10/17 1.1    Many minor updates.
06/01/18 2.0    Name change and other updates
13/02/18 2.1    Move scattering commands to xtl.Scatter

@author: DGPorter
"""

import numpy as np

# Internal functions
from Dans_Diffraction import functions_general as fg
from Dans_Diffraction import functions_crystallography as fc
from Dans_Diffraction.classes_properties import Properties
from Dans_Diffraction.classes_scattering import Scattering
from Dans_Diffraction.classes_plotting import Plotting, Multi_Plotting, Plotting_Superstructure


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
    
    def __init__(self,filename=None):
        "Initialise the Crystal class, read from cif file if given."
        
        # Instatiate crystal attributes
        self.Cell = Cell()
        self.Symmetry = Symmetry()
        self.Atoms= Atoms()
        
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
        self.generate_structure() # Creates self.Structure, an instance of Atoms
        
        # Add exta functions
        self.Plot = Plotting(self)
        self.Scatter = Scattering(self)
        self.Properties = Properties(self)
    
    def generate_structure(self):
        "Combines the atomic positions with symmetry operations, returning the full structure as an Atoms class"
        
        uvw,type,label,occ,uiso,mxmymz = self.Atoms.get()
        
        new_uvw = np.empty([0,3])
        new_type = []
        new_label = []
        new_occupancy = []
        new_uiso= []
        new_mxmymz= np.empty([0,3])
        for n in range(len(uvw)):
            sympos, symmag = self.Symmetry.symmetric_coordinates(uvw[n],mxmymz[n])
            Nsympos = len(sympos)
            #symmag = self.MagSymmetry.symmetric_coordinates(mxmymz[n])
            #symmag = np.tile(mxmymz[n],Nsympos).reshape([-1,3])
            
            # Append to P1 atoms arrays
            new_uvw = np.append(new_uvw,sympos,axis=0)
            new_mxmymz = np.append(new_mxmymz,symmag,axis=0)
            new_type += [type[n]]*Nsympos
            new_label += [label[n]]*Nsympos
            new_occupancy += [occ[n]]*Nsympos
            new_uiso += [uiso[n]]*Nsympos
        
        u = new_uvw[:,0]
        v = new_uvw[:,1]
        w = new_uvw[:,2]
        new_type = np.array(new_type)
        new_label = np.array(new_label)
        new_occupancy = np.array(new_occupancy)
        new_uiso = np.array(new_uiso)
        
        self.Structure = Atoms(u,v,w,new_type,new_label,new_occupancy,new_uiso,new_mxmymz)
    
    def new_cell(self,latt=[1.0]):
        "Replace the lattice parameters"
        
        self.Cell.latt(latt)
    
    def new_atoms(self,u=[0],v=[0],w=[0],type=None,
                 label=None,occupancy=None,uiso=None,mxmymz=None):
        "Replace current atomic positions with new ones and regenerate structure"
        
        self.Atoms = Atoms(u,v,w,type,label,occupancy,uiso,mxmymz)
        self.generate_structure()
    
    def generate_lattice(self,U=1,V=1,W=0):
        """
        Generate lattice of a cell
         latt = genlatt(crys,U=1,V=1,W=0)
        """
        
        uvw,type,label,occ,uiso,mxmymz = self.Structure.generate_lattice(U,V,W,centred=False)
        R = self.Cell.calculateR(uvw)
        lp = self.Cell.generate_lattice(U+1, V+1, W+1)
        
        latt = Crystal()
        latt.new_cell(lp)
        
        UV = latt.Cell.UV()
        uvw = latt.Cell.indexR(R)
        latt.new_atoms(u=uvw[:,0],v=uvw[:,1],w=uvw[:,2],
                      type=type,label=label,occupancy=occ,uiso=uiso)
        return latt
    
    def generate_superstructure(self,P):
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
        
        super = Superstructure(self,P)
        return super
    
    def add_parent(self,parent,P):
        """
        Add parent structure, returning Crystal as superstructure
        
        """
        
        uvw,type,label,occupancy,uiso,mxmymz = self.Structure.get()
        latt = self.Cell.lp()
        
        super = Superstructure(parent,P)
        super.name = self.name 
        
        super.new_cell(latt)
        super.new_atoms(uvw[:,0],uvw[:,1],uvw[:,2],type,label,occupancy,uiso,mxmymz)
        
        return super
    
    def start_gui(self):
        "Start Crystal GUI"
        try:
            from Dans_Diffraction.classes_gui import Crystalgui
            Crystalgui(self)
        except ImportError:
            print('Sorry, you need to install tkinter!')
    
    def info(self):
        "Prints information about the crystal structure"
        print('')
        print('###########################################')
        print(self.name)
        #print('')
        #print('Formula: {}'.format(fc.molname(self)))
        #fc.molfraction(self)
        print('')
        self.Cell.info()
        print('Density: %6.3f g/cm' % self.Properties.density())
        print('')
        self.Structure.info()
        print('')
        #print "To see the full list of structure positions, type Crystal.Structure.info()"

class Cell:
    """
    Contains lattice parameters and unit cell
    Provides tools to convert between orthogonal and lattice bases in real and reciprocal space.
    
    E.G.
        UC = Cell() # instantiate the Cell object
        UC.latt([2.85,2.85,10.8,90,90,120]) # Define the lattice parameters from a list
        UC.tth([0,0,12],energy_kev=8.0) # Calculate the two-theta of a reflection
        UC.lp() # Returns the current lattice parameters
    """
    def __init__(self,a=1.0,b=1.0,c=1.0,alpha=90.,beta=90.0,gamma=90.0):
        " Called on intialisation, defines lattice defaults"
        self.latt([a,b,c,alpha,beta,gamma])
    
    def latt(self,lattice_parameters):
        """ 
        Generate lattice parameters with list
          latt([1]) -> a=b=c=1,alpha=beta=gamma=90
          latt([1,2,3]) -> a=1,b=2,c=3,alpha=beta=gamma=90
          latt([1,2,3,120]) -> a=1,b=2,c=3,alpha=beta=90,gamma=120
          latt([1,2,3,10,20,30]) -> a=1,b=2,c=3,alpha=10,beta=20,gamma=30
        """
        
        self.a = float(lattice_parameters[0])
        self.alpha = 90.0
        self.beta = 90.0
        self.gamma = 90.0
        
        if len(lattice_parameters) == 1:
            self.b = 1.0*self.a
            self.c = 1.0*self.a
        else:
            self.b = float(lattice_parameters[1])
            self.c = float(lattice_parameters[2])
        
        if len(lattice_parameters) == 4:
            self.gamma = 120.0
        
        if len(lattice_parameters) == 6:
            self.alpha = float(lattice_parameters[3])
            self.beta  = float(lattice_parameters[4])
            self.gamma = float(lattice_parameters[5])
    
    def fromcif(self,cifvals):
        "Import lattice parameters from a cif dictionary"
        
        a,da = fg.readstfm(cifvals['_cell_length_a'])
        b,db = fg.readstfm(cifvals['_cell_length_b'])
        c,dc = fg.readstfm(cifvals['_cell_length_c']) 
        alpha,dalpha = fg.readstfm(cifvals['_cell_angle_alpha'])
        beta,dbeta = fg.readstfm(cifvals['_cell_angle_beta'])
        gamma,dgamma = fg.readstfm(cifvals['_cell_angle_gamma'])
        
        self.latt([a,b,c,alpha,beta,gamma])
    
    def lp(self):
        " Returns the lattice parameters"
        return self.a,self.b,self.c,self.alpha,self.beta,self.gamma
    
    def UV(self):
        """
         Returns the unit cell as a [3x3] array, [A,B,C]
         The vector A is directed along the x-axis
        """
        return fc.latpar2UV_rot(*self.lp())
    
    def UVstar(self):
        " Returns the reciprocal unit cell as a [3x3] array, [A*,B*,C*]"
        return fc.RcSp(self.UV())
    
    def volume(self):
        " Returns the volume of the unit cell, in A^3"
        return fc.calc_vol(self.UV())
    
    def Bmatrix(self):
        """
        Calculate the Busing and Levy B matrix from a real space UV
        """
        return fc.Bmatrix(self.UV())
    
    def info(self):
        " Prints the lattice parameters and cell volume"
        print('a = %6.3f A,  b = %6.3f A,  c = %6.3f A\nA = %8.2f,  B = %8.2f,  G = %8.2f' % self.lp())
        print('Volume: %6.2f A^3' % self.volume())
    
    def generate_lattice(self,U,V,W):
        """
         Returns the lattice parameters of a larger lattice
        """
        
        return U*self.a,V*self.b,W*self.c,self.alpha,self.beta,self.gamma
    
    def calculateQ(self,HKL):
        """
        Convert coordinates [h,k,l], in the basis of the reciprocal lattice, to
        coordinates [x,y,z], in an orthogonal basis, in units of A-1
                Q(x,y,z) = hA* + kB* + lC*
        
        E.G.
            Q = Cell.calculateQ([1,0,0]) # for a hexagonal system, a = 2.85
            >>> Q = array([[2.2046264, 1.2728417, 0.0000000]])
        """
        HKL = np.reshape( np.asarray(HKL,dtype=np.float) ,[-1,3])
        return np.dot(HKL,self.UVstar())
    
    def indexQ(self,Q):
        """
        Convert coordinates [x,y,z], in an orthogonal basis, to
        coordinates [h,k,l], in the basis of the reciprocal lattice
                    H(h,k,l) = Q(x,y,z) / [A*,B*,C*]
        
        E.G.
            HKL = indexQ([2.2046264, 1.2728417, 0.0000000]) # for a hexagonal system, a = 2.85
            >>> HKL = [1,0,0]
        """
        Q = np.reshape( np.asarray(Q,dtype=np.float) ,[-1,3])
        return fc.indx(Q,self.UVstar())
    
    def calculateR(self,UVW):
        """
        Convert coordinates [u,v,w], in the basis of the unit cell, to
        coordinates [x,y,z], in an orthogonal basis, in units of A
                    R(x,y,z) = uA + vB + wC
        
        E.G.
            R = Cell.calculateR([0.1,0,0]) # for a hexagonal system, a = 2.85
            >>> R = array([[0.285, 0, 0]])
        """
        UVW = np.reshape( np.asarray(UVW,dtype=np.float) ,[-1,3])
        return np.dot(UVW,self.UV())
    
    def indexR(self,R):
        """
        Convert coordinates [x,y,z], in an orthogonal basis, to
        coordinates [u,v,w], in the basis of the unit cell
                U(u,v,w) = R(x,y,z) / [A,B,C]
        
        E.G.
            UVW = indexR([0.285, 0, 0]) # for a hexagonal system, a = 2.85
            >>> UVW = [0.1,0,0]
        """
        R = np.reshape( np.asarray(R,dtype=np.float) ,[-1,3])
        return fc.indx(R,self.UV())
    
    def Qmag(self,HKL):
        "Returns the magnitude of wave-vector transfer of [h,k,l], in A-1"
        Q = self.calculateQ(HKL)
        return fg.mag(Q)
    
    def tth(self,HKL,energy_kev=8.048):
        "Returns the two-theta angle, in deg, of [h,k,l] at specified energy in keV"
        Qmag = self.Qmag(HKL)
        return fc.cal2theta(Qmag,energy_kev)
    
    def theta_reflection(self,HKL,energy_kev=8.048,specular=[0,0,1],theta_offset=0):
        """
        Calculate the sample angle for diffraction in reflection geometry given a particular specular direction
        """
        
        Q = self.calculateQ(HKL)
        tth = self.tth(HKL,energy_kev)
        angle = np.zeros(Q.shape[0])
        for n in range(Q.shape[0]):
            angle[n] = np.rad2deg(fg.ang(Q[n],specular))+theta_offset
        return (tth/2)+angle
    
    def theta_transmission(self,HKL,energy_kev=8.048,parallel=[0,0,1],theta_offset=0):
        """
        Calculate the sample angle for diffraction in transmission geometry given a particular direction parallel to the beam
        """
        
        Q = self.calculateQ(HKL)
        tth = self.tth(HKL,energy_kev)
        angle = np.zeros(Q.shape[0])
        for n in range(Q.shape[0]):
            angle[n] = np.rad2deg(fg.ang(Q[n],parallel))-90+theta_offset
        return (tth/2)+angle
    
    def dspace(self,HKL):
        Qmag = self.Qmag(HKL)
        return fc.caldspace(Qmag)
    
    def max_hkl(self,energy_kev=8.048,max_angle=180.0):
        "Returns the maximum index of h, k and l for a given energy"
        
        Qmax = fc.calQmag(max_angle,energy_kev)
        Qpos = [[Qmax,Qmax,Qmax],
                [-Qmax,Qmax,Qmax],
                [Qmax,-Qmax,Qmax],
                [-Qmax,-Qmax,Qmax],
                [Qmax,Qmax,-Qmax],
                [-Qmax,Qmax,-Qmax],
                [Qmax,-Qmax,-Qmax],
                [-Qmax,-Qmax,-Qmax]]
        hkl = self.indexQ(Qpos)
        return np.ceil(np.abs(hkl).max(axis=0)).astype(int)
    
    def all_hkl(self,energy_kev=8.048,max_angle=180.0):
        "Returns an array of all (h,k,l) reflection at this energy"
        
        Qmax = fc.calQmag(max_angle,energy_kev)
        
        # Find the largest indices
        hmax,kmax,lmax = self.max_hkl(energy_kev, max_angle)
        # Generate the grid
        HKL = fc.genHKL([hmax,-hmax],[kmax,-kmax],[lmax,-lmax])
        # Some will be above the threshold
        Qm = self.Qmag(HKL)
        return HKL[Qm<=Qmax,:]
    
    def sort_hkl(self,HKL,ascend=True):
        "Returns array of (h,k,l) sorted by two-theta"
        
        HKL = np.reshape( np.asarray(HKL,dtype=np.float) ,[-1,3])
        Qm = self.Qmag(HKL)
        idx = np.argsort(Qm)
        return HKL[idx,:]
    
    def reciprocal_space_plane(self,x_axis=[1,0,0],y_axis=[0,1,0],centre=[0,0,0],q_max=4.0,cut_width=0.05):
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
        x_cart,y_cart,z_cart = fc.orthogonal_axes(x_cart,y_cart)
        c_cart = self.calculateQ(centre)
        
        # Generate lattice of reciprocal space points
        hmax,kmax,lmax  = fc.maxHKL(q_max,self.UVstar())
        HKL = fc.genHKL([hmax,-hmax],[kmax,-kmax],[lmax,-lmax])
        HKL = HKL + centre # reflection about central reflection
        Q = self.calculateQ(HKL)
        
        # generate box in reciprocal space
        CELL = np.array([2*q_max*x_cart,-2*q_max*y_cart,cut_width*z_cart])
        
        box_coord = fg.index_coordinates(Q-c_cart, CELL)
        incell = np.all(np.abs(box_coord)<=0.5,axis=1)
        plane_coord = 2*q_max*box_coord[incell,:]
        return plane_coord[:,0],plane_coord[:,1],HKL[incell,:]

# Atoms section

class Atoms:
    """
    Contains properties of atoms within the crystal
    Each atom has properties:
        u,v,w >> atomic coordinates, in the basis of the unit cell
        type >> element species, given as element name, e.g. 'Fe'
        label >> Name of atomic position, e.g. 'Fe1'
        occupancy >> Occupancy of this atom at this atomic position
        uiso >> atomic displacement factor (ADP) <u^2>
    """
    
    _default_atom = 'Fe'
    _default_uiso = 1.0/(8*np.pi**2) # B=1
    
    def __init__(self,u=[0],v=[0],w=[0],type=None,
                 label=None,occupancy=None,uiso=None,mxmymz=None):
        " Initialisation, defines Atoms defaults"
        self.u = np.asarray(u,dtype=np.float).reshape(-1)
        self.v = np.asarray(v,dtype=np.float).reshape(-1)
        self.w = np.asarray(w,dtype=np.float).reshape(-1)
        Natoms = len(u)
        
        #---Defaults---
        # type
        if type is None:
            self.type = np.asarray([self._default_atom]*Natoms)
        else:
            self.type = type
        # label
        if label is None:
            self.label = np.asarray([self._default_atom]*Natoms)
        else:
            self.label = label
        # occupancy
        if occupancy is None:
            self.occupancy = np.ones(Natoms)
        else:
            self.occupancy = occupancy
        # Uiso
        if uiso is None:
            self.uiso = self._default_uiso*np.ones(Natoms)
        else:
            self.uiso = uiso
        # Mag vector mxmymz
        if mxmymz is None:
            self.mx = np.zeros(Natoms)
            self.my = np.zeros(Natoms)
            self.mz = np.zeros(Natoms)
        else:
            mpos = np.asarray(mxmymz,dtype=np.float).reshape(Natoms,3)
            self.mx = mpos[:,0]
            self.my = mpos[:,1]
            self.mz = mpos[:,2]
    
    def __call__(self,u=[0],v=[0],w=[0],type=None,
                 label=None,occupancy=None,uiso=None,mxmymz=None):
        "Re-initialises the class, generating new atomic positions"
        self.__init__(u,v,w,type,label,occupancy,uiso,mxmymz=None)
    
    def fromcif(self,cifvals):
        "Import atom parameters from a cif dictionary"
        
        keys = cifvals.keys()
        
        # Get atom names & labels
        label = cifvals['_atom_site_label']
        
        if '_atom_site_type_symbol' in keys: 
            element = [x.strip('+-0123456789') for x in cifvals['_atom_site_type_symbol']]
        else:
            element = [x.strip('+-0123456789') for x in cifvals['_atom_site_label']]
        
        # Get other properties
        if '_atom_site_U_iso_or_equiv' in keys: 
            uiso = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_U_iso_or_equiv']])
        elif '_atom_site_B_iso_or_equiv' in keys:
            biso = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_B_iso_or_equiv']])
            uiso = fc.biso2uiso(biso)
        else:
            uiso = np.zeros(len(element))
        if '_atom_site_occupancy' in keys: 
            occ = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_occupancy']])
        else:
            occ = np.ones(len(element))
        
        # Get coordinates
        u = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_x']])  
        v = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_y']])
        w = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_fract_z']])
        
        # Get magnetic vectors
        mx = np.zeros(len(u))
        my = np.zeros(len(u))
        mz = np.zeros(len(u))
        if '_atom_site_moment_label' in keys:
            mag_atoms = cifvals['_atom_site_moment_label']
            mxi = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_x']]) 
            myi = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_y']]) 
            mzi = np.array([fg.readstfm(x)[0] for x in cifvals['_atom_site_moment_crystalaxis_z']]) 
            for n,ma in enumerate(mag_atoms):
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
    
    def changeatom(self,idx,u=None,v=None,w=None,type=None,
                 label=None,occupancy=None,uiso=None,mxmymz=None):
        """
        Change an atoms properties
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
            self.label[idx] = label
        
        if occupancy is not None:
            self.occupancy[idx] = occupancy
        
        if uiso is not None:
            self.uiso[idx] = uiso
        
        if mxmymz is not None:
            mpos = np.asarray(mxmymz,dtype=np.float).reshape(-1,3)
            self.mx[idx] = mpos[:,0]
            self.my[idx] = mpos[:,1]
            self.mz[idx] = mpos[:,2]
    
    def addatom(self,u=0,v=0,w=0,type=None,label=None,occupancy=None,uiso=None,mxmymz=None):
        " Adds a new atom"
        
        self.u = np.append(self.u,[u])
        self.v = np.append(self.v,[v])
        self.w = np.append(self.w,[w])
        
        # type
        if type is None:
            self.type = np.append(self.type,[self._default_atom])
        else:
            self.type = np.append(self.type,[type])
        
        # label
        if label is None:
            self.label = np.append(self.label,[self._default_atom])
        else:
            self.label = np.append(self.label,[label])
        
        # occupancy
        if occupancy is None:
            self.occupancy = np.append(self.occupancy,[1.0])
        else:
            self.occupancy = np.append(self.occupancy,[occupancy])
        
        # Uiso
        if uiso is None:
            self.uiso = np.append(self.uiso,[self._default_uiso])
        else:
            self.uiso = np.append(self.uiso,[uiso])
        
        if mxmymz is None:
            self.mx = np.append(self.mx,[0])
            self.my = np.append(self.my,[0])
            self.mz = np.append(self.mz,[0])
        else:
            self.mx = np.append(self.mx,mxmymz[0])
            self.my = np.append(self.my,mxmymz[1])
            self.mz = np.append(self.mz,mxmymz[2])
    
    def check(self):
        "Checks the validity of the contained attributes"
        
        good = True
        
        # Check lengths
        Natoms = len(self.u)
        if len(self.v) != Natoms: good=False; print('Cell.v is the wrong length!')
        if len(self.w) != Natoms: good=False; print('Cell.w is the wrong length!')
        if len(self.type) != Natoms: good=False; print('Cell.type is the wrong length!')
        if len(self.label) != Natoms: good=False; print('Cell.label is the wrong length!')
        if len(self.occupancy) != Natoms: good=False; print('Cell.occupancy is the wrong length!')
        if len(self.uiso) != Natoms: good=False; print('Cell.uiso is the wrong length!')
        if len(self.mx) != Natoms: good=False; print('Cell.mx is the wrong length!')
        if len(self.my) != Natoms: good=False; print('Cell.my is the wrong length!')
        if len(self.mz) != Natoms: good=False; print('Cell.mz is the wrong length!')
        
        # Check atom types
        atoms = fc.atom_properties(fields='Element')
        for atom in self.type:
            if atom not in atoms:
                good=False
                print(atom,' has no properties assigned.')
        
        # Check occupancy
        for n in range(Natoms):
            if self.occupancy[n] > 1:
                good=False
                print('Atom %d has occupancy greater than 1!' % n)
            elif self.occupancy[n] < 0:
                good=False
                print('Atom %d has occupancy less than 0!' % n)
        
        # Check uiso
        for n in range(Natoms):
            if self.uiso[n] < 0:
                good=False
                print('Atom %d has uiso less than 0!' % n)
            elif self.uiso[n] > 0.1:
                good=False
                print('Atom %d has uiso greater than 0.1!' % n)
        
        if good:
            print("Atoms look good.")
    
    def uvw(self):
        """
        Returns a [nx3] array of current positions
        """
        return np.asarray([self.u,self.v,self.w],dtype=np.float).T
    
    def mxmymz(self):
        "Returns a [nx3] array of magnetic vectors"
        return np.asarray([self.mx,self.my,self.mz],dtype=np.float).T
    
    def get(self):
        """
        Returns the structure arrays
         uvw,type,label,occupancy,uiso,mxmymz = Atoms.get()
        """
        return self.uvw(),np.asarray(self.type),np.asarray(self.label),np.asarray(self.occupancy),np.asarray(self.uiso),self.mxmymz()
    
    def generate_lattice(self,U=1,V=1,W=0,centred=True):
        """
        Expand the atomic positions beyond the unit cell, creating a lattice
            uvw,type,label,occ,uiso,mxmymz = self.generate_lattice(U,V,W,centred)
              U,V,W = maximum lattice index to loop to
              centred = if True, positions will loop from e.g. -U to U,
                        otherwise, will loop from e.g. 0 to U
              uvw,type,label,occ,uiso,mxmymz = standard array outputs of Atoms
        """
        
        uvw,type,label,occ,uiso,mxmymz = self.get()
        
        new_uvw = np.ndarray([0,3])
        new_type = np.ndarray([0])
        new_label = np.ndarray([0])
        new_occupancy = np.ndarray([0])
        new_uiso = np.ndarray([0])
        new_mxmymz = np.ndarray([0,3])
        if centred:
            urange = range(-U,U+1)
            vrange = range(-V,V+1)
            wrange = range(-W,W+1)
        else:
            urange = range(0,U+1)
            vrange = range(0,V+1)
            wrange = range(0,W+1)
        
        for uval in urange:
            for vval in vrange:
                for wval in wrange:
                    new_uvw = np.vstack([new_uvw,uvw+[uval,vval,wval]])
                    new_type = np.hstack([new_type,type])
                    new_label = np.hstack([new_label,label])
                    new_occupancy = np.hstack([new_occupancy,occ])
                    new_uiso = np.hstack([new_uiso,uiso])
                    new_mxmymz = np.vstack([new_mxmymz,mxmymz])
        
        return new_uvw, new_type, new_label, new_occupancy, new_uiso, new_mxmymz
    
    def weight(self):
        "Calculate the molecular weight in g/mol of all the atoms"
        weights = fc.atom_properties(self.type, 'Weight')
        total_weight = sum(weights*self.occupancy) #g per mol per unit cell
        return total_weight
    
    def mass_fraction(self):
        "Return the mass fraction per element"
        
        weights = fc.atom_properties(self.type, 'Weight')*self.occupancy
        total_weight = sum(weights)
        
        return weights/total_weight
    
    def info(self):
        "Prints properties of all atoms"
        
        if np.any(fg.mag(self.mxmymz())>0):
            print('  n Atom  u       v       w        Occ  Uiso      mx      my      mz      ')
            fmt = '%3.0f %4s %7.4f %7.4f %7.4f   %4.2f %6.4f   %7.4f %7.4f %7.4f'
            for n in range(0,len(self.u)):
                print(fmt % (n,self.type[n],self.u[n],self.v[n],self.w[n],self.occupancy[n],self.uiso[n],self.mx[n],self.my[n],self.mz[n]))
        else:
            print('  n Atom  u       v       w        Occ  Uiso')
            fmt = '%3.0f %4s %7.4f %7.4f %7.4f   %4.2f %6.4f'
            for n in range(0,len(self.u)):
                print(fmt % (n,self.type[n],self.u[n],self.v[n],self.w[n],self.occupancy[n],self.uiso[n]))

class Symmetry:
    """
    Contains symmetry information about the crystal, including the symmetry operations.
    """
    
    spacegroup = 'P1'
    spacegroup_number = 1
    symmetry_operations = ['x,y,z']
    symmetry_operations_magnetic = ['x,y,z']
    #symmetry_operations_time = [1]
    
    def __init__(self,symmetry_operations=None,symmetry_operations_magnetic=None):
        "Initialises the symmetry group"
        
        if symmetry_operations is not None:
            self.addsym(symmetry_operations, symmetry_operations_magnetic)
        
        self.generate_matrices()
    
    def fromcif(self,cifvals):
        "Import symmetry information from a cif dictionary"
        
        keys = cifvals.keys()
        
        # Get symmetry operations
        ops = ['+1/2','+1/3','+2/3','+1/4','+3/4','+1/6','+5/6',\
               '1/2+','1/3+','2/3+','1/4+','3/4+','1/6+','5/6+',\
               '-1/2','-1/3','-2/3','-1/4','-3/4','-1/6','-5/6',\
               '1/2-','1/3-','2/3-','1/4-','3/4-','1/6-','5/6-'] # replace these in magnetic symmetry operations
        if '_symmetry_equiv_pos_as_xyz' in keys:
            symops = cifvals['_symmetry_equiv_pos_as_xyz']
            symcen = ['x,y,z']
            
            # add magnetic symmetries (symops without translation)
            symops_mag = [fg.multi_replace(sp, ops, '') for sp in symops]
            cenops_mag = [fg.multi_replace(sp, ops, '') for sp in symcen]
            #self.symmetry_operations_time = [1]*len(symops)
            #self.centring_operations_time = [1]
        elif '_space_group_symop_operation_xyz' in keys:
            symops = cifvals['_space_group_symop_operation_xyz']
            symcen = ['x,y,z']
            
            # add magnetic symmetries (symops without translation)
            symops_mag = [fg.multi_replace(sp, ops, '') for sp in symops]
            cenops_mag = [fg.multi_replace(sp, ops, '') for sp in symcen]
            #self.symmetry_operations_time = [1]*len(symops)
            #self.centring_operations_time = [1]
        elif '_space_group_symop_magn_operation_xyz' in keys:
            symops_tim = cifvals['_space_group_symop_magn_operation_xyz']
            # Each symop given with time value: x,y,z,+1, separate them:
            symops = [','.join(s.split(',')[:3]) for s in symops_tim] # x,y,z
            symtim = [np.int(s.split(',')[-1]) for s in symops_tim] # +1
            if '_space_group_symop_magn_operation_mxmymz' in keys:
                symmag = cifvals['_space_group_symop_magn_operation_mxmymz'] # mx,my,mz
            else:
                symmag = [fg.multi_replace(sp, ops, '') for sp in symops]
            
            # Centring vectors also given in this case
            symcen_tim = cifvals['_space_group_symop_magn_centering_xyz']
            symcen = [','.join(s.split(',')[:3]) for s in symcen_tim] # x,y,z
            symcentim = [np.int(s.split(',')[-1]) for s in symcen_tim] # +1
            if '_space_group_symop_magn_centering_mxmymz' in keys:
                symcenmag = cifvals['_space_group_symop_magn_centering_mxmymz'] # mx,my,mz
            else:
                symcenmag = [fg.multi_replace(sp, ops, '') for sp in symcen]
            
            # add magnetic symmetries
            symops_mag = [op.replace('m','') for op in symmag]
            cenops_mag = [op.replace('m','') for op in symcenmag]
            #self.symmetry_operations_time = symtim
            #self.centring_operations_time = symcentim
            
        else:
            symops = ['x,y,z']
            symcen = ['x,y,z']
            symops_mag = ['x,y,z']
            cenops_mag = ['x,y,z']
            #self.symmetry_operations_time = [1]
            #self.centring_operations_time = [1]
        self.symmetry_operations = fc.gen_symcen_ops(symops,symcen)
        #self.centring_operations = ['x,y,z']
        self.symmetry_operations_magnetic = fc.gen_symcen_ops(symops_mag,cenops_mag)
        #self.centring_operations_magnetic = ['x,y,z']
        
        # Get space group
        if '_symmetry_space_group_name_H-M' in keys:
            spacegroup = cifvals['_symmetry_space_group_name_H-M']
        elif '_space_group_name_H-M_alt' in keys:
            spacegroup = cifvals['_space_group_name_H-M_alt']
        elif '_space_group_magn_name_BNS' in keys:
            spacegroup = cifvals['_space_group_magn_name_BNS']
        elif len(symops) == 1:
            spacegroup = 'P1'
        else:
            spacegroup = ''
        self.spacegroup = spacegroup
        
        
        if '_symmetry_Int_Tables_number' in keys:
            sgn = float(cifvals['_symmetry_Int_Tables_number'])
        elif '_space_group_IT_number' in keys:
            sgn = float(cifvals['_space_group_IT_number'])
        elif '_space_group_magn_number_BNS' in keys:
            sgn = float(cifvals['_space_group_magn_number_BNS'].strip('\'"'))
        elif spacegroup == 'P1':
            sgn = 1
        else:
            sgn = 0
        self.spacegroup_number = sgn
        
        self.generate_matrices()
    
    def changesym(self,idx,operation):
        "Change a symmetry operation"
        
        self.symmetry_operations[idx] = operation
        self.generate_matrices()
    
    def invert_magsym(self,idx):
        "Invert the time symmetry of a magnetic symmetry"
        
        ID = np.asarray(idx).reshape(-1)
        for idx in ID:
            old_op = self.symmetry_operations_magnetic[idx]
            new_op = old_op.replace('x','-x').replace('y','-y').replace('z','-z').replace('--','+').replace('+-','-')
            self.symmetry_operations_magnetic[idx] = new_op
    
    def addsym(self,operations,mag_operations=None):
        """
        Add symmetry operations
            Symmetry.addsym('x,y,z+1/2') >> adds single symmetry operation, magnetic operation is infered
            Symmetry.addsym(['x,y,z+1/2','z,x,y']) >> adds multiple symmetry operation, magnetic operations are infered
            Symmetry.addsym('x,y,z+1/2','x,y,-z') >> adds single symmetry operation + magnetic operation
        """
        
        operations = np.array(operations).reshape(-1)
        mag_operations = np.array(mag_operations).reshape(-1)
        
        if mag_operations[0] is None:
            mag_operations = fc.symmetry_ops2magnetic(operations)
        
        for op,mag_op in zip(operations,mag_operations): 
            if op not in self.symmetry_operations:
                self.symmetry_operations += [op]
                self.symmetry_operations_magnetic += [mag_op]
        self.generate_matrices()
    
    def addcen(self,operations,mag_operations=None):
        """
        Apply centring operations to current symmetry operations
        """
        
        operations = np.array(operations).reshape(-1)
        mag_operations = np.array(mag_operations).reshape(-1)
        
        if mag_operations[0] is None:
            mag_operations = fc.symmetry_ops2magnetic(operations)
        
        self.symmetry_operations = fc.gen_symcen_ops(self.symmetry_operations,operations)
        self.symmetry_operations_magnetic = fc.gen_symcen_ops(self.symmetry_operations_magnetic,mag_operations)
        self.generate_matrices()
    
    def generate_matrices(self):
        "Generates the symmetry matrices from string symmetry operations"
        
        self.symmetry_matrices = fc.gen_sym_mat(self.symmetry_operations)
    
    def symmetric_coordinates(self,UVW,MXMYMZ=None):
        """
        Returns array of symmetric coordinates
        Uses DansCrystalProgs.gen_sym_pos
        Returns coordinates wrapped within the unit cell, with identical positions removed.
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.symmetric_coordinates([0.1,0,0])
            >>>  array([[0.1, 0.0, 0.0],
                        [0.9, 0.0, 0.0],
                        [0.0, 0.1, 0.0]])
        """
        
        # Apply symmetry operations to atomic positions
        u,v,w = UVW
        sym_uvw = fc.gen_sym_pos(self.symmetry_operations,u,v,w)
        sym_uvw = fc.fitincell(sym_uvw)
        
        # Remove identical positions
        unique_uvw,uniqueidx,matchidx = fg.unique_vector(sym_uvw,tol=0.01)
        
        if MXMYMZ is None:
            return unique_uvw
        
        # Apply magnetic symmetry operations to magnetic vectors
        mx,my,mz = MXMYMZ
        sym_xyz = fc.gen_sym_pos(self.symmetry_operations_magnetic,mx,my,mz)
        
        # Remove positions that were identical
        unique_xyz = sym_xyz[uniqueidx]
        return unique_uvw,unique_xyz
    
    def symmetric_coordinate_operations(self,UVW,MXMYMZ=None):
        """
        Returns array of symmetric operations for given position
        Uses DansCrystalProgs.gen_sym_pos
        Returns list of identical symmetry operations
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.symmetric_coordinate_operations([0.1,0,0])
            >>>  []
        """
        
        # Apply symmetry operations to atomic positions
        u,v,w = UVW
        sym_uvw = fc.gen_sym_pos(self.symmetry_operations,u,v,w)
        sym_uvw = fc.fitincell(sym_uvw)
        
        # Remove identical positions
        unique_uvw,uniqueidx,matchidx = fg.unique_vector(sym_uvw,tol=0.01)
        
        if MXMYMZ is None:
            return np.array(self.symmetry_operations)[uniqueidx]
        
        return np.array(self.symmetry_operations)[uniqueidx], np.array(self.symmetry_operations_magnetic)[uniqueidx]
    
    def print_symmetric_coordinate_operations(self,UVW):
        """
        Returns array of symmetric operations for given position
        Uses DansCrystalProgs.gen_sym_pos
        Returns list of identical symmetry operations
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.symmetric_coordinate_operations([0.1,0,0])
            >>>  []
        """
        
        # Apply symmetry operations to atomic positions
        u,v,w = UVW
        sym_uvw = fc.gen_sym_pos(self.symmetry_operations,u,v,w)
        sym_uvw = fc.fitincell(sym_uvw)
        
        # Remove identical positions
        unique_uvw,uniqueidx,matchidx = fg.unique_vector(sym_uvw,tol=0.01)
        
        print(' u       v       w                   Symmetry    Magnetic Symmetry')
        for n in range(len(sym_uvw)):
            print(' %7.4f %7.4f %7.4f %20s %20s' % (sym_uvw[n,0],sym_uvw[n,1],sym_uvw[n,2],self.symmetry_operations[n],self.symmetry_operations_magnetic[n]))
    
    def symmetric_reflections(self,HKL):
        """
        Returns array of symmetric reflection indices
        Uses DansCrystalProgs.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            Symmetry.symmetric_reflections([1,0,0])
            >>>  array([[1,  0, 0],
                        [-1, 0, 0],
                        [0, -1, 0]])
        """
        
        NsymOps = len(self.symmetry_operations)
        
        HKL = np.asarray(HKL,dtype=np.float).reshape((-1,3))
        
        symHKL = np.zeros([len(HKL)*NsymOps,3])
        ii=0
        for n in range(len(HKL)):
            for m in range(NsymOps):
                opHKL = np.dot(HKL[n],self.symmetry_matrices[m])
                symHKL[ii,:] = opHKL[:3]
                ii +=1
        return symHKL
    
    def symmetric_reflections_unique(self,HKL):
        """
        Returns array of symmetric reflection indices, with identical reflections removed
        Uses DansCrystalProgs.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            Symmetry.symmetric_reflections([1,1,0])
            >>>  array([[1,  1, 0],
                        [-1,-1, 0]])
        """
        
        symHKL = self.symmetric_reflections(HKL)
        
        # Remove identical positions
        unique_hkl,uniqueidx,matchidx = fg.unique_vector(symHKL,tol=0.01)
        return unique_hkl
    
    def symmetric_reflections_count(self,HKL):
        """
        Returns array of symmetric reflection indices, with identical reflections removed plus the counted sum of each reflection
        Uses DansCrystalProgs.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            HKL, count = Symmetry.symmetric_reflections([1,1,0])
            >>>  HKL = array([[1,  1, 0],
                              [-1,-1, 0]])
            >>> count = array([2,1])
                        
        """
        
        symHKL = self.symmetric_reflections(HKL)
        
        # Remove identical positions
        unique_hkl,uniqueidx,matchidx = fg.unique_vector(symHKL,tol=0.01)
        
        # Count reflections at identical positions
        count = np.zeros(len(unique_hkl))
        for n in range(len(unique_hkl)):
            count[n] = matchidx.count(n)
        return unique_hkl, count
    
    def symmetric_intensity(self,HKL,I,dI=None):
        """
        Returns symmetric reflections with summed intensities of repeated reflections
        Uses DansCrystalProgs.gen_sym_mat
        E.G.
            Symmetry.symmetry_operations = ['x,y,z','-x,-y,-z','y,x,z']
            Symmetry.generate_matrixes()
            HKL, I = Symmetry.symmetric_intensity([1,1,0],10)
            >>>  HKL = array([[1,  1, 0],
                              [-1,-1, 0]])
            >>> I = array([20,10])
            OR
            HKL, I, dI = Symmetry.symmetric_intensity([1,1,0],10,1)
            >>>  HKL = array([[1,  1, 0],
                              [-1,-1, 0]])
            >>> I = array([20,10])
            >>> dI = array([2,1])
        """
        
        HKL = np.asarray(HKL,dtype=np.float).reshape((-1,3))
        I = np.asarray(I,dtype=np.float)
        
        NsymOps = len(self.symmetry_operations)
        
        symHKL = self.symmetric_reflections(HKL)
        symI = np.repeat(I,NsymOps) # repeats each element of I
        
        # Remove identical positions
        unique_hkl,uniqueidx,matchidx = fg.unique_vector(symHKL,tol=0.01)
        unique_I = symI[uniqueidx]
        
        # Sum intensity at duplicate positions
        count = np.zeros(len(unique_hkl))
        for n in range(len(unique_hkl)):
            count[n] = matchidx.count(n)
        
        sum_I = unique_I*count
        
        if dI is None:
            return unique_hkl, sum_I
        
        dI = np.asarray(dI,dtype=np.float)
        sym_dI = np.repeat(dI,NsymOps) # repeats each element of I
        unique_dI = sym_dI[uniqueidx]
        sum_dI = unique_dI*count # *np.sqrt(count)?
        return unique_hkl, sum_I, sum_dI
    
    def symmetric_magnetic_vectors(self,MXMYMZ):
        """
        NOT COMPLETE
        """
        
        for n,(mx,my,mz) in enumerate(MXMYMZ):
            """
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
            
            mag_uvw = fc.gen_symcen_pos(S,mx,my,mz)
    
    def info(self):
        "Prints the symmetry information"
        
        print 'Spacegoup: %s (%g)' % (self.spacegroup,self.spacegroup_number)
        
        print 'Symmetry operations:'
        for n in range(len(self.symmetry_operations)):
            x1 = self.symmetry_operations[n].strip('\"\'')
            x2 = self.symmetry_operations_magnetic[n].strip('\"\'').replace('x','mx').replace('y','my').replace('z','mz')
            print('%2d %25s %25s' %(n,x1,x2))
        #print 'Centring operations:'
        #for n in range(len(self.centring_operations)):
        #    x1 = self.centring_operations[n].strip('\"\'')
        #    x2 = self.centring_operations_magnetic[n].strip('\"\'').replace('x','mx').replace('y','my').replace('z','mz')
        #    print '%2d %25s %25s' %(n,x1,x2)
        print '-----------------------------------------------------'
        print('')

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
    
    Parent = Crystal()
    P = [[1,0,0],[0,1,0],[0,0,1]]
    
    def __init__(self,Parent,P):
        
        # Instatiate crystal attributes
        self.Cell = Cell()
        self.Symmetry = Symmetry()
        self.Atoms= Atoms()
        
        # Add exta functions
        #self.Plot = Plotting(self)
        #self.Scatter = Scattering(self)
        
        self.name = Parent.name + ' supercell'
        self.P = np.asarray(P,dtype=np.float)
        self.Parent = Parent
        newUV = Parent.Cell.calculateR(P)
        self.new_cell(fc.UV2latpar(newUV))
        
        # Build lattice of points
        UVW = fc.genHKL(2*np.max(P))
        real_lattice = Parent.Cell.calculateR(UVW)
        
        # Determine lattice points within the cell
        indx_lattice = fc.indx(real_lattice, newUV)
        isincell = np.all(indx_lattice<=0.99,axis=1) * np.all(indx_lattice>-0.01,axis=1) 
        #for n in range(len(real_lattice)):
        #    print UVW[n,:],real_lattice[n,:],indx_lattice[n,:],isincell[n]
        UVW = UVW[isincell,:]
        #print UVW
        Ncell = len(UVW)
        
        # Increase scale to match size of cell
        self.scale = Parent.scale*Ncell
        
        # Get Parent atoms
        uvw,type,label,occ,uiso,mxmymz = Parent.Structure.get()
        
        # Generate all atomic positions
        new_uvw = np.empty([0,3])
        new_mxmymz= np.empty([0,3])
        new_type = np.tile(type,Ncell)
        new_label = np.tile(label,Ncell)
        new_occ = np.tile(occ,Ncell)
        new_uiso= np.tile(uiso,Ncell)
        for n in range(len(UVW)):
            latpos = np.array([UVW[n,0]+uvw[:,0],UVW[n,1]+uvw[:,1],UVW[n,2]+uvw[:,2]]).T
            latR = Parent.Cell.calculateR(latpos)
            latuvw = fc.indx(latR, newUV)
            
            # Append to P1 atoms arrays
            new_uvw = np.append(new_uvw,latuvw,axis=0)
            new_mxmymz = np.append(new_mxmymz,mxmymz,axis=0)
        
        self.new_atoms(u=new_uvw[:,0],v=new_uvw[:,1],w=new_uvw[:,2],
                       type=new_type,label=new_label,occupancy=new_occ,uiso=new_uiso,
                       mxmymz=new_mxmymz)
        
        # Add exta functions
        self.Plot = Plotting_Superstructure(self)
        self.Scatter = Scattering(self)
        self.Properties = Properties(self)
    
    def set_scale(self):
        """
        Set scale parameter automatically
        Based on ratio of parent - to superstructure volume
        """
        vol_parent = self.Parent.Cell.volume()
        vol_super = self.Cell.volume()
        self.scale = self.Parent.scale*vol_super/vol_parent
    
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
        return np.dot(np.linalg.inv(self.P),self.Cell.UV())
    
    def parentUVstar(self):
        """
        Returns the parent reciprocal cell unit vectors defined relative to the supercell
        """
        return fc.RcSp(self.parentUV())
    
    def calculateQ_parent(self,super_hkl):
        """
        Indexes (h,k,l) coordinates in the frame same cartesian frame as the Parent structure
            Q = h'*a'* + k'*b'* + l'*c'*
        Where a'*, b'*, c'* are defined relative to the parent lattice, a*,b*,c*
            
            [qx,qy,qz] = calculateQ_parent([h',k',l'])
        """
        return np.dot(super_hkl,self.superUVstar())
    
    def superhkl2parent(self,super_HKL):
        """
        Indexes (h,k,l) coordinates in the frame of the Parent structure
            Q = h*a* + k*b* + l*c* = h'*a'* + k'*b'* + l'*c'*
            [h',k',l'] = Q/[a'*,b'*,c'*]
            
            [h,k,l] = superhkl2parent([h',k',l'])
        """
        
        Q = self.calculateQ_parent(super_HKL)
        return fc.indx(Q,self.Parent.Cell.UVstar())
    
    def parenthkl2super(self,parent_HKL):
        """
        Indexes (h,k,l) coordinates in the frame of the Parent structure
            Q = h*a* + k*b* + l*c* = h'*a'* + k'*b'* + l'*c'*
            [h',k',l'] = Q/[a'*,b'*,c'*]
            
            [h',k',l'] = parenthkl2super([h,k,l])
        """
        
        Q = self.Parent.Cell.calculateQ(parent_HKL)
        return fc.indx(Q,self.superUVstar())

class Multi_Crystal(Multi_Plotting):
    """
    Multi_Crystal class for combining multiple phases
    """
    _scattering_type = 'xray'
    def __init__(self,crystal_list):
        """
        Multi-crystal class
        """
        self.crystal_list = crystal_list
    
    def print_all_reflections(self,energy_kev=None, max_angle=180.0,print_symmetric=False):
        "Prints a list of all allowed of all crystal reflections at this energy"
        
        if energy_kev is None:
            energy_kev = fc.getenergy()
        
        HKL_list = np.empty([0,3])
        TTH_list = np.empty([0])
        I_list = np.empty([0])
        NAMES_list = np.empty([0])
        for xtl in self.crystal_list:
            xtl._scattering_type = self._scattering_type
            HKL = xtl.Cell.all_hkl(energy_kev, max_angle)
            TTH = xtl.Cell.tth(HKL,energy_kev)
            I = xtl.Scatter.intensity(HKL)
            NAMES = np.asarray(xtl.name).repeat(len(I))
            
            HKL_list = np.append(HKL_list, HKL, axis=0)
            TTH_list = np.append(TTH_list, TTH)
            I_list = np.append(I_list, I)
            NAMES_list = np.append(NAMES_list, NAMES)
        
        # Sort by TTH
        index = np.argsort(TTH_list)
        HKL_list = HKL_list[index,:]
        TTH_list = TTH_list[index]
        I_list = I_list[index]
        NAMES_list = NAMES_list[index]
        
        fmt = '(%3.0f,%3.0f,%3.0f) %8.2f  %9.2f %s' 
        
        print('Energy = %6.3f keV' % energy_kev)
        print('( h, k, l) TwoTheta  Intensity Crystal')
        #print fmt % (HKL[0,0],HKL[0,1],HKL[0,2],tth[0],inten[0],) # hkl(0,0,0)
        for n in range(1,len(TTH_list)):
            if I_list[n] < 0.01: continue
            if not print_symmetric and np.abs(TTH_list[n]-TTH_list[n-1]) < 0.01 and NAMES_list[n] == NAMES_list[n-1]: continue # only works if sorted
            print(fmt % (HKL_list[n,0],HKL_list[n,1],HKL_list[n,2],TTH_list[n],I_list[n],NAMES_list[n]))
    
    def find_close_reflections(self,HKL,energy_kev=None,max_twotheta=2,max_angle=10):
        """
        Find and print list of reflections close to the given one
        """
        
        if energy_kev is None:
            energy_kev = fc.getenergy()
        
        HKL_tth = self.crystal_list[0].Cell.tth(HKL,energy_kev)
        HKL_Q = self.crystal_list[0].Cell.calculateQ(HKL)
        
        HKL_list = np.empty([0,3])
        TTH_list = np.empty([0])
        ANGLE_list = np.empty([0])
        I_list = np.empty([0])
        NAMES_list = np.empty([0])
        for xtl in self.crystal_list:
            #xtl._scattering_type = self._scattering_type
            all_HKL = xtl.Cell.all_hkl(energy_kev, xtl._scattering_max_twotheta)
            all_TTH = xtl.Cell.tth(all_HKL,energy_kev)
            dif_TTH = np.abs(all_TTH-HKL_tth)
            all_Q = xtl.Cell.calculateQ(all_HKL)
            all_ANG = np.abs([fg.ang(HKL_Q,Q,'deg') for Q in all_Q])
            selected = (dif_TTH < max_twotheta)*(all_ANG < max_angle)
            sel_HKL = all_HKL[selected,:]
            sel_TTH = all_TTH[selected]
            sel_ANG = all_ANG[selected]
            sel_INT = xtl.Scatter.intensity(sel_HKL)
            NAMES = np.asarray(xtl.name).repeat(len(sel_INT))
            
            HKL_list = np.append(HKL_list, sel_HKL, axis=0)
            TTH_list = np.append(TTH_list, sel_TTH)
            ANGLE_list = np.append(ANGLE_list, sel_ANG)
            I_list = np.append(I_list, sel_INT)
            NAMES_list = np.append(NAMES_list, NAMES)
        
        # Sort by TTH
        index = np.argsort(TTH_list)
        HKL_list = HKL_list[index,:]
        TTH_list = TTH_list[index]
        ANGLE_list = ANGLE_list[index]
        I_list = I_list[index]
        NAMES_list = NAMES_list[index]
        
        fmt = '(%3.0f,%3.0f,%3.0f) %8.2f %8.2f %9.2f %s' 
        
        print('Energy = %6.3f keV' % energy_kev())
        print('%s Reflection: (%3.0f,%3.0f,%3.0f)' % (self.crystal_list[0].name,HKL[0],HKL[1],HKL[2]))
        print ('( h, k, l) TwoTheta Angle    Intensity Crystal')
        for n in range(0,len(TTH_list)):
            print(fmt % (HKL_list[n,0],HKL_list[n,1],HKL_list[n,2],TTH_list[n],ANGLE_list[n],I_list[n],NAMES_list[n]))
    
    def info(self):
        " Display information about the contained crystals"
        
        print("Crystals: %d" % len(self.crystal_list))
        for n,xtl in enumerate(self.crystal_list):
            print("%1.0f %s" %(n,xtl.name))


if __name__ == '__main__':
    
    xtl = Crystal()
    
