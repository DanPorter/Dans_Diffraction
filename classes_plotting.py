# -*- coding: utf-8 -*-
"""
Plotting Class "classes_plotting.py"
 functions to plot the crystal structure and simulated 
 diffraction patterns.

By Dan Porter, PhD
Diamond
2017

Version 1.2
Last updated: 05/03/18

Version History:
18/08/17 0.1    Program created
30/10/17 1.0    Main functions finished.
06/10/18 1.1    Program renamed
05/03/18 1.2    Added plt.show() to functions

@author: DGPorter
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d

from Dans_Diffraction import functions_general as fg
from Dans_Diffraction import functions_plotting as fp
from Dans_Diffraction import functions_crystallography as fc


class Plotting:
    """
    Plotting functions for the Crystal Object
    """
    def __init__(self,xtl):
        "initialise"
        self.xtl = xtl
    
    def plot_crystal(self,show_labels=False):
        """
        Plot the atomic cell in 3D
            Click and drag to rotate the structure in 3D
            Atoms are coloured according to their label
            set show_labels=True to show the label of each sphere
        """
        
        # Generate lattice
        tol = 0.05
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.generate_lattice(1,1,1)
        #uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.get()
        
        # Split atom types, color & radii
        labels,idx,invidx = np.unique(label,return_index=True,return_inverse=True)
        types = type[idx]
        colors = plt.cm.gist_rainbow(np.linspace(0,1,len(types)))
        #sizes = 300*np.ones(len(types))
        #sizes[types=='O'] = 50.
        sizes = fc.atom_properties(types, 'Radii')
        
        # Get atomic positions
        R = self.xtl.Cell.calculateR(uvw)
        #I = np.all(np.logical_and(uvw<1+tol, uvw>0-tol, occ>0.2),1)
        I = np.all(np.hstack([uvw<(1+tol),uvw>(0-tol),occ.reshape([-1,1])>0.2]),1)
        
        # Magnetic vectors
        V = self.xtl.Cell.calculateR(mxmymz/np.asarray(self.xtl.Cell.lp()[:3]))
        
        # Create plot
        fig = plt.figure(figsize=[12,12])
        ax = fig.add_subplot(111, projection='3d')
        
        # Loop over each atom type
        for n in range(len(types)):
            # don't plot unoccupied positions
            tot_occ = np.array([occ[m] for m in range(len(R)) if invidx[m] == n])
            if sum(tot_occ) == 0: continue
            
            xyz = np.array([R[m,:] for m in range(len(R)) if invidx[m] == n])
            iii = np.array([I[m] for m in range(len(R)) if invidx[m] == n])
            col = np.tile(colors[n], (len(xyz),1) )
            ax.scatter(xyz[iii,0],xyz[iii,1],xyz[iii,2],s=sizes[n],c=col,label=labels[n],cmap=colors)
            
            #mxyz = np.array([mxmymz[m,:] for m in range(len(R)) if invidx[m] == n])
            
            for m in range(len(R)): 
                if invidx[m] == n and I[m]:
                    xyz = R[m,:]
                    vec = V[m,:]
                    if fg.mag(vec) < 0.1: continue
                    vx,vy,vz = np.asarray([xyz - vec/2, xyz + vec/2]).T
                    fp.plot_arrow(vx,vy,vz,col='r',arrow_size=20,width=3)
        
        # Labels
        if show_labels:
            uvw_st,type_st,label_st,occ_st,uiso_st,mxmymz_st = self.xtl.Structure.get()
            R_st = self.xtl.Cell.calculateR(uvw_st)
            for n in range(len(R_st)):
                ax.text(R_st[n,0],R_st[n,1],R_st[n,2],'%2d: %s' %(n,label_st[n]),fontsize=10)
        
        # Create cell box
        uvw = np.array([[0.,0,0],[1,0,0],[1,0,1],[1,1,1],[1,1,0],[0,1,0],[0,1,1],\
                        [0,0,1],[1,0,1],[1,0,0],[1,1,0],[1,1,1],[0,1,1],[0,1,0],[0,0,0],[0,0,1]])
        bpos = np.dot(uvw,self.xtl.Cell.UV())
        ax.plot(bpos[:,0],bpos[:,1],bpos[:,2],c='k') # cell box
        fp.plot_arrow(bpos[[0,1],0],bpos[[0,1],1],bpos[[0,1],2],col='r',width=4) # a
        fp.plot_arrow(bpos[[0,5],0],bpos[[0,5],1],bpos[[0,5],2],col='g',width=4) # b
        fp.plot_arrow(bpos[[0,7],0],bpos[[0,7],1],bpos[[0,7],2],col='b',width=4) # c
        lim=np.max(self.xtl.Cell.lp()[:3])
        ax.set_xlim(-lim/2, lim)
        ax.set_ylim(-lim/2, lim)
        ax.set_zlim(-lim/2, lim)
        #ax.axis('equal')
        ax.set_axis_off()
        
        plt.legend()
        
        plt.title(self.xtl.name,fontsize=20,fontweight='bold')
        plt.show()
    
    def plot_layers(self,layer_axis=2,layer_width=0.05,show_labels=False):
        """
        Separate the structure into layers along the chosen axis
        and plot the atoms in each layer in a separate figure.
        """
        
        if layer_axis == 'a': layer_axis = 0
        if layer_axis == 'b': layer_axis = 1
        if layer_axis == 'c': layer_axis = 2
        
        # Choose x,y
        if layer_axis == 0:
            layer_axis_x = 1 # b
            layer_axis_y = 2 # c
        elif layer_axis == 1:
            layer_axis_x = 0 # a
            layer_axis_y = 2 # c
        elif layer_axis == 2:
            layer_axis_x = 0 # a
            layer_axis_y = 1 # b
        
        # Generate layers
        uvw_st,type_st,label_st,occ_st,uiso_st,mxmymz_st = self.xtl.Structure.get()
        vals,uniqeidx,matchidx = fg.unique_vector(uvw_st[:,layer_axis], layer_width)
        # unique_vector takes the first value of each layer, the average is better
        layers = [np.mean(uvw_st[np.asarray(matchidx)==n,layer_axis]) for n in range(len(vals))]
        
        # Generate atomic positions
        uvw,type,label,occ,uiso,mxmymz = self.xtl.Structure.generate_lattice(1,1,1)
        
        # Split atom types, color & radii
        labels,idx,invidx = np.unique(label,return_index=True,return_inverse=True)
        label_colors = plt.cm.gist_rainbow(np.linspace(0,1,len(labels)))
        colors = label_colors[invidx,:]
        sizes = fc.atom_properties(type, 'Radii')
        
        # Get atomic positions
        R = self.xtl.Cell.calculateR(uvw)
        
        # Loop over each layer
        for L,layer in enumerate(layers):
            # Find occupied atoms within the layer
            idx = np.all([np.abs(uvw[:,layer_axis]-layer) < layer_width, occ > 0.2],axis=0) 
            #print L,layer,np.sum(idx)
            layx = R[idx,layer_axis_x]
            layy = R[idx,layer_axis_y]
            laycol = colors[idx,:]
            laysize = sizes[idx]
            
            # Create Figure
            plt.figure(figsize=[12,10])
            plt.scatter(layx,layy,laysize,laycol,marker='o')
            
            # Plot unoccupied atoms
            idx_unocc = np.all([np.abs(uvw[:,layer_axis]-layer) < layer_width, occ <= 0.2],axis=0) 
            layx_unocc = R[idx_unocc,layer_axis_x]
            layy_unocc = R[idx_unocc,layer_axis_y]
            laycol_unocc = colors[idx_unocc,:]
            plt.scatter(layx_unocc,layy_unocc,50,laycol_unocc,marker='+')
            
            # Labels
            if show_labels:
                idx_st = np.abs(uvw_st[:,layer_axis]-layer) < layer_width
                idx_p = np.where(idx_st)[0]
                R_st = self.xtl.Cell.calculateR(uvw_st[idx_st,:])
                lab_st = label_st[idx_st]
                for n in range(len(R_st)):
                    plt.text(R_st[n,layer_axis_x],R_st[n,layer_axis_y],'%2d: %s' %(idx_p[n],lab_st[n]),fontsize=10)
            
            # Create cell box
            box = np.zeros([5,3])
            box[[1,2],layer_axis_x] = 1
            box[[2,3],layer_axis_y] = 1
            bpos = self.xtl.Cell.calculateR(box)
            plt.plot(bpos[:,layer_axis_x],bpos[:,layer_axis_y],'-k') # cell box
            fp.plot_arrow(bpos[[0,1],layer_axis_x],bpos[[0,1],layer_axis_y],col='r',width=4) # a
            fp.plot_arrow(bpos[[0,3],layer_axis_x],bpos[[0,3],layer_axis_y],col='g',width=4) # b
            
            plt.axis('equal')
            plt.xlim(1.1*np.min(bpos[:,layer_axis_x])-1, 1.1*np.max(bpos[:,layer_axis_x])+1)
            plt.ylim(1.1*np.min(bpos[:,layer_axis_y])-1, 1.1*np.max(bpos[:,layer_axis_y])+1)
            #ax.set_axis_off()
            
            # Supercell grid
            if hasattr(self.xtl,'Parent'):
                parentUV = self.xtl.parentUV()
                fp.plot_vector_lines(parentUV[layer_axis_x,:], parentUV[layer_axis_y,:], linewidth=0.5, shade=0.5, color='k')
            #plt.legend()
            
            ttl = '%s\nLayer %2.0f = %5.3f' %(self.xtl.name,L,layer)
            plt.title(ttl,fontsize=20,fontweight='bold')
            plt.show()
    
    def generate_powder(self,q_max=8,peak_width=0.01,background=0):
        """
        Generates a powder pattern and returns the results
          tth,I = generate_powder(energy_kev=8.0,peak_width=0.05,background=0)
            q_max = maximum Q, in A-1
            peak_width = width of convolution, in A-1
            background = average of normal background
          Returns:
            Q = [1000x1] array of wave-vector values
            I = [1000x1] array of intensity values
        """
        
        # Get reflections
        hmax,kmax,lmax  = fc.maxHKL(q_max,self.xtl.Cell.UVstar())
        HKL = fc.genHKL([hmax,-hmax],[kmax,-kmax],[lmax,-lmax])
        HKL = self.xtl.Cell.sort_hkl(HKL) # required for labels
        Qmag = self.xtl.Cell.Qmag(HKL)
        HKL = HKL[Qmag<q_max,:]
        Qmag = Qmag[Qmag<q_max]
        
        # Calculate intensities
        I = self.xtl.Scatter.intensity(HKL)
        
        # create plotting mesh
        pixels = 2000*q_max # reduce this to make convolution faster
        pixel_size = q_max/(1.0*pixels)
        peak_width_pixels = peak_width/(1.0*pixel_size)
        mesh = np.zeros([pixels])
        mesh_q = np.linspace(0,q_max,pixels)
        
        # add reflections to background
        pixel_coord = Qmag/(1.0*q_max)
        pixel_coord = (pixel_coord*pixels).astype(int)
        
        for n in range(1,len(I)):
            mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + I[n]
        
        # Convolve with a gaussian (if >0 or not None)
        if peak_width:
            gauss_x = np.arange(-peak_width_pixels,peak_width_pixels+1) # gaussian width = 2*FWHM
            G = fg.gauss(gauss_x,0,height=1,cen=0,FWHM=peak_width_pixels,bkg=0)
            mesh = np.convolve(mesh,G, mode='same') 
        
        # Add background (if >0 or not None)
        if background:
            bkg = np.random.normal(background,np.sqrt(background), [pixels])
            mesh = mesh+bkg
        return mesh_q,mesh
    
    def simulate_powder(self,energy_kev=8.0,peak_width=0.01,background=0):
        """
        Generates a powder pattern, plots in a new figure with labels
        """
        
        # Get reflections
        angle_max = 180
        q_max = fc.calQmag(angle_max, energy_kev)
        HKL = self.xtl.Cell.all_hkl(energy_kev, angle_max)
        HKL = self.xtl.Cell.sort_hkl(HKL) # required for labels
        Qmag = self.xtl.Cell.Qmag(HKL)
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        #d = self.xtl.Cell.dspace(HKL)
        
        # Calculate intensities
        I = self.xtl.Scatter.intensity(HKL)
        
        # create plotting mesh
        pixels = 2000*q_max # reduce this to make convolution faster
        pixel_size = q_max/(1.0*pixels)
        peak_width_pixels = peak_width/(1.0*pixel_size)
        mesh = np.zeros([int(pixels)])
        mesh_q = np.linspace(0,q_max,pixels)
        mesh_tth = fc.cal2theta(mesh_q, energy_kev)
        #mesh_d = fc.caldspace(mesh_q)
        
        # add reflections to background
        # scipy.interpolate.griddata?
        pixel_coord = Qmag/q_max
        pixel_coord = (pixel_coord*pixels).astype(int)
        
        ref_tth = [0]
        ref_int = [0]
        ref_txt = ['']
        ext_tth = []
        ext_int = []
        ext_txt = []
        for n in range(1,len(I)):
            mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + I[n]
            
            if np.abs(pixel_coord[n]-pixel_coord[n-1]) > peak_width_pixels/2:
                # generate label if not too close to another reflection
                # Note taht at high angle with many close reflections, it may end in 
                # a situation where new peaks are never added
                #print 'A: ',HKL[n,:],tth[n],pixel_coord[n],pixel_coord[n-1],I[n]
                if I[n] > 0.1:
                    ref_tth += [tth[n]]
                    ref_int += [I[n]]
                    ref_txt += ['(%1.0f,%1.0f,%1.0f)' % (HKL[n,0],HKL[n,1],HKL[n,2])]
                else:
                    ext_tth += [tth[n]]
                    ext_int += [I[n]]
                    ext_txt += ['(%1.0f,%1.0f,%1.0f)' % (HKL[n,0],HKL[n,1],HKL[n,2])]
            elif mesh[pixel_coord[n]] > I[n-1] and mesh[pixel_coord[n]] > 0.1:
                # Replace the label if next reflection is larger
                # Note for multiple equivalent reflections, mesh[pixel_coord[n]] will get larger 
                # each time as more intensity is added, so this will always take the last reflection
                #print 'B: ',HKL[n,:],tth[n],pixel_coord[n],pixel_coord[n-1],I[n],mesh[pixel_coord[n]]
                ref_tth[-1] = tth[n]
                ref_int[-1] = mesh[pixel_coord[n]]
                ref_txt[-1] = '(%1.0f,%1.0f,%1.0f)' % (HKL[n,0],HKL[n,1],HKL[n,2])
        
        # Convolve with a gaussian (if >0 or not None)
        if peak_width:
            gauss_x = np.arange(-3*peak_width_pixels,3*peak_width_pixels+1) # gaussian width = 2*FWHM
            G = fg.gauss(gauss_x,0,height=1,cen=0,FWHM=peak_width_pixels,bkg=0)
            mesh = np.convolve(mesh,G, mode='same') 
        
        # Add background (if >0 or not None)
        if background:
            bkg = np.random.normal(background,np.sqrt(background), [pixels])
            mesh = mesh+bkg
        
        # create figure
        plt.figure(figsize=[16,8])
        plt.plot(mesh_tth,mesh,'k-',lw=2)
        
        # Reflection labels
        #print 'Refelctions'
        for n in range(len(ref_tth)):
            #print ref_tth[n],ref_int[n],ref_txt[n]
            plt.text(ref_tth[n],1.01*ref_int[n],ref_txt[n],
                     fontname='Times',fontsize=8,color='b',
                     rotation='vertical',ha='center',va='bottom')
        # Extinction labels
        #print 'Extinctions'
        ymax=plt.ylim()[1]
        for n in range(len(ext_tth)):
            #print ext_tth[n],ext_int[n],ext_txt[n]
            plt.text(ext_tth[n],0.01*ymax,ext_txt[n],
                     fontname='Times',fontsize=8,color='r',
                     rotation='vertical',ha='center',va='bottom')
        
        # Plot labels
        xlab = u'2-Theta [Deg]'
        ylab = u'Intensity'
        ttl = '%s\nE = %1.3f keV' % (self.xtl.name,energy_kev)
        fp.labels(ttl,xlab,ylab)
        plt.show()
    
    def generate_intensity_cut(self,x_axis=[1,0,0],y_axis=[0,1,0],centre=[0,0,0],
                                    q_max=4.0,cut_width=0.05,background=0.0, peak_width=0.05):
        """
        Generate a cut through reciprocal space, returns an array with centred reflections
        Inputs:
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
          background = average background value
          peak_width = reflection width in A-1
        Returns:
          Qx/Qy = [1000x1000] array of coordinates
          plane = [1000x1000] array of plane in reciprocal space 
        
        E.G. hk plane at L=3 for hexagonal system:
            Qx,Qy,plane = xtl.generate_intensity_cut([1,0,0],[0,1,0],[0,0,3])
            plt.figure()
            plt.pcolormesh(Qx,Qy,plane)
            plt.axis('image')
        """
        
        Qx,Qy,HKL = self.xtl.Cell.reciprocal_space_plane(x_axis,y_axis,centre,q_max,cut_width)
        
        # Calculate intensities
        I = self.xtl.Scatter.intensity(HKL)
        
        # create plotting mesh
        pixels = 1000 # reduce this to make convolution faster
        pixel_size = (2.0*q_max)/pixels
        mesh = np.zeros([pixels,pixels])
        mesh_x = np.linspace(-q_max,q_max,pixels)
        X,Y = np.meshgrid(mesh_x,mesh_x)
        
        # add reflections to background
        pixel_i = ((Qx/(2*q_max) + 0.5)*pixels).astype(int)
        pixel_j = ((Qy/(2*q_max) + 0.5)*pixels).astype(int)
        
        mesh[pixel_j,pixel_i] = I
        
        # Convolve with a gaussian (if not None or 0)
        if peak_width:
            peak_width_pixels = peak_width/pixel_size
            gauss_x = np.arange(-2*peak_width_pixels,2*peak_width_pixels+1)
            G = fg.gauss(gauss_x,gauss_x,height=1,cen=0,FWHM=peak_width_pixels,bkg=0)
            mesh = convolve2d(mesh,G, mode='same') # this is the slowest part
        
        # Add background (if not None or 0)
        if background:
            bkg = np.random.normal(background,np.sqrt(background), [pixels,pixels])
            mesh = mesh+bkg
        
        return X,Y,mesh
    
    def simulate_intensity_cut(self,x_axis=[1,0,0],y_axis=[0,1,0],centre=[0,0,0],
                                    q_max=4.0,cut_width=0.05,background=0.0, peak_width=0.05):
        """
        Plot a cut through reciprocal space, visualising the intensity
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
          background = average background value
          peak_width = reflection width in A-1
        
        E.G. hk plot at L=3 for hexagonal system:
            xtl.simulate_intensity_cut([1,0,0],[0,1,0],[0,0,3])
             hhl plot:
            xtl.simulate_intensity_cut([1,1,0],[0,0,1],[0,0,0])
        """
        
        # Determine the directions in cartesian space
        x_cart = fg.norm(self.xtl.Cell.calculateQ(x_axis))
        y_cart = fg.norm(self.xtl.Cell.calculateQ(y_axis))
        z_cart = fg.norm(np.cross( x_cart, y_cart )) # z is perp. to x+y
        y_cart = np.cross(x_cart,z_cart) # make sure y is perp. to x
        c_cart = self.xtl.Cell.calculateQ(centre)
        
        # Correct y-axis for label - original may not have been perp. to x_axis (e.g. hexagonal)
        y_axis = fg.norm(self.xtl.Cell.indexQ(y_cart))
        y_axis = -y_axis/np.min(np.abs(y_axis[np.abs(y_axis)>0])) + 0.0 # +0.0 to remove -0
        
        # Determine orthogonal lattice vectors for plotting lines and labels
        vec_a = x_axis
        vec_c = np.cross(x_axis,y_axis)
        vec_b = fg.norm(np.cross(vec_c,vec_a))
        
        # Generate intensity cut
        X,Y,mesh = self.generate_intensity_cut(x_axis, y_axis, centre, q_max, cut_width, background, peak_width)
        
        # create figure
        plt.figure(figsize=[12,10])
        cmap = plt.get_cmap('hot_r')
        plt.pcolormesh(X,Y,mesh,cmap=cmap)
        plt.axis('image')
        plt.colorbar()
        plt.clim([background-(np.max(mesh)/200),background+(np.max(mesh)/50)])
        
        # Lattice points and vectors within the plot
        Q_vec_a = self.xtl.Cell.calculateQ(vec_a)
        Q_vec_b = self.xtl.Cell.calculateQ(vec_b)
        
        CELL = np.array([2*q_max*x_cart,-2*q_max*y_cart,cut_width*z_cart]) # Plot/mesh unit cell
        
        mesh_vec_a = fg.index_coordinates(Q_vec_a, CELL)*2*q_max # coordinates wrt plot axes
        mesh_vec_b = fg.index_coordinates(Q_vec_b, CELL)*2*q_max
        
        # Vector arrows and lattice point labels
        cen_lab = '(%1.3g,%1.3g,%1.3g)' % (centre[0],centre[1],centre[2])
        vec_a_lab = '(%1.3g,%1.3g,%1.3g)' % (vec_a[0]+centre[0],vec_a[1]+centre[1],vec_a[2]+centre[2])
        vec_b_lab = '(%1.3g,%1.3g,%1.3g)' % (vec_b[0]+centre[0],vec_b[1]+centre[1],vec_b[2]+centre[2])
        
        fp.plot_vector_lines(mesh_vec_a, mesh_vec_b)
        fp.plot_vector_arrows(mesh_vec_a, mesh_vec_b, vec_a_lab, vec_b_lab)
        #fp.plot_vector_lines(Q_vec_a, Q_vec_b)
        #fp.plot_vector_arrows(Q_vec_a, Q_vec_b, vec_a_lab, vec_b_lab)
        plt.text(0-(0.2*q_max),0-(0.1*q_max),cen_lab,fontname='Times',weight='bold',size=18)
        
        # Plot labels
        xlab = u'Q || (%1.3g,%1.3g,%1.3g) [$\AA^{-1}$]' % (x_axis[0],x_axis[1],x_axis[2])
        ylab = u'Q || (%1.3g,%1.3g,%1.3g) [$\AA^{-1}$]' % (y_axis[0],y_axis[1],y_axis[2])
        ttl = '%s\n(%1.3g,%1.3g,%1.3g)' % (self.xtl.name,centre[0],centre[1],centre[2])
        fp.labels(ttl,xlab,ylab)
        plt.show()
    
    def simulate_hk0(self,L=0,**kwargs):
        """
        Plots the hk(L) layer of reciprocal space
         for inputs, see help(xtl.simulate_intensity_cut)
        """
        self.simulate_intensity_cut([1,0,0], [0,1,0], [0,0,L],**kwargs)
    
    def simulate_h0l(self,K=0,**kwargs):
        """
        Plots the h(K)l layer of reciprocal space
         for inputs, see help(xtl.simulate_intensity_cut)
        """
        self.simulate_intensity_cut([1,0,0], [0,0,1], [0,K,0],**kwargs)
    
    def simulate_0kl(self,H=0,**kwargs):
        """
        Plots the (H)kl layer of reciprocal space
         for inputs, see help(xtl.simulate_intensity_cut)
        """
        self.simulate_intensity_cut([0,1,0], [0,0,1], [H,0,0],**kwargs)
    
    def simulate_hhl(self,HmH=0,**kwargs):
        """
        Plots the hhl layer of reciprocal space
         for inputs, see help(xtl.simulate_intensity_cut)
        """
        self.simulate_intensity_cut([1,1,0], [0,0,1], [HmH,-HmH,0],**kwargs)
    
    def simulate_ewald_coverage(self,energy_kev=8.0,sample_normal=[0,0,1],sample_para=[1,0,0],phi=0,chi=0,**kwargs):
        """
        NOT FINISHED
        Plot ewald space coverage within a particular scattering plane
          energy_kev = energy of incident radiation in keV
          sample_normal = direction of scattering plane perp to beam direction
          sample_para = direction of scattering plane || to beam direction
          phi = rotation of lattice about scattering plane verticle
          chi = rotation of lattice perpendicular to scattering plane
        """
        
        # rotate x_axis about y_axis by phi
        # not done yet
        
        # rotate y_axis about x_axis by chi
        # not done yet
        
        # calclate max_q
        q_max = fc.calQmag(180, energy_kev)
        
        # calculate diffractometer angles
        angles = np.arange(0,180,0.1)
        Q1x,Q1y = fc.diffractometer_Q(angles,180,energy_kev) # delta=180
        Q2x,Q2y = fc.diffractometer_Q(angles,angles,energy_kev) # eta=delta
        Q3x,Q3y = fc.diffractometer_Q(0,angles,energy_kev) # eta=0
        
        # Create intensity plot
        self.simulate_intensity_cut(sample_para, sample_normal,[0,0,0],q_max,**kwargs)
        
        # Add diffractometer angles
        plt.plot(Q1x,Q1y,'k',lw=2,label='delta=180')
        plt.plot(Q2x,Q2y,'k',lw=2,label='delta=eta')
        plt.plot(Q3x,Q3y,'k',lw=2,label='eta=0')
        plt.axis([-q_max,q_max,0,q_max])
        
        ttl = '%s\nE = %1.3f keV' % (self.xtl.name,energy_kev)
        fp.labels(ttl)
        plt.show()
    
    def plot_3Dlattice(self,q_max=4.0,x_axis=[1,0,0],y_axis=[0,1,0],centre=[0,0,0],cut_width=0.05):
        """
        Plot lattice points in 3D
        """
        
        # Determine the directions in cartesian space
        x_cart = fg.norm(self.xtl.Cell.calculateQ(x_axis))
        y_cart = fg.norm(self.xtl.Cell.calculateQ(y_axis))
        z_cart = fg.norm(np.cross( x_cart, y_cart )) # z is perp. to x+y
        y_cart = np.cross(x_cart,z_cart) # make sure y is perp. to x
        c_cart = self.xtl.Cell.calculateQ(centre)
        
        # Correct y-axis - original may not have been perp. to x_axis (e.g. hexagonal)
        y_axis = fg.norm(self.xtl.Cell.indexQ(y_cart))
        y_axis = -y_axis/np.min(np.abs(y_axis[np.abs(y_axis)>0])) + 0.0 # +0.0 to remove -0
        
        # Generate lattice of reciprocal space points
        hmax,kmax,lmax  = fc.maxHKL(q_max,self.xtl.Cell.UVstar())
        HKL = fc.genHKL([hmax,-hmax],[kmax,-kmax],[lmax,-lmax])
        HKL = HKL + centre # reflection about central reflection
        Q = self.xtl.Cell.calculateQ(HKL)
        
        # generate box in reciprocal space
        CELL = np.array([2*q_max*x_cart,-2*q_max*y_cart,cut_width*z_cart])
        
        # find reflections within this box
        inplot = fg.isincell(Q,c_cart,CELL)
        fp.newplot3(Q[:,0],Q[:,1],Q[:,2],'bo')
        fp.plot_cell(c_cart,CELL)
        plt.plot(Q[inplot,0],Q[inplot,1],Q[inplot,2],'ro')
        
        fp.labels(self.xtl.name,'Qx','Qy','Qz')
        plt.show()
    
    def quick_intensity_cut(self,x_axis=[1,0,0],y_axis=[0,1,0],centre=[0,0,0], q_max=4.0,cut_width=0.05):
        """
        Plot a cut through reciprocal space, visualising the intensity as different sized markers
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
        
        E.G. hk plot at L=3 for hexagonal system:
            xtl.quick_intensity_cut([1,0,0],[0,1,0],[0,0,3])
             hhl plot:
            xtl.quick_intensity_cut([1,1,0],[0,0,1],[0,0,0])
        """
        
        # Determine the directions in cartesian space
        x_cart = fg.norm(self.xtl.Cell.calculateQ(x_axis))
        y_cart = fg.norm(self.xtl.Cell.calculateQ(y_axis))
        z_cart = fg.norm(np.cross( x_cart, y_cart )) # z is perp. to x+y
        y_cart = np.cross(x_cart,z_cart) # make sure y is perp. to x
        c_cart = self.xtl.Cell.calculateQ(centre)
        
        # Correct y-axis - original may not have been perp. to x_axis (e.g. hexagonal)
        y_axis = fg.norm(self.xtl.Cell.indexQ(y_cart))
        y_axis = -y_axis/np.min(np.abs(y_axis[np.abs(y_axis)>0])) + 0.0 # +0.0 to remove -0
        
        # Determine orthogonal lattice vectors
        vec_a = x_axis
        vec_c = np.cross(x_axis,y_axis)
        vec_b = fg.norm(np.cross(vec_c,vec_a))
        
        # Generate lattice of reciprocal space points
        hmax,kmax,lmax  = fc.maxHKL(q_max,self.xtl.Cell.UVstar())
        HKL = fc.genHKL([hmax,-hmax],[kmax,-kmax],[lmax,-lmax])
        HKL = HKL + centre # reflection about central reflection
        Q = self.xtl.Cell.calculateQ(HKL)
        
        # generate box in reciprocal space
        CELL = np.array([2*q_max*x_cart,-2*q_max*y_cart,cut_width*z_cart])
        
        # find reflections within this box
        inplot = fg.isincell(Q,c_cart,CELL)
        HKL = HKL[inplot,:]
        Q = Q[inplot,:]
        mesh_coord = fg.index_coordinates(Q-c_cart, CELL)
        mesh_Q = mesh_coord*2*q_max
        
        # Calculate intensities
        I = self.xtl.Scatter.intensity(HKL)
        
        # Determine forbidden reflections
        forbidden = mesh_Q[ I < 0.01 ,:]
        
        # create figure
        plt.figure(figsize=[12,10])
        
        # Lattice points and vectors within the plot
        Q_vec_a = self.xtl.Cell.calculateQ(vec_a)
        Q_vec_b = self.xtl.Cell.calculateQ(vec_b)
        mesh_vec_a = fg.index_coordinates(Q_vec_a, CELL)*2*q_max
        mesh_vec_b = fg.index_coordinates(Q_vec_b, CELL)*2*q_max
        
        # Vector arrows and lattice point labels
        cen_lab = '(%1.3g,%1.3g,%1.3g)' % (centre[0],centre[1],centre[2])
        vec_a_lab = '(%1.3g,%1.3g,%1.3g)' % (vec_a[0]+centre[0],vec_a[1]+centre[1],vec_a[2]+centre[2])
        vec_b_lab = '(%1.3g,%1.3g,%1.3g)' % (vec_b[0]+centre[0],vec_b[1]+centre[1],vec_b[2]+centre[2])
        
        plt.text(0.4,0.46,cen_lab,fontname='Times',weight='bold',size=18,transform=plt.gca().transAxes)
        fp.plot_lattice_lines(mesh_Q,mesh_vec_a,mesh_vec_b)
        fp.plot_arrow([0,mesh_vec_a[0,0]],[0,mesh_vec_a[0,1]],arrow_size=40,col='b')
        plt.text(mesh_vec_a[0,0],mesh_vec_a[0,1],vec_a_lab,fontname='Times',weight='bold',size=18)
        fp.plot_arrow([0,mesh_vec_b[0,0]],[0,mesh_vec_b[0,1]],arrow_size=40,col='b')
        plt.text(mesh_vec_b[0,0],mesh_vec_b[0,1],vec_b_lab,fontname='Times',weight='bold',size=18)
        
        # Mark forbidden reflections
        plt.plot(forbidden[:,0],forbidden[:,1],'rx',markersize=12,markeredgewidth=2)
        
        # Plot reflections as circles using logged intensity as the radius
        for n in range(len(I)):
            plt.scatter(mesh_Q[n,0],mesh_Q[n,1],s=50*np.log10(I[n]+1))
        
        plt.axis('image')
        plt.axis([-q_max,q_max,-q_max,q_max])
        
        
        # Plot labels
        xlab = u'Q || (%1.3g,%1.3g,%1.3g) [$\AA^{-1}$]' % (x_axis[0],x_axis[1],x_axis[2])
        ylab = u'Q || (%1.3g,%1.3g,%1.3g) [$\AA^{-1}$]' % (y_axis[0],y_axis[1],y_axis[2])
        ttl = '%s\n(%1.3g,%1.3g,%1.3g)' % (self.xtl.name,centre[0],centre[1],centre[2])
        fp.labels(ttl,xlab,ylab)
        plt.show()
    
    def simulate_azimuth(self,hkl,energy_kev=None,polarisation='sp',F0=1,F1=1,F2=1,azim_zero=[1,0,0]):
        """
        Simulate azimuthal scan of resonant x-ray scattering
            energy_kev = x-ray energy in keV
            polarisation = x-ray polarisation: 'ss',{'sp'},'ps','pp'
            F0/1/2 = Resonance factor Flm
            azim_zero = [h,k,l] vector parallel to the origin of the azimuth
        """
        
        psi = np.arange(-180,180,0.2)
        IXR=self.xtl.Scatter.xray_resonant(hkl, energy_kev, polarisation,F0=F0,F1=F1,F2=F2,azim_zero=azim_zero,PSI=psi)
        
        if polarisation == 'ss':
            pol = '$\sigma$-$\sigma$'
        elif polarisation == 'sp':
            pol = '$\sigma$-$\pi$'
        elif polarisation == 'ps':
            pol = '$\pi$-$\sigma$'
        elif polarisation == 'pp':
            pol = '$\pi$-$\pi$'
        
        ttl = '%s %5.3f keV %s\n(%1.0f,%1.0f,%1.0f) aziref=(%1.0f,%1.0f,%1.0f)'
        ttl = ttl % (self.xtl.name,energy_kev,pol,hkl[0],hkl[1],hkl[2],azim_zero[0],azim_zero[1],azim_zero[2])
        
        plt.figure(figsize=[12,10])
        plt.plot(psi,IXR.T,'-',lw=2)
        plt.xlim([-180,180])
        plt.ylim([0,1.1*np.max(IXR)])
        fp.labels(ttl,'psi [Deg]',pol)
        plt.show()

class Plotting_Superstructure(Plotting):
    """
    Plotting functions for Superstructure class crystal object
    Copies the functions from Plotting, but changes several functions to apply additional actions
    This changes the behaviour of the xtl.Plot functions such as xtl.Plot.simulate_hk0()
    
    generate_intensity_cut & simulate_intensity_cut
     - Refelctions are symmetrised by the parent symmetry, accounting for multiple superlattice domains
     - Generated plots have lines and vectors associated with parent structure
     - Effects functions such as xtl.Plot.simulate_hk0()
    """
    
    def generate_intensity_cut(self,x_axis=[1,0,0],y_axis=[0,1,0],centre=[0,0,0],
                                    q_max=4.0,cut_width=0.05,background=0.0, peak_width=0.05):
        """
        Generate a cut through reciprocal space, returns an array with centred reflections
        Inputs:
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
          background = average background value
          peak_width = reflection width in A-1
        Returns:
          Qx/Qy = [1000x1000] array of coordinates
          plane = [1000x1000] array of plane in reciprocal space 
        
        E.G. hk plane at L=3 for hexagonal system:
            Qx,Qy,plane = xtl.generate_intensity_cut([1,0,0],[0,1,0],[0,0,3])
            plt.figure()
            plt.pcolormesh(Qx,Qy,plane)
            plt.axis('image')
        """
        
        Qx,Qy,HKL = self.xtl.Cell.reciprocal_space_plane(x_axis,y_axis,centre,q_max,cut_width)
        I = self.xtl.Scatter.intensity(HKL)
        
        # Apply Parent symmetry
        pHKL = self.xtl.superhkl2parent(HKL)
        #for n in range(len(pHKL)):
        #    print HKL[n],pHKL[n],I[n]
        HKL,I = self.xtl.Parent.Symmetry.symmetric_intensity(pHKL,I)
        Q=self.xtl.Parent.Cell.calculateQ(HKL)
        Qx,Qy,Qz = Q.T
        
        # create plotting mesh
        pixels = 1000 # reduce this to make convolution faster
        pixel_size = (2.0*q_max)/pixels
        mesh = np.zeros([pixels,pixels])
        mesh_x = np.linspace(-q_max,q_max,pixels)
        X,Y = np.meshgrid(mesh_x,mesh_x)
        
        # add reflections to background
        pixel_i = ((Qx/(2*q_max) + 0.5)*pixels).astype(int)
        pixel_j = ((Qy/(2*q_max) + 0.5)*pixels).astype(int)
        
        # Only take values within the mesh
        in_mesh = np.all([pixel_i>=0, pixel_i<pixels,pixel_j>=0, pixel_j<pixels],axis=0)
        pixel_i = pixel_i[in_mesh]
        pixel_j = pixel_j[in_mesh]
        I = I[in_mesh]
        
        mesh[pixel_j,pixel_i] = I
        
        # Convolve with a gaussian (if not None or 0)
        if peak_width:
            peak_width_pixels = peak_width/pixel_size
            gauss_x = np.arange(-2*peak_width_pixels,2*peak_width_pixels+1)
            G = fg.gauss(gauss_x,gauss_x,height=1,cen=0,FWHM=peak_width_pixels,bkg=0)
            mesh = convolve2d(mesh,G, mode='same') # this is the slowest part
        
        # Add background (if not None or 0)
        if background:
            bkg = np.random.normal(background,np.sqrt(background), [pixels,pixels])
            mesh = mesh+bkg
        
        return X,Y,mesh
    
    def simulate_intensity_cut(self,x_axis=[1,0,0],y_axis=[0,1,0],centre=[0,0,0],
                                    q_max=4.0,cut_width=0.05,background=0.0, peak_width=0.05):
        """
        Plot a cut through reciprocal space, visualising the intensity
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
          background = average background value
          peak_width = reflection width in A-1
        
        E.G. hk plot at L=3 for hexagonal system:
            xtl.simulate_intensity_cut([1,0,0],[0,1,0],[0,0,3])
             hhl plot:
            xtl.simulate_intensity_cut([1,1,0],[0,0,1],[0,0,0])
        """
        
        # Determine the directions in cartesian space
        x_cart,y_cart,z_cart = fc.orthogonal_axes(x_axis, y_axis)
        c_cart = self.xtl.Parent.Cell.calculateQ(centre)
        
        # Correct y-axis for label - original may not have been perp. to x_axis (e.g. hexagonal)
        y_axis = fg.norm(self.xtl.Parent.Cell.indexQ(y_cart))
        y_axis = -y_axis/np.min(np.abs(y_axis[np.abs(y_axis)>0])) + 0.0 # +0.0 to remove -0
        
        # Determine orthogonal lattice vectors for plotting lines and labels
        vec_a = x_axis
        vec_c = np.cross(x_axis,y_axis)
        vec_b = fg.norm(np.cross(vec_c,vec_a))
        
        # Generate intensity cut
        X,Y,mesh = self.generate_intensity_cut(x_axis, y_axis, centre, q_max, cut_width, background, peak_width)
        
        # create figure
        plt.figure(figsize=[12,10])
        cmap = plt.get_cmap('hot_r')
        plt.pcolormesh(X,Y,mesh,cmap=cmap)
        plt.axis('image')
        plt.colorbar()
        plt.clim([background-(np.max(mesh)/200),background+(np.max(mesh)/50)])
        
        # Lattice points and vectors within the plot
        Q_vec_a = self.xtl.Parent.Cell.calculateQ(vec_a)
        Q_vec_b = self.xtl.Parent.Cell.calculateQ(vec_b)
        
        # Vector arrows and lattice point labels
        cen_lab = '(%1.3g,%1.3g,%1.3g)' % (centre[0],centre[1],centre[2])
        vec_a_lab = '(%1.3g,%1.3g,%1.3g)' % (vec_a[0]+centre[0],vec_a[1]+centre[1],vec_a[2]+centre[2])
        vec_b_lab = '(%1.3g,%1.3g,%1.3g)' % (vec_b[0]+centre[0],vec_b[1]+centre[1],vec_b[2]+centre[2])
        
        fp.plot_vector_lines(Q_vec_a, Q_vec_b)
        fp.plot_vector_arrows(Q_vec_a, Q_vec_b, vec_a_lab, vec_b_lab)
        plt.text(0-(0.2*q_max),0-(0.1*q_max),cen_lab,fontname='Times',weight='bold',size=18)
        
        # Plot labels
        xlab = u'Q || (%1.3g,%1.3g,%1.3g) [$\AA^{-1}$]' % (x_axis[0],x_axis[1],x_axis[2])
        ylab = u'Q || (%1.3g,%1.3g,%1.3g) [$\AA^{-1}$]' % (y_axis[0],y_axis[1],y_axis[2])
        ttl = '%s\n(%1.3g,%1.3g,%1.3g)' % (self.xtl.name,centre[0],centre[1],centre[2])
        fp.labels(ttl,xlab,ylab)
        plt.show()

class Multi_Plotting:
    """
    Plotting functions for the Multi-Crystal Object
    """
    def simulate_powder(self,energy_kev=8.0,peak_width=0.05,background=0):
        """
        Generates a powder pattern for multiple phases
        """
        
        # Get reflections
        angle_max = 180
        q_max = fc.calQmag(angle_max, energy_kev)
        
        plt.figure(figsize=[16,8])
        colours = iter(['b','g','r','c','m','y','k'])
        
        for xtl in self.crystal_list:
            HKL = xtl.Cell.all_hkl(energy_kev, angle_max)
            HKL = xtl.Cell.sort_hkl(HKL) # required for labels
            Qmag = xtl.Cell.Qmag(HKL)
            tth = xtl.Cell.tth(HKL,energy_kev)
            I = xtl.Scatter.intensity(HKL)
            col = next(colours)
            
            # create plotting mesh
            pixels = 2000*q_max # reduce this to make convolution faster
            pixel_size = q_max/pixels
            peak_width_pixels = peak_width/pixel_size
            mesh = np.zeros([pixels])
            mesh_q = np.linspace(0,q_max,pixels)
            mesh_tth = fc.cal2theta(mesh_q, energy_kev)
            #mesh_d = fc.caldspace(mesh_q)
            
            # add reflections to background
            # scipy.interpolate.griddata?
            pixel_coord = Qmag/q_max
            pixel_coord = (pixel_coord*pixels).astype(int)
            
            ref_tth = [0]
            ref_int = [0]
            ref_txt = ['']
            ext_tth = []
            ext_int = []
            ext_txt = []
            for n in range(1,len(I)):
                mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + I[n]
                
                # generate label if not too close to another reflection
                if np.abs(pixel_coord[n]-pixel_coord[n-1]) > peak_width_pixels/2:
                    #print 'A: ',HKL[n,:],tth[n],pixel_coord[n],pixel_coord[n-1],I[n]
                    if I[n] > 0.1:
                        ref_tth += [tth[n]]
                        ref_int += [I[n]]
                        ref_txt += ['(%1.0f,%1.0f,%1.0f)' % (HKL[n,0],HKL[n,1],HKL[n,2])]
                    else:
                        ext_tth += [tth[n]]
                        ext_int += [I[n]]
                        ext_txt += ['(%1.0f,%1.0f,%1.0f)' % (HKL[n,0],HKL[n,1],HKL[n,2])]
                elif mesh[pixel_coord[n]] > I[n-1] and mesh[pixel_coord[n]] > 0.1:
                    #print 'B: ',HKL[n,:],tth[n],pixel_coord[n],pixel_coord[n-1],I[n],mesh[pixel_coord[n]]
                    ref_tth[-1] = tth[n]
                    ref_int[-1] = mesh[pixel_coord[n]]
                    ref_txt[-1] = '(%1.0f,%1.0f,%1.0f)' % (HKL[n,0],HKL[n,1],HKL[n,2])
            
            # Convolve with a gaussian
            if peak_width:
                gauss_x = np.arange(-peak_width_pixels,peak_width_pixels+1) # gaussian width = 2*FWHM
                G = fg.gauss(gauss_x,0,height=1,cen=0,FWHM=peak_width_pixels,bkg=0)
                mesh = np.convolve(mesh,G, mode='same') 
            
            # Add background
            if background:
                bkg = np.random.normal(background,np.sqrt(background), [pixels])
                mesh = mesh+bkg
            
            # create figure
            plt.plot(mesh_tth,mesh,'-',lw=2, label=xtl.name,c=col)
            
            # Reflection labels
            #print 'Refelctions'
            for n in range(len(ref_tth)):
                #print ref_tth[n],ref_int[n],ref_txt[n]
                plt.text(ref_tth[n],1.01*ref_int[n],ref_txt[n],
                         fontname='Times',fontsize=8,color=col,
                         rotation='vertical',ha='center',va='bottom')
            # Extinction labels
            #print 'Extinctions'
            ymax=plt.ylim()[1]
            for n in range(len(ext_tth)):
                #print ext_tth[n],ext_int[n],ext_txt[n]
                plt.text(ext_tth[n],0.01*ymax,ext_txt[n],
                         fontname='Times',fontsize=8,color=col,fontweight='bold',
                         rotation='vertical',ha='center',va='bottom')
            
        # Plot labels
        xlab = u'2-Theta [Deg]'
        ylab = u'Intensity'
        ttl = 'E = %1.3f keV' % (energy_kev)
        plt.legend(loc=0,fontsize=18,frameon=False)
        fp.labels(ttl,xlab,ylab)
        plt.show()
    
    def simulate_intensity_cut(self,x_axis_crystal=[[1,0,0]],y_axis_crystal=[[0,1,0]],centre=[0,0,0],
                                    q_max=4.0,cut_width=0.05,background=0.0, peak_width=0.05):
        """
        Plot a cut through reciprocal space, visualising the intensity
          x_axis_crystal = list of crytal x-directions in (hkl), for each crystal
          y_axis_crystal = list of crytal y-directions in (hkl), for each crystal
          centre = centre of the plot, in units of the reciprocal lattice (hkl) of crystal 1
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
          background = average background value
          peak_width = reflection width in A-1
        
        E.G. hk plot at L=3 for hexagonal system (xtl1) with xtl2 oriented with the (110) alond xtl1's (100):
            xtls = multi_crystal([xtl1,xtl2])
            xtl.simulate_intensity_cut([[1,0,0],[1,1,0]],[[0,1,0],[0,0,1]],[0,0,3])
        """
        
        # create plotting mesh
        pixels = 1000 # reduce this to make convolution faster
        pixel_size = (2.0*q_max)/pixels
        mesh = np.zeros([pixels,pixels])
        mesh_x = np.linspace(-q_max,q_max,pixels)
        X,Y = np.meshgrid(mesh_x,mesh_x)
        colours = iter(['b','g','r','c','m','y','k'])
        
        # Determine centre point
        c_cart = self.crystal_list[0].Cell.calculateQ(centre)
        
        # create figure
        plt.figure(figsize=[12,10])
        cmap = plt.get_cmap('hot_r')
        pp = plt.pcolormesh(X,Y,mesh.T,cmap=cmap)
        plt.axis('image')
        plt.colorbar()
        #plt.clim([background-(np.mean(I)/20),background+(np.mean(I)/5)])
        
        n = 0
        for xtl,x_axis,y_axis in zip(self.crystal_list,x_axis_crystal,y_axis_crystal):
            # Determine the directions in cartesian space
            x_cart = fg.norm(xtl.Cell.calculateQ(x_axis))
            y_cart = fg.norm(xtl.Cell.calculateQ(y_axis))
            z_cart = fg.norm(np.cross( x_cart, y_cart )) # z is perp. to x+y
            y_cart = np.cross(x_cart,z_cart) # make sure y is perp. to x
            
            # Correct y-axis - original may not have been perp. to x_axis (e.g. hexagonal)
            y_axis = fg.norm(xtl.Cell.indexQ(y_cart))
            y_axis = -y_axis/np.min(np.abs(y_axis[np.abs(y_axis)>0])) + 0.0 # +0.0 to remove -0
            
            # Determine orthogonal lattice vectors
            vec_a = x_axis
            vec_c = np.cross(x_axis,y_axis)
            vec_b = fg.norm(np.cross(vec_c,vec_a))
            xtl_centre = np.round(xtl.Cell.indexQ(c_cart))[0]
            
            # Generate lattice of reciprocal space points
            hmax,kmax,lmax  = fc.maxHKL(q_max,xtl.Cell.UVstar())
            HKL = fc.genHKL([hmax,-hmax],[kmax,-kmax],[lmax,-lmax])
            HKL = HKL + xtl_centre # reflection about central reflection
            Q = xtl.Cell.calculateQ(HKL)
            
            # generate box in reciprocal space
            CELL = np.array([2*q_max*x_cart,-2*q_max*y_cart,cut_width*z_cart])
            
            # find reflections within this box
            inplot = fg.isincell(Q,c_cart,CELL)
            HKL = HKL[inplot,:]
            Q = Q[inplot,:]
            
            # Calculate intensities
            I = xtl.Scatter.intensity(HKL)
            
            # add reflections to background
            mesh_coord = fg.index_coordinates(Q-c_cart, CELL)
            pixel_coord = mesh_coord + 0.5
            pixel_coord = (pixel_coord*pixels).astype(int)
            
            #for n in range(len(I)):
            #    print n,HKL[n,:],Q[n,:],mesh_coord[n,:],pixel_coord[n,:]
            
            mesh[pixel_coord[:,0],pixel_coord[:,1]] = mesh[pixel_coord[:,0],pixel_coord[:,1]] + I
            
            # Lattice points and vectors within the plot
            Q_vec_a = xtl.Cell.calculateQ(vec_a)
            Q_vec_b = xtl.Cell.calculateQ(vec_b)
            mesh_Q = mesh_coord*2*q_max
            mesh_vec_a = fg.index_coordinates(Q_vec_a, CELL)*2*q_max
            mesh_vec_b = fg.index_coordinates(Q_vec_b, CELL)*2*q_max
            
            col = next(colours)
            if n == 0:
                # 1st crystal only - plot lines and centre
                cen_lab = '(%1.3g,%1.3g,%1.3g)$_1$' % (centre[0],centre[1],centre[2])
                plt.text(0-(0.2*q_max),0-(0.1*q_max),cen_lab,fontname='Times',weight='bold',size=18,color=col)
                fp.plot_lattice_lines(mesh_Q,mesh_vec_a,mesh_vec_b)
                plt.clim([background-(np.mean(I)/20),background+(np.mean(I)/5)])
            else:
                plt.plot(mesh_Q[:,0],mesh_Q[:,1],'o',label=xtl.name,c=col,markerfacecolor='none')
            n += 1
            
            # Vector arrows and lattice point labels
            vec_a_lab = '(%1.3g,%1.3g,%1.3g)$_%d$' % (vec_a[0]+xtl_centre[0],vec_a[1]+xtl_centre[1],vec_a[2]+xtl_centre[2],n)
            vec_b_lab = '(%1.3g,%1.3g,%1.3g)$_%d$' % (vec_b[0]+xtl_centre[0],vec_b[1]+xtl_centre[1],vec_b[2]+xtl_centre[2],n)
            
            fp.plot_arrow([0,mesh_vec_a[0,0]],[0,mesh_vec_a[0,1]],arrow_size=40,col=col)
            plt.text(mesh_vec_a[0,0],mesh_vec_a[0,1],vec_a_lab,fontname='Times',weight='bold',size=18,color=col)
            fp.plot_arrow([0,mesh_vec_b[0,0]],[0,mesh_vec_b[0,1]],arrow_size=40,col=col)
            plt.text(mesh_vec_b[0,0],mesh_vec_b[0,1],vec_b_lab,fontname='Times',weight='bold',size=18,color=col)
        
        # Convolve with a gaussian
        peak_width_pixels = peak_width/pixel_size
        gauss_x = np.arange(-2*peak_width_pixels,2*peak_width_pixels+1)
        G = fg.gauss(gauss_x,gauss_x,height=1,cen=0,FWHM=peak_width_pixels,bkg=0)
        mesh = convolve2d(mesh,G, mode='same') # this is the slowest part
        
        # Add background
        if background > 0:
            bkg = np.random.normal(background,np.sqrt(background), [pixels,pixels])
            mesh = mesh+bkg
        
        # Update plot
        mesh = mesh[:-1,:-1].T # a very odd bug noted by lumbric: https://stackoverflow.com/questions/18797175/animation-with-pcolormesh-routine-in-matplotlib-how-do-i-initialize-the-data
        pp.set_array(mesh.ravel())
        
        # Plot labels
        xlab = u'Qx [$\AA^{-1}$]' 
        ylab = u'Qy [$\AA^{-1}$]' 
        #ttl = '%s\n(%1.3g,%1.3g,%1.3g)' % (self.name,centre[0],centre[1],centre[2])
        fp.labels(None,xlab,ylab)
        plt.show()
    
    def quick_intensity_cut(self,x_axis_crystal=[[1,0,0]],y_axis_crystal=[[0,1,0]],centre=[0,0,0], q_max=4.0,cut_width=0.05):
        """
        Plot a cut through reciprocal space, visualising the intensity as different sized markers
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
        
        E.G. hk plot at L=3 for hexagonal system:
            xtl.quick_intensity_cut([1,0,0],[0,1,0],[0,0,3])
             hhl plot:
            xtl.quick_intensity_cut([1,1,0],[0,0,1],[0,0,0])
        """
        
        # create figure
        plt.figure(figsize=[12,10])
        plt.axis('image')
        plt.axis([-q_max,q_max,-q_max,q_max])
        colours = iter(['b','g','r','c','m','y','k'])
        # Determine centre point
        c_cart = self.crystal_list[0].Cell.calculateQ(centre)
        
        legend_entries = []
        n=0
        for xtl,x_axis,y_axis in zip(self.crystal_list,x_axis_crystal,y_axis_crystal):
            # Determine the directions in cartesian space
            x_cart = fg.norm(xtl.Cell.calculateQ(x_axis))
            y_cart = fg.norm(xtl.Cell.calculateQ(y_axis))
            z_cart = fg.norm(np.cross( x_cart, y_cart )) # z is perp. to x+y
            y_cart = np.cross(x_cart,z_cart) # make sure y is perp. to x
            c_cart = xtl.Cell.calculateQ(centre)
            
            # Correct y-axis - original may not have been perp. to x_axis (e.g. hexagonal)
            y_axis = fg.norm(xtl.Cell.indexQ(y_cart))
            y_axis = -y_axis/np.min(np.abs(y_axis[np.abs(y_axis)>0])) + 0.0 # +0.0 to remove -0
            
            # Determine orthogonal lattice vectors
            vec_a = x_axis
            vec_c = np.cross(x_axis,y_axis)
            vec_b = fg.norm(np.cross(vec_c,vec_a))
            xtl_centre = np.round(xtl.Cell.indexQ(c_cart))[0]
            
            # Generate lattice of reciprocal space points
            hmax,kmax,lmax  = fc.maxHKL(q_max,xtl.Cell.UVstar())
            HKL = fc.genHKL([hmax,-hmax],[kmax,-kmax],[lmax,-lmax])
            HKL = HKL + xtl_centre # reflection about central reflection
            Q = xtl.Cell.calculateQ(HKL)
            
            # generate box in reciprocal space
            CELL = np.array([2*q_max*x_cart,-2*q_max*y_cart,cut_width*z_cart])
            
            # find reflections within this box
            inplot = fg.isincell(Q,c_cart,CELL)
            HKL = HKL[inplot,:]
            Q = Q[inplot,:]
            mesh_coord = fg.index_coordinates(Q-c_cart, CELL)
            mesh_Q = mesh_coord*2*q_max
            
            # Calculate intensities
            I = xtl.Scatter.intensity(HKL)
            
            # Determine forbidden reflections
            forbidden = mesh_Q[ I < 0.01 ,:]
            
            # Lattice points and vectors within the plot
            Q_vec_a = xtl.Cell.calculateQ(vec_a)
            Q_vec_b = xtl.Cell.calculateQ(vec_b)
            mesh_vec_a = fg.index_coordinates(Q_vec_a, CELL)*2*q_max
            mesh_vec_b = fg.index_coordinates(Q_vec_b, CELL)*2*q_max
            
            col = next(colours)
            if n == 0:
                # 1st crystal only - plot lines and centre
                cen_lab = '(%1.3g,%1.3g,%1.3g)$_1$' % (centre[0],centre[1],centre[2])
                plt.text(0-(0.2*q_max),0-(0.1*q_max),cen_lab,fontname='Times',weight='bold',size=18,color=col)
                fp.plot_lattice_lines(mesh_Q,mesh_vec_a,mesh_vec_b)
            n += 1
            
            # Mark forbidden reflections
            plt.plot(forbidden[:,0],forbidden[:,1],'x',markersize=12,markeredgewidth=2,color=col)
            
            pt = plt.scatter([0],[0],s=50,c=col)
            legend_entries += [pt]
            
            # Plot reflections as circles using logged intensity as the radius
            for n in range(len(I)):
                plt.scatter(mesh_Q[n,0],mesh_Q[n,1],s=50*np.log10(I[n]+1),c=col)
        
        # Plot labels
        xlab = u'Qx [$\AA^{-1}$]'
        ylab = u'Qy [$\AA^{-1}$]'
        fp.labels(None,xlab,ylab)
        names = [xtl.name for xtl in self.crystal_list]
        plt.legend(legend_entries,names,frameon=True,fontsize=16)
        plt.show()