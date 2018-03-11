# -*- coding: utf-8 -*-
"""
Module: Handy plotting functions "functions_plotting.py"

Contains various plotting shortcuts using matplotlib.pyplot

By Dan Porter, PhD
Diamond
2018

Usage: 
    - Run this file in an interactive console
    OR
    - from Dans_Diffraction import functions_general as fg
    

Version 1.1
Last updated: 05/03/18

Version History:
06/01/18 1.0    Program created from DansGeneralProgs.py V2.3
05/03/18 1.1    Removed plt.show from arrow functions

@author: DGPorter
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from Dans_Diffraction import functions_general as fg
from Dans_Diffraction import functions_crystallography as fc

'----------------------------Plot manipulation--------------------------'
def labels(ttl=None,xvar=None,yvar=None,zvar=None,size='Normal'):
    """
    Add good labels to current plot 
     labels(title,xlabel,ylabel,zlabel,size)
     
     size = 'normal', 'big'
    """
    
    if size.lower() == 'big':
        tik = 30
        tit = 32
        lab = 35
    else:
        # Normal
        tik = 18
        tit = 20
        lab = 22
    
    
    
    plt.xticks(fontsize=tik)
    plt.yticks(fontsize=tik)
    
    if ttl != None:
        plt.gca().set_title(ttl,fontsize=tit,fontweight='bold')
    
    if xvar != None:
        plt.gca().set_xlabel(xvar,fontsize=lab)
    
    if yvar != None:
        plt.gca().set_ylabel(yvar,fontsize=lab)
    
    if zvar != None:
        # Don't think this works, use ax.set_zaxis
        plt.gca().set_zlabel(zvar,fontsize=lab)
    return

def saveplot(name,dpi=None):
    """
    Saves current figure as a png in the home directory
    E.G.
    ---select figure to save by clicking on it---
    saveplot('test')
    """
    
    if type(name) is int:
        name = str(aa)
    
    gcf = plt.gcf()
    def_directory = os.path.expanduser('~')
    savefile = os.path.join(def_directory, '{}.png'.format(saveable(name)))
    gcf.savefig(savefile,dpi=dpi)
    print( 'Saved Figure {} as {}'.format(gcf.number,savefile) )

def newplot(*args,**kwargs):
    """
    Shortcut to creating a simple plot
    """
    
    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2
    
    plt.figure(figsize=[12,12])
    plt.plot(*args,**kwargs)
    plt.show()

def newplot3(*args,**kwargs):
    """
    Shortcut to creating a simple 3D plot
    Automatically tiles 1 dimensional x and y arrays to match 2D z array,
    assuming z.shape = (len(x),len(y))
    
    E.G.
      newplot3([1,2,3,4],[9,8,7],[[2,4,6],[8,10,12],[14,16,18],[20,22,24]],'-o')
    """
    
    
    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2
    
    fig = plt.figure(figsize=[12,12])
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.asarray(args[0], dtype=np.float)
    y = np.asarray(args[1], dtype=np.float)
    z = np.asarray(args[2], dtype=np.float)
    
    if z.ndim == 2:
        if x.ndim < 2:
            x = np.tile(x,z.shape[1]).reshape(z.T.shape).T
        if y.ndim < 2:
            y = np.tile(y,z.shape[0]).reshape(z.shape)
        
        # Plot each array independently
        for n in range(len(z)):
            ax.plot(x[n],y[n],z[n],*args[3:],**kwargs)
    else:
        ax.plot(*args,**kwargs)
    plt.show()

def sliderplot(YY,X=None,slidervals=None,*args,**kwargs):
    """
    Shortcut to creating a simple 2D plot with a slider to go through a third dimension
    YY = [nxm]: y axis data (initially plots Y[0,:])
     X = [n] or [nxm]:  x axis data (can be 1D or 2D, either same length or shape as Y)
     slidervals = None or [m]: Values to give in the slider
    
    E.G.
      sliderplot([1,2,3],[[2,4,6],[8,10,12],[14,16,18],[20,22,24]],slidervals=[3,6,9,12])
    """
    
    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2
    
    fig = plt.figure(figsize=[12,12])
    
    X = np.asarray(X, dtype=np.float)
    Y = np.asarray(YY, dtype=np.float)
    if slidervals is None:
        slidervals = range(Y.shape[0])
    slidervals = np.asarray(slidervals, dtype=np.float)
    
    if X.ndim < 2:
        X = np.tile(X,Y.shape[0]).reshape(Y.shape)
    
    plotline, = plt.plot(X[0,:],Y[0,:],*args,**kwargs)
    plt.axis([X.min(),X.max(),Y.min(),Y.max()])
    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    
    " Create slider on plot"
    axsldr = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='lightgoldenrodyellow')
    
    sldr = plt.Slider(axsldr, '', 0, len(slidervals)-1)
    txt = axsldr.set_xlabel('{} [{}]'.format(slidervals[0],0),fontsize=18 )
    
    plt.sca(ax)
    
    " Slider update function"
    def update(val):
        "Update function for pilatus image"
        pno = int(np.floor(sldr.val))
        plotline.set_xdata(X[pno,:])
        plotline.set_ydata(Y[pno,:])
        txt.set_text('{} [{}]'.format(slidervals[pno], pno))
        plt.draw()
        plt.gcf().canvas.draw()
        #fig1.canvas.draw()
    sldr.on_changed(update)
    plt.show()

def sliderplot2D(ZZZ,XX=None,YY=None,slidervals=None,*args,**kwargs):
    """
    Shortcut to creating an image plot with a slider to go through a third dimension
    ZZZ = [nxmxo]: z axis data
     XX = [nxm] or [n]:  x axis data
     YY = [nxm] or [m]: y axis data 
     slidervals = None or [o]: Values to give in the slider
    
    if XX and/or YY have a single dimension, the 2D values are generated via meshgrid 
    
    E.G.
      sliderplot([1,2,3],[[2,4,6],[8,10,12],[14,16,18],[20,22,24]],slidervals=[3,6,9,12])
    """
    
    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2
    
    fig = plt.figure(figsize=[12,12])
    
    ZZZ = np.asarray(ZZZ, dtype=np.float)
    
    if slidervals is None:
        slidervals = range(ZZZ.shape[2])
    slidervals = np.asarray(slidervals, dtype=np.float)
    
    if XX is None:
        XX = range(ZZZ.shape[1])
    if YY is None:
        YY = range(ZZZ.shape[0])
    XX = np.asarray(XX, dtype=np.float)
    YY = np.asarray(YY, dtype=np.float)
    if XX.ndim < 2:
        XX,YY = np.meshgrid(XX,YY)
    
    p = plt.pcolormesh(XX,YY,ZZZ[:,:,0])
    #p.set_clim(cax)
    
    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.autoscale(tight=True)
    
    " Create slider on plot"
    axsldr = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='lightgoldenrodyellow')
    
    sldr = plt.Slider(axsldr, '', 0, len(slidervals)-1)
    txt = axsldr.set_xlabel('{} [{}]'.format(slidervals[0],0),fontsize=18 )
    
    plt.sca(ax)
    
    " Slider update function"
    def update(val):
        "Update function for pilatus image"
        pno = int(np.round(sldr.val))
        p.set_array(ZZZ[:-1,:-1,pno].ravel())
        txt.set_text('{} [{}]'.format(slidervals[pno], pno))
        plt.draw()
        plt.gcf().canvas.draw()
        #fig1.canvas.draw()
    sldr.on_changed(update)
    plt.show()

def plot_cell(cell_centre=[0,0,0],CELL=np.eye(3)):
    """
    Plot a box defined by a unit cell on the current plot
    """
    
    uvw = np.array([[0.,0,0],[1,0,0],[1,0,1],[1,1,1],[1,1,0],[0,1,0],[0,1,1],\
                    [0,0,1],[1,0,1],[1,0,0],[1,1,0],[1,1,1],[0,1,1],[0,1,0],[0,0,0],[0,0,1]])
    uvw = uvw - 0.5 # plot around box centre
    bpos = np.dot(uvw,CELL)
    bpos = bpos + cell_centre
    plt.plot(bpos[:,0],bpos[:,1],bpos[:,2],c='k') # cell box

def plot_arrow(x,y,z=None, col='r', width=2, arrow_size=40):
    """
    Plot arrow in 2D or 3D on current axes
    Usage 2D:
      plot_arrow([xi,xf],[yi,yf])
    Usage 3D:
      plot_arrow([xi,xf],[yi,yf],[zi,zf])
    
    Options:
      width = line width (Def. = 2)
      arrow_size = size of arrow head (Def. = 40)
      col = arrow color (Deg. = red)
    """
    
    # 2D Arrow
    if z is None or not hasattr(plt.gca(),'get_zlim'):
        x0 = x[0]
        y0 = y[0]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        plt.arrow(x0,y0,dx,dy, width=arrow_size/4000.0, color = col,length_includes_head=True)
        #V = FancyArrowPatch(x,y, mutation_scale=arrow_size, lw=width, arrowstyle="-|>", color=col)
        #plt.gca().add_artist(V)
        return
    
    # 3D Arrow
    V = Arrow3D(x,y,z, mutation_scale=arrow_size, lw=width, arrowstyle="-|>", color=col)
    plt.gca().add_artist(V)

class Arrow3D(FancyArrowPatch):
    """
    FancyArrow3D patch for 3D arrows, by CT Zhu
     http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    Useage: 
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot([0,1],[0,0],[0,0],'k-')
      ax.plot([0,0],[0,1],[0,0],'k-')
      ax.plot([0,0],[0,0],[0,1],'k-')
      v = Arrow3D([0,1],[0,1],[0,1], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
      ax.add_artist(v)
      plt.show()
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        if 'arrowstyle' not in kwargs.keys():
            kwargs['arrowstyle'] = "-|>"
        if 'mutation_scale' not in kwargs.keys():
            kwargs['mutation_scale'] = 20
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

'----------------------- Crystal Plotting Programs----------------------'

def vecplot(UV,mode='hk0',linewidth=1,alpha=0.2,color='k'):
    """
    Plot grid of a,b vectors
    """
    
    if mode == 'h0l':
        # h0l
        UV = np.dot(np.array([[1,0,0],[0,0,1],[0,1,0]]),fg.rot3D(UV,gamma=-90))
    elif mode == '0kl':
        # 0kl
        UV = np.dot(np.array([[0,1,0],[0,0,1],[1,0,0]]),UV)
    elif mode == 'hhl':
        # hhl ***untested
        UV = np.dot(np.array([[1,1,0],[0,0,1],[0,1,0]]),UV)

    
    # Get current axis size
    ax = plt.gca()
    axsize = ax.axis()
    max_ax = [axsize[1],axsize[3],0]
    
    # Generate HKL points within axis
    max_hkl = np.max(np.ceil(np.abs(fc.indx(max_ax,UV))))
    # Generate all hkl values in this range
    HKL = fc.genHKL(max_hkl+1,negative=True)
    # lattice positions
    Q = np.dot(HKL,UV)
    
    # Angle between vectors
    A = fg.ang(UV[0],UV[1])
    
    # At each lattice point, draw the unit vectors
    for n in range(len(Q)):
        lp = Q[n,:]
        uv1 = lp + UV[0,:]
        uv2 = lp + UV[1,:]
        ax.plot([lp[0],uv1[0]],[lp[1],uv1[1]],'-',linewidth=linewidth,alpha=alpha,color=color)
        ax.plot([lp[0],uv2[0]],[lp[1],uv2[1]],'-',linewidth=linewidth,alpha=alpha,color=color)
        if abs(A-np.pi/3)<0.01 or abs(A-2*np.pi/3)<0.01:
            uv3 = lp - UV[0,:] + UV[1,:]
            ax.plot([lp[0],uv3[0]],[lp[1],uv3[1]],'-',linewidth=linewidth,alpha=alpha,color=color)
    ax.axis(axsize)

def UV_arrows(UV):
    """
    Plot arrows with a*,b* on current figure
    """
    # Get current axis size
    ax = plt.gca()
    axsize = ax.axis()
    asty = dict(arrowstyle="->")
    plt.annotate("", xy=(UV[0,0],UV[0,1]), xytext=(0.0,0.0),arrowprops=asty)
    plt.annotate("", xy=(UV[1,0],UV[1,1]), xytext=(0.0,0.0),arrowprops=asty)
    plt.annotate("a", (0.1 + UV[0,0],UV[0,1]-0.2))
    plt.annotate("b", (UV[1,0]-0.2,0.1 + UV[1,1]))
    ax.axis(axsize)

def plot_lattice_points2D(Q,markersize=12,color='b',marker='o'):
    "Add lines defining the reciprocal lattice to the current plot"
    
    ax = plt.gca()
    axsize = ax.axis()
    
    ax.plot(Q[:,0],Q[:,1],markersize=markersize,color=color,marker=marker)
    ax.axis(axsize)

def plot_lattice_lines(Q,vec_a=[1,0,0],vec_b=[0,1,0],linewidth=0.5,shade=1.0,color='k'):
    "Add lines defining the reciprocal lattice to the current plot"
    
    ax = plt.gca()
    axsize = ax.axis()
    
    # vectors
    A = np.asarray(vec_a,dtype=float).reshape([3])
    B = np.asarray(vec_b,dtype=float).reshape([3])
    
    # Angle between vectors
    angle = fg.ang(A,B)
    
    # At each lattice point, draw the unit vectors
    for n in range(len(Q)):
        lp = Q[n,:]
        uv1_1 = lp - A
        uv1_2 = lp + A
        uv2_1 = lp - B
        uv2_2 = lp + B
        
        #print n,lp
        #print uv1_1,uv1_2
        #print uv2_1,uv2_2
        
        ax.plot([uv1_1[0],uv1_2[0]],[uv1_1[1],uv1_2[1]],'-',linewidth=linewidth,alpha=shade,color=color)
        ax.plot([uv2_1[0],uv2_2[0]],[uv2_1[1],uv2_2[1]],'-',linewidth=linewidth,alpha=shade,color=color)
        if abs(angle-np.pi/3)<0.01 or abs(angle-2*np.pi/3)<0.01:
            uv3_1 = lp + A - B
            uv3_2 = lp - A + B
            ax.plot([uv3_1[0],uv3_2[0]],[uv3_1[1],uv3_2[1]],'-',linewidth=linewidth,alpha=shade,color=color)
    ax.axis(axsize)

def axis_lattice_points(vec_a=[1,0,0],vec_b=[0,1,0],axis=[-4,4,-4,4]):
    """
    Generate a 2D lattice of points generated by 2 vectors within a 2D axis
    """
    
    # Vectors
    A = np.asarray(vec_a,dtype=float).reshape([3])
    B = np.asarray(vec_b,dtype=float).reshape([3])
    # Generate a 3D cell to make use of indx function
    U = np.array([A,B,[0,0,1]])
    corners = [[axis[1],axis[2],0], 
               [axis[1],axis[3],0], 
               [axis[0],axis[2],0], 
               [axis[0],axis[3],0]]
    # Determine the coefficients required to generate lattice points of the 2 vectors at 
    # all 4 corners of the axis
    idx = fc.indx(corners,U)
    min_x = np.floor(np.min(idx[:,0]))
    max_x = np.ceil(np.max(idx[:,0]))
    min_y = np.floor(np.min(idx[:,1]))
    max_y = np.ceil(np.max(idx[:,1]))
    hkl = fc.genHKL([min_x,max_x], [min_y,max_y], 0)
    Q = np.dot(hkl,U)
    return Q

def plot_vector_lines(vec_a=[1,0,0],vec_b=[0,1,0],linewidth=0.5,shade=1.0,color='k'):
    """
    Add lines defining the reciprocal lattice to the current plot
    """
    
    ax = plt.gca()
    axsize = ax.axis()
    
    # vectors
    A = np.asarray(vec_a,dtype=float).reshape([3])
    B = np.asarray(vec_b,dtype=float).reshape([3])
    
    # Angle between vectors
    angle = fg.ang(A,B)
    
    # Assume the origin is at (0,0)
    # Determine the lattice points possible in this plot using these vectors
    Q = axis_lattice_points(vec_a,vec_b,axsize)
    
    # At each lattice point, draw the unit vectors
    for n in range(len(Q)):
        lp = Q[n,:]
        uv1_1 = lp - A
        uv1_2 = lp + A
        uv2_1 = lp - B
        uv2_2 = lp + B
        
        #print n,lp
        #print uv1_1,uv1_2
        #print uv2_1,uv2_2
        
        ax.plot([uv1_1[0],uv1_2[0]],[uv1_1[1],uv1_2[1]],'-',linewidth=linewidth,alpha=shade,color=color)
        ax.plot([uv2_1[0],uv2_2[0]],[uv2_1[1],uv2_2[1]],'-',linewidth=linewidth,alpha=shade,color=color)
        if abs(angle-np.pi/3)<0.01: # 60Deg
            uv3_1 = lp + A - B
            uv3_2 = lp - A + B
            ax.plot([uv3_1[0],uv3_2[0]],[uv3_1[1],uv3_2[1]],'-',linewidth=linewidth,alpha=shade,color=color)
        elif abs(angle-2*np.pi/3)<0.01: # 120 Deg
            uv3_1 = lp + A + B
            uv3_2 = lp - A - B
            ax.plot([uv3_1[0],uv3_2[0]],[uv3_1[1],uv3_2[1]],'-',linewidth=linewidth,alpha=shade,color=color)
    ax.axis(axsize)

def plot_vector_arrows(vec_a,vec_b,vec_a_lab=None,vec_b_lab=None,arrow_size=40,color='b',fontsize = 18):
    """
    Plot vector arrows for Cell on current axis
    """
    
    ax = plt.gca()
    axsize = ax.axis()
    
    # Vector arrows and lattice point labels
    if vec_a_lab is None:
        vec_a_lab = 'a*'
    if vec_b_lab is None:
        vec_b_lab = 'b*'
    
    plot_arrow([0,vec_a[0,0]],[0,vec_a[0,1]],arrow_size=arrow_size,col=color)
    plt.text(vec_a[0,0],vec_a[0,1],vec_a_lab,fontname='Times',weight='bold',size=fontsize)
    plot_arrow([0,vec_b[0,0]],[0,vec_b[0,1]],arrow_size=arrow_size,col=color)
    plt.text(vec_b[0,0],vec_b[0,1],vec_b_lab,fontname='Times',weight='bold',size=fontsize)
    
    ax.axis(axsize)
