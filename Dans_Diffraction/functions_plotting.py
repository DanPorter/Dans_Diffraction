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
    - from Dans_Diffraction import functions_plotting as fp

All plots generated require plt.show() call, unless using interactive mode


Version 1.8
Last updated: 03/04/20

Version History:
06/01/18 1.0    Program created from DansGeneralProgs.py V2.3
05/03/18 1.1    Removed plt.show from arrow functions
17/04/18 1.2    Removed plt.show from other functions
03/05/18 1.3    Removed plot_vector_lines(vec_a,vec_b), replaced with plot_lattice_lines(Q, vec_a, vec_b)
30/05/18 1.4    Added multiplot
07/03/19 1.5    Added plot_attenuation and others
22/03/19 1.6    Added plot_circle, updated vecplot and plot_lattice_lines
03/05/19 1.7    Added legend to labels function
09/03/20 1.8    Added default figure size

@author: DGPorter
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from . import functions_general as fg
from . import functions_crystallography as fc

__version__ = '1.8'

DEFAULT_FONT = 'Times New Roman'
FIGURE_SIZE = [12, 10]
FIGURE_DPI = 60

'----------------------------Plot manipulation--------------------------'


def labels(ttl=None, xvar=None, yvar=None, zvar=None, legend=False, size='Normal', font='Times New Roman'):
    """
    Add formatted labels to current plot, also increases the tick size
    :param ttl: title
    :param xvar: x label
    :param yvar: y label
    :param zvar: z label (3D plots only)
    :param legend: False/ True, adds default legend to plot 
    :param size: 'Normal' or 'Big'
    :param font: str font name, 'Times New Roman'
    :return: None
    """

    if size.lower() in ['big', 'large', 'xxl', 'xl']:
        tik = 30
        tit = 32
        lab = 35
        leg = 25
    else:
        # Normal
        tik = 18
        tit = 20
        lab = 22
        leg = 18

    plt.xticks(fontsize=tik, fontname=font)
    plt.yticks(fontsize=tik, fontname=font)
    plt.setp(plt.gca().spines.values(), linewidth=2)
    if plt.gca().get_yaxis().get_scale() != 'log':
        plt.ticklabel_format(useOffset=False)
        plt.ticklabel_format(style='sci',scilimits=(-3,3))

    if ttl is not None:
        plt.gca().set_title(ttl, fontsize=tit, fontweight='bold', fontname=font)

    if xvar is not None:
        plt.gca().set_xlabel(xvar, fontsize=lab, fontname=font)

    if yvar is not None:
        plt.gca().set_ylabel(yvar, fontsize=lab, fontname=font)

    if zvar is not None:
        # Don't think this works, use ax.set_zaxis
        plt.gca().set_zlabel(zvar, fontsize=lab, fontname=font)

    if legend:
        plt.legend(loc=0, frameon=False, prop={'size': leg, 'family': 'serif'})


def saveplot(name, dpi=None, figure=None):
    """
    Saves current figure as a png in the home directory
    :param name: filename, including or expluding directory and or extension
    :param dpi: image resolution, higher means larger image size, default=matplotlib default
    :param figure: figure number, default = plt.gcf()
    :return: None

    E.G.
    ---select figure to save by clicking on it---
    saveplot('test')
    E.G.
    saveplot('c:\somedir\apicture.jpg', dpi=600, figure=3)
    """

    if type(name) is int:
        name = str(name)

    if figure is None:
        gcf = plt.gcf()
    else:
        gcf = plt.figure(figure)

    dir = os.path.dirname(name)
    file, ext = os.path.basename(name)

    if len(dir) == 0:
        dir = os.path.expanduser('~')

    if len(ext) == 0:
        ext = '.png'

    savefile = os.path.join(dir, file+ext)
    gcf.savefig(savefile, dpi=dpi)
    print('Saved Figure {} as {}'.format(gcf.number, savefile))


def newplot(*args, **kwargs):
    """
    Shortcut to creating a simple plot
    E.G.
      x = np.arange(-5,5,0.1)
      y = x**2
      newplot(x,y,'r-',lw=2,label='Line')
    """

    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2

    plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plt.plot(*args, **kwargs)

    plt.setp(plt.gca().spines.values(), linewidth=2)
    plt.xticks(fontsize=25, fontname='Times New Roman')
    plt.yticks(fontsize=25, fontname='Times New Roman')
    plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', scilimits=(-3, 3))


def multiplot(xvals, yvals=None, datarange=None, cmap='jet', labels=None, marker=None):
    """
    Shortcut to creating a simple multiplot with either colorbar or legend
    E.G.
      x = np.arange(-5,5,0.1)
      ys = [x**2, 1+x**2, 2+x**2, 3+x**2, 4+x**2]
      datarange = [0,1,2,3,4]
      multiplot(x, ys, datarange, cmap='winter')
    OR:
      x = np.arange(-5,5,0.1)
      ys = [x**2, 1+x**2, 2+x**2, 3+x**2, 4+x**2]
      labels = ['x*x','2+x*x','3+x*x','4+x*x']
      multiplot(x, ys, labels=labels)
    """

    if yvals is None:
        yvals = xvals
        xvals = []
    yvals = np.asarray(yvals)
    xvals = np.asarray(xvals)

    if datarange is None:
        datarange = range(len(yvals))
    datarange = np.asarray(datarange,dtype=np.float)

    cm = plt.get_cmap(cmap)
    colrange = (datarange - datarange.min()) / (datarange.max() - datarange.min())
    
    if marker is None:
        marker = ''
    linearg = '-' + marker

    plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    for n in range(len(datarange)):
        col = cm(colrange[n])
        if len(xvals) == 0:
            plt.plot(yvals[n], linearg, lw=2, color=col)
        elif len(xvals.shape) == 1:
            plt.plot(xvals, yvals[n], linearg, lw=2, color=col)
        else:
            plt.plot(xvals[n], yvals[n], linearg, lw=2, color=col)

    plt.setp(plt.gca().spines.values(), linewidth=2)
    plt.xticks(fontsize=25, fontname='Times New Roman')
    plt.yticks(fontsize=25, fontname='Times New Roman')
    plt.ticklabel_format(useOffset=False)
    plt.ticklabel_format(style='sci', scilimits=(-3, 3))

    if labels is None:
        # Add Colorbar
        sm = plt.cm.ScalarMappable(cmap=cm)
        sm.set_array(datarange)
        cbar = plt.colorbar(sm)
        #cbar.set_label('variation [unit]', fontsize=24, fontweight='bold', fontname='Times New Roman')
    else:
        # Add legend
        plt.legend(labels, loc=0, frameon=False, prop={'size':20,'family':'serif'})


def newplot3(*args, **kwargs):
    """
    Shortcut to creating a simple 3D plot
    Automatically tiles 1 dimensional x and y arrays to match 2D z array,
    assuming z.shape = (len(x),len(y))

    E.G.
      newplot3([1,2,3,4],[9,8,7],[[2,4,6],[8,10,12],[14,16,18],[20,22,24]],'-o')
    """

    if 'linewidth' and 'lw' not in kwargs.keys():
        kwargs['linewidth'] = 2

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax = fig.add_subplot(111, projection='3d')

    x = np.asarray(args[0], dtype=np.float)
    y = np.asarray(args[1], dtype=np.float)
    z = np.asarray(args[2], dtype=np.float)

    if z.ndim == 2:
        if x.ndim < 2:
            x = np.tile(x, z.shape[1]).reshape(z.T.shape).T
        if y.ndim < 2:
            y = np.tile(y, z.shape[0]).reshape(z.shape)

        # Plot each array independently
        for n in range(len(z)):
            ax.plot(x[n], y[n], z[n], *args[3:], **kwargs)
    else:
        ax.plot(*args, **kwargs)


def sliderplot(YY, X=None, slidervals=None, *args, **kwargs):
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

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    X = np.asarray(X, dtype=np.float)
    Y = np.asarray(YY, dtype=np.float)
    if slidervals is None:
        slidervals = range(Y.shape[0])
    slidervals = np.asarray(slidervals, dtype=np.float)

    if X.ndim < 2:
        X = np.tile(X, Y.shape[0]).reshape(Y.shape)

    plotline, = plt.plot(X[0, :], Y[0, :], *args, **kwargs)
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()

    " Create slider on plot"
    axsldr = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='lightgoldenrodyellow')

    sldr = plt.Slider(axsldr, '', 0, len(slidervals) - 1)
    txt = axsldr.set_xlabel('{} [{}]'.format(slidervals[0], 0), fontsize=18)

    plt.sca(ax)

    " Slider update function"

    def update(val):
        "Update function for pilatus image"
        pno = int(np.floor(sldr.val))
        plotline.set_xdata(X[pno, :])
        plotline.set_ydata(Y[pno, :])
        txt.set_text('{} [{}]'.format(slidervals[pno], pno))
        plt.draw()
        plt.gcf().canvas.draw()
        # fig1.canvas.draw()

    sldr.on_changed(update)


def sliderplot2D(ZZZ, XX=None, YY=None, slidervals=None, *args, **kwargs):
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

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

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
        XX, YY = np.meshgrid(XX, YY)

    p = plt.pcolormesh(XX, YY, ZZZ[:, :, 0])
    # p.set_clim(cax)

    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.autoscale(tight=True)

    " Create slider on plot"
    axsldr = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='lightgoldenrodyellow')

    sldr = plt.Slider(axsldr, '', 0, len(slidervals) - 1)
    txt = axsldr.set_xlabel('{} [{}]'.format(slidervals[0], 0), fontsize=18)

    plt.sca(ax)

    " Slider update function"

    def update(val):
        "Update function for pilatus image"
        pno = int(np.round(sldr.val))
        p.set_array(ZZZ[:-1, :-1, pno].ravel())
        txt.set_text('{} [{}]'.format(slidervals[pno], pno))
        plt.draw()
        plt.gcf().canvas.draw()
        # fig1.canvas.draw()
    sldr.on_changed(update)


def plot_cell(cell_centre=[0, 0, 0], CELL=np.eye(3)):
    """
    Plot a box defined by a unit cell on the current plot
    :param cell_centre: [1x3] array : centre of cell, default [0,0,0]
    :param CELL: [3x3] array : unit cell vectors [A,B,C]
    :return: None
    """

    uvw = np.array([[0., 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], \
                    [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
    uvw = uvw - 0.5  # plot around box centre
    bpos = np.dot(uvw, CELL)
    bpos = bpos + cell_centre
    plt.plot(bpos[:, 0], bpos[:, 1], bpos[:, 2], c='k')  # cell box


def plot_circle(radius=1.0, centre=[0,0], height=0, *args, **kwargs):
    """
    Generate circle on current plot
    :param radius: radius of the circle
    :param centre: [x,y] centre of the circle
    :param height: reduce the radius by increasing the height from a surface
    :param args: plot commands
    :param kwargs: plot commands
    :return: none
    """

    deg = np.linspace(0, 360, 361)
    rad = np.deg2rad(deg)
    x = centre[0] + np.sqrt((radius**2-height**2))*np.cos(rad)
    y = centre[1] + np.sqrt((radius**2-height**2))*np.sin(rad)
    plt.plot(x, y, *args, **kwargs)


def plot_arrow(x, y, z=None, col='r', width=2, arrow_size=40):
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
    if z is None or not hasattr(plt.gca(), 'get_zlim'):
        x0 = x[0]
        y0 = y[0]
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        plt.arrow(x0, y0, dx, dy, width=arrow_size / 4000.0, color=col, length_includes_head=True)
        # V = FancyArrowPatch(x,y, mutation_scale=arrow_size, lw=width, arrowstyle="-|>", color=col)
        # plt.gca().add_artist(V)
        return

    # 3D Arrow
    V = Arrow3D(x, y, z, mutation_scale=arrow_size, lw=width, arrowstyle="-|>", color=col)
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

    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        if 'arrowstyle' not in kwargs.keys():
            kwargs['arrowstyle'] = "-|>"
        if 'mutation_scale' not in kwargs.keys():
            kwargs['mutation_scale'] = 20
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


'----------------------- Crystal Plotting Programs----------------------'


def vecplot(UV, mode='hk0', axis=None, *args, **kwargs):
    """
    Plot grid of a,b vectors on current axis
    :param UV: [a;b;c] array of unit vectors
    :param mode: definition of axis plane, 'hk0', 'h0l', '0kl', 'hhl'
    :param axis: axis to create lines on, if None, plt.gca is used
    :param args: arguments to pass to plot command, e.g. linewidth, alpha, color
    :return: None
    """

    if mode == 'h0l':
        # h0l
        UV = np.dot(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), fg.rot3D(UV, gamma=-90))
    elif mode == '0kl':
        # 0kl
        UV = np.dot(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), UV)
    elif mode == 'hhl':
        # hhl ***untested
        UV = np.dot(np.array([[1, 1, 0], [0, 0, 1], [0, 1, 0]]), UV)

    if axis is None:
        axis = plt.gca()
    axsize = axis.axis()
    latt = axis_lattice_points(UV[0], UV[1], axis=axsize)
    plot_lattice_lines(latt, UV[0], UV[1], axis=axis, *args, **kwargs)


def UV_arrows(UV):
    """
    Plot arrows with a*,b* on current figure
    """
    # Get current axis size
    ax = plt.gca()
    axsize = ax.axis()
    asty = dict(arrowstyle="->")
    plt.annotate("", xy=(UV[0, 0], UV[0, 1]), xytext=(0.0, 0.0), arrowprops=asty)
    plt.annotate("", xy=(UV[1, 0], UV[1, 1]), xytext=(0.0, 0.0), arrowprops=asty)
    plt.annotate("a", (0.1 + UV[0, 0], UV[0, 1] - 0.2))
    plt.annotate("b", (UV[1, 0] - 0.2, 0.1 + UV[1, 1]))
    ax.axis(axsize)


def axis_lattice_points(vec_a=[1, 0, 0], vec_b=[0, 1, 0], axis=[-4, 4, -4, 4]):
    """
    Generate a 2D lattice of points generated by 2 vectors within a 2D axis
    :param vec_a: [1x3] array : a* vector
    :param vec_b: [1x3] array : b* vector
    :param axis: [1x4] axis array, plt.axis()
    :return: [nx3] array of lattice points
    """

    # Vectors
    A = np.asarray(vec_a, dtype=float).reshape([3])
    B = np.asarray(vec_b, dtype=float).reshape([3])
    # Generate a 3D cell to make use of indx function
    U = np.array([A, B, np.cross(A, B)])
    corners = [[axis[1], axis[2], 0],
               [axis[1], axis[3], 0],
               [axis[0], axis[2], 0],
               [axis[0], axis[3], 0]]
    # Determine the coefficients required to generate lattice points of the 2 vectors at
    # all 4 corners of the axis
    idx = fc.indx(corners, U)
    min_x = np.floor(np.min(idx[:, 0]))
    max_x = np.ceil(np.max(idx[:, 0]))
    min_y = np.floor(np.min(idx[:, 1]))
    max_y = np.ceil(np.max(idx[:, 1]))
    hkl = fc.genHKL([min_x, max_x], [min_y, max_y], 0)
    latt = np.dot(hkl, U)
    return latt


def plot_lattice_points2D(Q, markersize=12, color='b', marker='o'):
    """
    Add points to the current axis
    :param Q: [nx2/3] array : lattice points to plot
    :param markersize: default 12
    :param color: default 'b'
    :param marker: default 'o'
    :return: None
    """

    ax = plt.gca()
    axsize = ax.axis()

    ax.plot(Q[:, 0], Q[:, 1], markersize=markersize, color=color, marker=marker)
    ax.axis(axsize)


def plot_lattice_lines(latt, vec_a=[1, 0, 0], vec_b=[0, 1, 0], axis=None, *args, **kwargs):
    """
    Add lines defining the reciprocal lattice to the current plot
        Generates square or hexagonal lines where vertices are the lattice points within the image.
    :param latt: [nx2/3] array : points at which to generate lattice
    :param vec_a: [1x2/3] array : a* vector
    :param vec_b: [1x2/3] array : b* vector
    :param axis: axis to plot on (None for plt.gca)
    :param args: argments to pass to plot function, e.g. linewidth, alpha, color
    :return: None
    """

    if axis is None:
        axis = plt.gca()
    axsize = axis.axis()

    # vectors
    A = np.asarray(vec_a, dtype=float).reshape([3])
    B = np.asarray(vec_b, dtype=float).reshape([3])

    # Angle between vectors
    angle = fg.ang(A, B)

    # At each lattice point, draw the unit vectors
    for n in range(len(latt)):
        lp = latt[n, :]
        uv1_1 = lp - A
        uv1_2 = lp + A
        uv2_1 = lp - B
        uv2_2 = lp + B

        axis.plot([uv1_1[0], uv1_2[0]], [uv1_1[1], uv1_2[1]], 'k-', *args, **kwargs)
        axis.plot([uv2_1[0], uv2_2[0]], [uv2_1[1], uv2_2[1]], 'k-', *args, **kwargs)
        if abs(angle - np.pi / 3) < 0.01:  # 60Deg
            uv3_1 = lp + A - B
            uv3_2 = lp - A + B
            axis.plot([uv3_1[0], uv3_2[0]], [uv3_1[1], uv3_2[1]], 'k-', *args, **kwargs)
        elif abs(angle - 2 * np.pi / 3) < 0.01:  # 120 Deg
            uv3_1 = lp + A + B
            uv3_2 = lp - A - B
            axis.plot([uv3_1[0], uv3_2[0]], [uv3_1[1], uv3_2[1]], 'k-', *args, **kwargs)
    axis.axis(axsize)


def plot_vector_arrows(vec_a=[1, 0, 0], vec_b=[1, 0, 0], vec_a_lab=None, vec_b_lab=None,
                       arrow_size=40, color='b', fontsize=18):
    """
    Plot vector arrows for Cell on current axis
        Will generate two arrows on the current axis, pointing from the origin to vec_a and vec_b, respectivley.
    :param vec_a: [1x2/3] array : a* vector
    :param vec_b: [1x2/3] array : b* vector
    :param vec_a_lab: str : e.g. 'a*'
    :param vec_b_lab: str : e.g. 'b*'
    :param arrow_size: size of arrow, default 40
    :param color:  arror colour, default 'b'
    :param fontsize: text size, default 18
    :return: None
    """

    vec_a = np.asarray(vec_a).reshape([-1, np.shape(vec_a)[-1]])
    vec_b = np.asarray(vec_b).reshape((-1, np.shape(vec_b)[-1]))

    ax = plt.gca()
    axsize = ax.axis()

    # Vector arrows and lattice point labels
    if vec_a_lab is None:
        vec_a_lab = 'a*'
    if vec_b_lab is None:
        vec_b_lab = 'b*'

    plot_arrow([0, vec_a[0, 0]], [0, vec_a[0, 1]], arrow_size=arrow_size, col=color)
    plt.text(vec_a[0, 0], vec_a[0, 1], vec_a_lab, fontname=DEFAULT_FONT, weight='bold', size=fontsize)
    plot_arrow([0, vec_b[0, 0]], [0, vec_b[0, 1]], arrow_size=arrow_size, col=color)
    plt.text(vec_b[0, 0], vec_b[0, 1], vec_b_lab, fontname=DEFAULT_FONT, weight='bold', size=fontsize)
    ax.axis(axsize)


def plot_ewald_coverage(energy_kev, color='k', linewidth=2):
    """
    Plot Ewald coverage of a single axis diffractometer on current plot in 2D
    Includes boundaries for theta=0, twotheta=180 and theta=twotheta

    :param energy_kev: float
    :return: None
    """

    q_max = fc.calqmag(180, energy_kev)

    # calculate diffractometer angles
    angles = np.arange(0, 180, 0.1)
    Q1x, Q1y = fc.diffractometer_Q(angles, 180, energy_kev)  # delta=180
    Q2x, Q2y = fc.diffractometer_Q(angles, angles, energy_kev)  # eta=delta
    Q3x, Q3y = fc.diffractometer_Q(0, angles, energy_kev)  # eta=0

    # Add diffractometer angles
    plt.plot(Q1x, Q1y, color, linewidth, label=r'2$\theta$=180')
    plt.plot(Q2x, Q2y, color, linewidth, label=r'2$\theta$=$\theta$')
    plt.plot(Q3x, Q3y, color, linewidth, label=r'$\theta$=0')
    plt.axis([-q_max, q_max, 0, q_max])


def plot_xray_scattering_factor(elements, maxq=10):
    """
    Plot x-ray scattering factor for 1 or more elements
    :param elements:
    :return: None
    """

    q = np.linspace(0, maxq, 200)
    xrf = fc.xray_scattering_factor(elements, q)

    newplot(q, xrf)
    plt.legend(np.asarray(elements).reshape(-1), loc=0, frameon=False, fontsize=18)
    labels('X-Ray Scattering Factor', 'Q [$\AA^{-1}$]')


def plot_magnetic_form_factor(elements, maxq=10):
    """
    Plot magnetic form factor for 1 or more elements
    :param elements:
    :return: None
    """

    q = np.linspace(0, maxq, 200)
    mff = fc.magnetic_form_factor(elements, q)

    newplot(q, mff)
    plt.legend(np.asarray(elements).reshape(-1), loc=0, frameon=False, fontsize=18)
    labels('Magnetic Form Factor', 'Q [$\AA^{-1}$]')


def plot_xray_attenuation(elements, min_energy=0, max_energy=20):
    """
    Plot x-ray scattering factor for 1 or more elements
    :param elements:
    :return: None
    """

    Zarray = fc.atom_properties(elements, 'Z')
    ene = np.arange(min_energy, max_energy+0.01, 0.01)

    Aarray = fc.attenuation(Zarray, ene)
    newplot(ene, Aarray)
    plt.yscale('log')
    plt.xlim([min_energy, max_energy])
    plt.legend(np.asarray(elements).reshape(-1), loc=0, frameon=False, fontsize=18)
    labels('X-Ray Attenuation', 'Energy [keV]', r'$\mu/\rho$ [cm$^2$/g]')


def plot_xray_scattering_factor(element, min_energy=0.5, max_energy=20):
    """
    Plot x-ray scattering factor for 1 or more elements
    :param element: str name of element to plot
    :param min_energy: float min energy in keV
    :param max_energy: float max energy in keV
    :return: None
    """

    ene = np.arange(min_energy, max_energy+0.01, 0.01)
    f1, f2 = fc.atomic_scattering_factor(element, ene)

    newplot(ene, f1, '-', lw=2, label='f1')
    plt.plot(ene, f2, '-', lw=2, label='f2')
    plt.xlim([min_energy, max_energy])
    labels('X-Ray Scattering Factor\n%s' % element, 'Energy [keV]', None, legend=True)

