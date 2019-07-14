"""
DansCrystalGUI.py
  Creates a graphical user interface for Dans_Diffraction software. 
  Based on tkinter.
  
  Usage:
      ipython -i -matplotlib tk DansCrystalGUI.py
  
  Requires:
      numpy, matplotlib, tkinter

By Dan Porter, PhD
Diamond
2019

Version 1.2
Last updated: 13/07/19

Version History:
10/11/17 0.1    Program created
23/02/19 1.0    Finished the basic program and gave it colours
09/03/19 1.1    Added properties, superstructure, other improvements
13/07/19 1.2    Added FDMNES windows

@author: DGPorter
"""

# Built-ins
import sys, os

import matplotlib.pyplot as plt # Plotting
import numpy as np
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
else:
    import tkinter as tk
    from tkinter import filedialog
    #from tkinter import messagebox

# Internal functions
from .classes_crystal import Crystal
from . import functions_general as fg
from . import functions_plotting as fp
from . import functions_crystallography as fc
from .classes_structures import Structures
from .classes_fdmnes import fdmnes_checker
if fdmnes_checker():
    from .classes_fdmnes import Fdmnes, FdmnesAnalysis

structure_list = Structures()

__version__ = '1.2'

# Fonts
TF = ["Times", 12]
BF = ["Times", 14]
SF = ["Times New Roman", 14]
LF = ["Times", 14]
HF = ['Courier',12]
# Colours - background
bkg = 'snow'
ety = 'white'
btn = 'azure' #'light slate blue'
opt = 'azure' #'light slate blue'
btn2 = 'gold'
# Colours - active
btn_active = 'grey'
opt_active = 'grey'
# Colours - Fonts
txtcol = 'black'
btn_txt = 'black'
ety_txt = 'black'
opt_txt = 'black'


class Crystalgui:
    """
    Provide options for plotting and viewing Crystal data
    """
    def __init__(self, xtl=None):
        """"Initialise Main GUI"""
        if xtl is None:
            self.xtl = Crystal()
        else:
            self.xtl = xtl
        
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Crystal  by D G Porter [dan.porter@diamond.ac.uk]')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        # Create Widget elements from top down
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT,anchor=tk.N)
        
        # Filename (variable)
        f_file = tk.Frame(frame)
        f_file.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
        self.file = tk.StringVar(frame, self.xtl.filename)
        var = tk.Label(f_file, text='CIF file:', font=SF, width=10)
        var.pack(side=tk.LEFT, expand=tk.NO)
        var = tk.Label(f_file, textvariable=self.file, font=TF)
        var.pack(side=tk.LEFT, expand=tk.YES)
        var = tk.Button(f_file, text='Load CIF', font=BF, bg=btn, activebackground=btn_active, command=self.fun_loadcif)
        var.pack(side=tk.LEFT, expand=tk.NO, padx=5)
        
        # Name (variable)
        f_name = tk.Frame(frame)
        f_name.pack(side=tk.TOP,expand=tk.YES,fill=tk.X)
        self.name = tk.StringVar(frame, self.xtl.name)
        var = tk.Label(f_name, text='Name:', font=SF, width=10)
        var.pack(side=tk.LEFT)
        var = tk.Entry(f_name, textvariable=self.name, font=TF, width=40, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>',self.update_name)
        var.bind('<KP_Enter>',self.update_name)

        # List of structures
        self.struct_list = tk.StringVar(frame, 'Structures')
        var = tk.OptionMenu(f_name, self.struct_list, *structure_list.list, command=self.fun_loadstruct)
        var.config(font=SF, width=20, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.RIGHT, fill=tk.X, padx=5)
        
        # Buttons 1
        f_but = tk.Frame(frame)
        f_but.pack(side=tk.TOP)
        var = tk.Button(f_but, text='Lattice\nParameters', font=BF, bg=btn, activebackground=btn_active, command=self.fun_latpar)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Symmetric\nPositions', font=BF, bg=btn, activebackground=btn_active, command=self.fun_atoms)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Symmetry\nOperations', font=BF, bg=btn, activebackground=btn_active, command=self.fun_symmetry)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='General\nPositions', font=BF, bg=btn, activebackground=btn_active, command=self.fun_structure)
        var.pack(side=tk.LEFT)
        
        # Buttons 2
        f_but = tk.Frame(frame)
        f_but.pack(side=tk.TOP)
        var = tk.Button(f_but, text='Plot\nCrystal', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_plotxtl)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Plot\nLayers', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_plotlayers)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Simulate\nStructure Factors', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_simulate)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Properties\n& Conversions', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_properties)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Super\nStructure', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_superstructure)
        var.pack(side=tk.LEFT)
        if fdmnes_checker():
            var = tk.Button(f_but, text='Run\nFDMNES', bg=btn, activebackground=btn_active, font=BF,
                            command=self.fun_fdmnes)
            var.pack(side=tk.LEFT)

        # start mainloop
        # In interactive mode, this freezes the terminal
        # To stop this, need a way of checking if interactive or not ('-i' in sys.argv)
        # However sys in this file is not the same as sys in the operating script
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    ###################################################################################
    ############################## FUNCTIONS ##########################################
    ###################################################################################
    def fun_set(self):
        self.file.set(self.xtl.filename)
        self.name.set(self.xtl.name)
    
    def fun_get(self):
        self.xtl.name = self.name.get()
    
    def fun_loadcif(self):
        #root = Tk().withdraw() # from Tkinter
        defdir = os.path.join(os.path.dirname(__file__), 'Structures')
        #defdir = os.path.expanduser('~')
        filename = filedialog.askopenfilename(initialdir=defdir,
                                              filetypes=[('cif file', '.cif'), ('magnetic cif', '.mcif'),
                                                         ('All files', '.*')])  # from tkFileDialog
        if filename:
            self.xtl = Crystal(filename)
            self.fun_set()

    def fun_loadstruct(self, event=None):
        """Load from structure_list"""
        if self.struct_list.get() in structure_list.list:
            self.xtl = getattr(structure_list, self.struct_list.get()).build()
            self.fun_set()

    def update_name(self):
        newname = self.name.get()
        self.xtl.name = newname
    
    def fun_latpar(self):
        self.fun_set()
        Latpargui(self.xtl)
    
    def fun_atoms(self):
        self.fun_set()
        if np.any(self.xtl.Atoms.mxmymz()):
            Atomsgui(self.xtl,True,True)
        else:
            Atomsgui(self.xtl,True,False)
    
    def fun_structure(self):
        self.fun_set()
        if np.any(self.xtl.Structure.mxmymz()):
            Atomsgui(self.xtl,False,True)
        else:
            Atomsgui(self.xtl,False,False)
    
    def fun_symmetry(self):
        self.fun_set()
        Symmetrygui(self.xtl)
    
    def fun_plotxtl(self):
        self.fun_set()
        self.xtl.Plot.plot_crystal()
        plt.show()

    def fun_plotlayers(self):
        self.fun_set()
        self.xtl.Plot.plot_layers(layer_axis=2, layer_width=0.15, show_labels=True)
        plt.show()
    
    def fun_simulate(self):
        self.fun_set()
        Scatteringgui(self.xtl)

    def fun_properties(self):
        self.fun_set()
        Propertiesgui(self.xtl)

    def fun_superstructure(self):
        self.fun_set()
        Superstructuregui(self.xtl)

    def fun_fdmnes(self):
        from .classes_fdmnes import Fdmnes, FdmnesAnalysis
        RunFDMNESgui(self.xtl)

    def on_closing(self):
        """End mainloop on close window"""
        self.root.destroy()


class Latpargui:
    """
    View and edit the lattice parameters
    """
    def __init__(self,xtl):
        """"Initialise"""
        self.xtl = xtl
        self.Cell = xtl.Cell
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(xtl.name+' Lattice Parameters')
        #self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT,anchor=tk.N)
        
        # Name
        #var = tk.Label(frame, text='Name: {}'.format(self.xtl.name))
        #var.pack(side=tk.TOP)
        
        # Lattice constants
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.TOP)
        
        self.a = tk.DoubleVar(frame,self.Cell.a)
        self.b = tk.DoubleVar(frame,self.Cell.b)
        self.c = tk.DoubleVar(frame,self.Cell.c)
        
        var = tk.Label(frm1, text='a:', width=10, font=SF)
        var.pack(side=tk.LEFT,padx=5,pady=5)
        var = tk.Entry(frm1, textvariable=self.a, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT,padx=2,pady=5)
        var = tk.Label(frm1, text='b:', width=10, font=SF)
        var.pack(side=tk.LEFT,padx=5,pady=5)
        var = tk.Entry(frm1, textvariable=self.b, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT,padx=2,pady=5)
        var = tk.Label(frm1, text='c:', width=10, font=SF)
        var.pack(side=tk.LEFT,padx=5,pady=5)
        var = tk.Entry(frm1, textvariable=self.c, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT,padx=2,pady=5)
        
        # Lattice angles
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.TOP)
        
        self.alpha = tk.DoubleVar(frame,self.Cell.alpha)
        self.beta = tk.DoubleVar(frame,self.Cell.beta)
        self.gamma = tk.DoubleVar(frame,self.Cell.gamma)
        
        var = tk.Label(frm1, text='Alpha:', width=10, font=SF)
        var.pack(side=tk.LEFT,padx=5,pady=5)
        var = tk.Entry(frm1, textvariable=self.alpha, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT,padx=2,pady=5)
        var = tk.Label(frm1, text='Beta:', width=10, font=SF)
        var.pack(side=tk.LEFT,padx=5,pady=5)
        var = tk.Entry(frm1, textvariable=self.beta, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT,padx=2,pady=5)
        var = tk.Label(frm1, text='Gamma:', width=10, font=SF)
        var.pack(side=tk.LEFT,padx=5,pady=5)
        var = tk.Entry(frm1, textvariable=self.gamma, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT,padx=2,pady=5)
        
        # Button
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.TOP)
        
        var = tk.Button(frm1, text='Update', command=self.fun_update, bg=btn, activebackground=btn_active, font=BF)
        var.pack()
    
    def fun_update(self):
        "Update lattice parameters, close window"
        a = self.a.get()
        b = self.b.get()
        c = self.c.get()
        alpha = self.alpha.get()
        beta = self.beta.get()
        gamma = self.gamma.get()
        
        self.Cell.latt([a,b,c,alpha,beta,gamma])
        
        # Close window
        self.root.destroy()


class Atomsgui:
    """
    View and edit the atomic positions
    """
    def __init__(self, xtl, symmetric_only=True, magnetic_moments=False):
        """Initialise"""
        self.xtl = xtl
        self.symmetric_only = symmetric_only
        self.magnetic_moments = magnetic_moments
        if symmetric_only:
            self.Atoms = xtl.Atoms
            ttl = xtl.name + ' Symmetric Atomic Sites'
        else:
            self.Atoms = xtl.Structure
            ttl = xtl.name + ' General Atomic Sites'
        
        if magnetic_moments:
            label_text = '    n   Atom   Label       u              v            w             Occ         Uiso      mx             my           mz             '
            default_width = 55
            mag_button = 'Hide Magnetic Moments'
        else:
            label_text = '    n   Atom   Label       u              v            w             Occ         Uiso'
            default_width = 55
            mag_button = 'Show Magnetic Moments'
        
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(ttl)
        #self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP,fill=tk.BOTH, expand=tk.YES)
        
        #--- label ---
        labframe = tk.Frame(frame,relief='groove')
        labframe.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(labframe, text=label_text, font=SF, justify='left')
        var.pack(side=tk.LEFT)
        
        #--- Button ---
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.BOTTOM, fill=tk.X)
        var = tk.Button(frm1, text='Update', font=BF, command=self.fun_update, bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        var = tk.Button(frm1, text=mag_button, font=BF, command=self.fun_magnets, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        
        #--- Text box ---
        frame_box = tk.Frame(frame)
        frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        
        # Scrollbars
        scanx = tk.Scrollbar(frame_box,orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)
        
        scany = tk.Scrollbar(frame_box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Editable string box
        self.text = tk.Text(frame_box, width=default_width, height=10, font=HF, wrap=tk.NONE, bg=ety)
        self.text.pack(side=tk.TOP,fill=tk.BOTH, expand=tk.YES)
        self.fun_set()
        
        self.text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)

    def fun_set(self):
        "Get positions from crystal object, fill text boxes"
        # Build string
        str = ''
        if self.magnetic_moments:
            fmt = '%3.0f %4s %5s %7.4f %7.4f %7.4f   %4.2f %6.4f   %7.4f %7.4f %7.4f\n'
            for n in range(0,len(self.Atoms.u)):
                str += fmt % (n,self.Atoms.type[n],self.Atoms.label[n],\
                           self.Atoms.u[n],self.Atoms.v[n],self.Atoms.w[n],\
                           self.Atoms.occupancy[n],self.Atoms.uiso[n],\
                           self.Atoms.mx[n],self.Atoms.my[n],self.Atoms.mz[n])
        else:
            fmt = '%3.0f %4s %5s %7.4f %7.4f %7.4f   %4.2f %6.4f\n'
            for n in range(0,len(self.Atoms.u)):
                 str += fmt % (n,self.Atoms.type[n],self.Atoms.label[n],\
                               self.Atoms.u[n],self.Atoms.v[n],self.Atoms.w[n],\
                               self.Atoms.occupancy[n],self.Atoms.uiso[n])
            
        # Insert string in text box
        self.text.insert(tk.END,str)
    
    def fun_magnets(self):
        """"ReOpen window with magnetic moments"""
        
        # Close window
        self.root.destroy()
        
        Atomsgui(self.xtl,self.symmetric_only,not self.magnetic_moments)
    
    def fun_update(self):
        "Update atomic properties, close window"
        # Get string from text box
        str = self.text.get('1.0', tk.END)
        
        # Analyse string
        """
        values = np.genfromtxt(str)
        n = values[:,0]
        type = values[:,1]
        label = values[:,2]
        u = values[:,3]
        v = values[:,4]
        w = values[:,5]
        occ = values[:,6]
        uiso = values[:,7]
        """
        lines = str.splitlines()
        n = []
        type = []
        label = []
        u = []
        v = []
        w = []
        occ = []
        uiso = []
        mx = []
        my = []
        mz = []
        for ln in lines:
            items = ln.split()
            if len(items) < 8: continue
            n += [int(items[0])]
            type += [items[1]]
            label += [items[2]]
            u += [float(items[3])]
            v += [float(items[4])]
            w += [float(items[5])]
            occ += [float(items[6])]
            uiso += [float(items[7])]
            if self.magnetic_moments:
                mx += [float(items[8])]
                my += [float(items[9])]
                mz += [float(items[10])]
        
        self.Atoms.type = type
        self.Atoms.label = label
        self.Atoms.u = np.array(u)
        self.Atoms.v = np.array(v)
        self.Atoms.w = np.array(w)
        self.Atoms.occupancy = np.array(occ)
        self.Atoms.uiso = np.array(uiso)
        if self.magnetic_moments:
            self.Atoms.mx = np.array(mx)
            self.Atoms.my = np.array(my)
            self.Atoms.mz = np.array(mz)
        else:
            self.Atoms.mx = np.zeros(len(u))
            self.Atoms.my = np.zeros(len(u))
            self.Atoms.mz = np.zeros(len(u))
        
        if self.symmetric_only:
            # Apply symmetry if updating basic structure parameters
            self.xtl.generate_structure()
        
        # Close window
        self.root.destroy()


class Symmetrygui:
    """
    View and edit the symmetry operations
    """
    def __init__(self,xtl):
        """"Initialise"""
        self.xtl = xtl
        self.Symmetry = xtl.Symmetry
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(xtl.name+' Symmetry Operations')
        #self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP)
        
        # ---Labels---
        label_frame = tk.Frame(frame,relief=tk.GROOVE)
        label_frame.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(label_frame, text='n', font=SF, justify=tk.CENTER, width=4, height=2, relief=tk.RIDGE)
        var.pack(side=tk.LEFT,pady=1)
        var = tk.Label(label_frame, text='Symmetry Operations', font=SF, justify=tk.CENTER, width=25, height=2,
                       relief=tk.RIDGE)
        var.pack(side=tk.LEFT,pady=1)
        var = tk.Label(label_frame, text='Magnetic Operations', font=SF, justify=tk.CENTER, width=25, height=2,
                       relief=tk.RIDGE)
        var.pack(side=tk.LEFT,pady=1)
        
        # ---Edit Boxes---
        box_frame = tk.Frame(frame)
        box_frame.pack(side=tk.TOP, fill=tk.BOTH)
        
        # Vertical Scrollbar
        self.scany = tk.Scrollbar(box_frame, command=self.fun_scroll)
        self.scany.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Box 1: n
        self.text_1 = tk.Text(box_frame, width=4, height=10, font=SF, bg=ety)
        self.text_1.pack(side=tk.LEFT,pady=1,padx=1)
        # Box 1: Symmetry operations
        self.text_2 = tk.Text(box_frame, width=28, height=10, font=SF, bg=ety)
        self.text_2.pack(side=tk.LEFT,pady=1)
        # Box 1: Magnetic operations
        self.text_3 = tk.Text(box_frame, width=28, height=10, font=SF, bg=ety)
        self.text_3.pack(side=tk.LEFT,pady=1)
        
        # Mouse Wheel scroll:
        self.text_1.bind("<MouseWheel>", self.fun_mousewheel)
        self.text_2.bind("<MouseWheel>", self.fun_mousewheel)
        self.text_3.bind("<MouseWheel>", self.fun_mousewheel)
        
        self.text_1.config(yscrollcommand=self.fun_move)
        self.text_2.config(yscrollcommand=self.fun_move)
        self.text_3.config(yscrollcommand=self.fun_move)
        #scany.config(command=self.text1.yview)
        
        self.fun_set()

        frm1 = tk.Frame(self.root)
        frm1.pack(side=tk.TOP)

        # Button
        var = tk.Button(frm1, text='Update', command=self.fun_update, height=2, font=BF, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # uvw entry
        var = tk.Label(frm1, text='(u v w)=')
        var.pack(side=tk.LEFT)
        self.uvw = tk.StringVar(frm1, '0.5 0 0')
        var = tk.Entry(frm1, textvariable=self.uvw, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_symuvw)
        var.bind('<KP_Enter>', self.fun_symuvw)
        var = tk.Button(frm1, text='Symmetric\nPositions', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_symuvw)
        var.pack(side=tk.LEFT)

        # hkl entry
        var = tk.Label(frm1, text='(h k l)=')
        var.pack(side=tk.LEFT)
        self.hkl = tk.StringVar(frm1, '1 0 0')
        var = tk.Entry(frm1, textvariable=self.hkl, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_symhkl)
        var.bind('<KP_Enter>', self.fun_symhkl)
        var = tk.Button(frm1, text='Symmetric\nReflections', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_symhkl)
        var.pack(side=tk.LEFT)

    def update(self):
        """Update crystal"""

        # Get string from text box
        sym = self.text_2.get('1.0', tk.END)  # symmetry operations
        mag = self.text_3.get('1.0', tk.END)  # magnetic operations

        # remove 'm'
        mag = mag.replace('m', '')

        # Analyse string
        sym_lines = sym.splitlines()
        mag_lines = mag.splitlines()

        if len(sym_lines) != len(mag_lines):
            print('Warning: The number of symmetry and magnetic operations are not the same!')

        sym_ops = []
        mag_ops = []
        for sym, mag in zip(sym_lines, mag_lines):
            if len(sym.strip()) == 0: continue
            if len(mag.strip()) == 0: mag = fc.symmetry_ops2magnetic([sym.strip()])[0]
            sym_ops += [sym.strip()]
            mag_ops += [mag.strip()]

        self.Symmetry.symmetry_operations = sym_ops
        self.Symmetry.symmetry_operations_magnetic = mag_ops
        self.Symmetry.generate_matrices()

    def fun_move(self, *args):
        """Move within text frame"""
        self.scany.set(*args)
        self.text_1.yview('moveto',args[0])
        self.text_2.yview('moveto',args[0])
        self.text_3.yview('moveto',args[0])
    
    def fun_scroll(self, *args):
        """Move scrollbar"""
        self.text_1.yview(*args)
        self.text_2.yview(*args)
        self.text_3.yview(*args)
    
    def fun_mousewheel(self, event):
        """Move scollbar"""
        self.text_1.yview("scroll", event.delta,"units")
        self.text_2.yview("scroll",event.delta,"units")
        self.text_3.yview("scroll",event.delta,"units")
        # this prevents default bindings from firing, which
        # would end up scrolling the widget twice
        return "break"
    
    def fun_set(self):
        """Generate the text box"""
        # Build string
        Nops = len(self.Symmetry.symmetry_operations)
        symstr = ''
        magstr = ''
        for n in range(Nops):
            self.text_1.insert(tk.END,'%3.0f\n'%n)
            sym = self.Symmetry.symmetry_operations[n].strip('\"\'')
            mag = self.Symmetry.symmetry_operations_magnetic[n]
            mag = mag.strip('\"\'').replace('x','mx').replace('y','my').replace('z','mz')
            symstr += '%25s\n' %(sym)
            magstr += '%25s\n' %(mag)
        
        # Insert string in text box
        self.text_2.insert(tk.END,symstr)
        self.text_3.insert(tk.END,magstr)

    def fun_symuvw(self, event=None):
        """create symmetric uvw position"""
        self.update()
        uvw = self.uvw.get()
        uvw = uvw.replace(',', ' ')  # remove commas
        uvw = uvw.replace('(', '').replace(')', '')  # remove brackets
        uvw = uvw.replace('[', '').replace(']', '')  # remove brackets
        uvw = np.fromstring(uvw, sep=' ')
        out = self.xtl.Symmetry.print_symmetric_coordinate_operations(uvw)
        StringViewer(out, 'Symmetric Positions')

    def fun_symhkl(self, event=None):
        """create symmetric hkl reflections"""
        self.update()
        hkl = self.hkl.get()
        hkl = hkl.replace(',', ' ')  # remove commas
        hkl = hkl.replace('(', '').replace(')', '')  # remove brackets
        hkl = hkl.replace('[', '').replace(']', '')  # remove brackets
        hkl = np.fromstring(hkl, sep=' ')
        out = self.xtl.Symmetry.print_symmetric_vectors(hkl)
        StringViewer(out, 'Symmetric Reflections')
    
    def fun_update(self):
        """Update button"""

        self.update()

        # Close window
        self.root.destroy()


class Propertiesgui:
    """
    Show properties of atomis in Crystal
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Properties %s' % xtl.name)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Crystal Atoms
        atoms = np.unique(xtl.Atoms.type)
        elements = fc.atom_properties(None,'Element')

        # Variables
        self.zfraction = tk.DoubleVar(frame, 1)
        self.atoms = tk.StringVar(frame, ' '.join(atoms))
        self.atomopt = tk.StringVar(frame, 'Elements:')
        self.energy_kev = tk.DoubleVar(frame, 8.0)
        self.wavelength = tk.DoubleVar(frame, 1.5498)
        self.edge = tk.StringVar(frame, 'Edge')
        self.twotheta = tk.DoubleVar(frame, 90.0)
        self.qmag = tk.DoubleVar(frame, 5.733)
        self.dspace = tk.DoubleVar(frame, 1.096)

        # X-ray edges:
        self.xr_edges, self.xr_energies = self.xtl.Properties.xray_edges()
        self.xr_edges.insert(0, 'Cu Ka')
        self.xr_edges.insert(1, 'Mo Ka')
        self.xr_energies.insert(0, fg.Cu)
        self.xr_energies.insert(1, fg.Mo)

        # ---Line 1---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, pady=5)

        # Cell properties
        var = tk.Label(line, text='Weight = %8.2f g/mol'%xtl.Properties.weight(), font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text='Volume = %8.2f A^3' % xtl.Properties.volume(), font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text='Density = %8.2f g/cm' % xtl.Properties.density(), font=TF)
        var.pack(side=tk.LEFT)

        # ---Line 2---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, pady=5)

        # Energy wavelength Conversions
        var = tk.Label(line, text='Energy:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.energy_kev, font=TF, width=16, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>',self.fun_energy2wave)
        var.bind('<KP_Enter>',self.fun_energy2wave)
        var = tk.Label(line, text='keV <-> Wavelength:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.wavelength, font=TF, width=16, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_wave2energy)
        var.bind('<KP_Enter>', self.fun_wave2energy)
        var = tk.Label(line, text='A', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.edge, *self.xr_edges, command=self.fun_edge)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # ---Line 3---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, pady=5)

        # Two-Theta - Q - d conversion
        var = tk.Label(line, text='Two-Theta:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.twotheta, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_tth2q)
        var.bind('<KP_Enter>', self.fun_tth2q)
        var = tk.Label(line, text='Deg <-> Q:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.qmag, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_q2tth)
        var.bind('<KP_Enter>', self.fun_q2tth)
        var = tk.Label(line, text='A^-1 <-> d-spacing:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.dspace, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_d2tth)
        var.bind('<KP_Enter>', self.fun_d2tth)
        var = tk.Label(line, text='A', font=TF)
        var.pack(side=tk.LEFT)


        # ---Line 3---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, pady=5)

        var = tk.Label(line, text='Elements', font=LF)
        var.pack(side=tk.LEFT)

        var = tk.Entry(line, textvariable=self.atoms, font=TF, width=16, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        var = tk.OptionMenu(line, self.atomopt, *elements, command=self.fun_element)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Mass Fraction', font=BF, command=self.fun_frac, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text=' Z:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.zfraction, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # ---Line 4---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, pady=5)

        var = tk.Button(line, text='Properties', height=2, font=BF, command=self.fun_prop, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='X-Ray\nScattering Factor', font=BF, command=self.fun_xrscat, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Magnetic\nForm Factor', font=BF, command=self.fun_magff, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='X-Ray\nAttenuation', font=BF, command=self.fun_xratten, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

    def fun_frac(self):
        """Atomic Fraction"""
        z = self.zfraction.get()
        out = self.xtl.Properties.molfraction(z)
        StringViewer(out, self.xtl.name)

    def fun_energy2wave(self, event=None):
        """Convert energy to wavelength"""
        energy = self.energy_kev.get()
        wavelength = fc.energy2wave(energy)
        self.wavelength.set(round(wavelength, 4))
        self.fun_tth2q()

    def fun_wave2energy(self, event=None):
        """Convert wavelength to energy"""
        wavelength = self.wavelength.get()
        energy = fc.wave2energy(wavelength)
        self.energy_kev.set(round(energy, 5))
        self.fun_tth2q()

    def fun_edge(self, event=None):
        """Add edge energy"""
        edge = self.edge.get()
        if self.edge.get() in self.xr_edges:
            idx = self.xr_edges.index(edge)
            self.energy_kev.set(self.xr_energies[idx])
            self.fun_energy2wave()

    def fun_tth2q(self, event=None):
        """Convert two-theta to q and d"""
        tth = self.twotheta.get()
        energy = self.energy_kev.get()
        qmag = fc.calQmag(tth, energy)
        dspace = fc.q2dspace(qmag)
        self.qmag.set(round(qmag, 4))
        self.dspace.set(round(dspace, 4))

    def fun_q2tth(self, event=None):
        """Convert Q to tth and d"""
        qmag = self.qmag.get()
        energy = self.energy_kev.get()
        tth = fc.cal2theta(qmag, energy)
        dspace = fc.q2dspace(qmag)
        self.twotheta.set(round(tth, 4))
        self.dspace.set(round(dspace, 4))

    def fun_d2tth(self, event=None):
        """Convert d to tth and q"""
        dspace = self.dspace.get()
        energy = self.energy_kev.get()
        qmag = fc.q2dspace(dspace)
        tth = fc.cal2theta(qmag, energy)
        self.qmag.set(round(qmag, 4))
        self.twotheta.set(round(tth, 4))

    def fun_element(self, event=None):
        """Dropdown menu"""
        atom = self.atomopt.get()
        self.atoms.set(atom)

    def fun_prop(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',',' ')
        elelist = elements.split()
        out = fc.print_atom_properties(elelist)
        StringViewer(out, 'Element Properties')

    def fun_xrscat(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',',' ')
        elelist = elements.split()
        fp.plot_xray_scattering_factor(elelist)
        plt.show()

    def fun_magff(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',',' ')
        elelist = elements.split()
        fp.plot_magnetic_form_factor(elelist)
        plt.show()

    def fun_xratten(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',',' ')
        elelist = elements.split()
        fp.plot_xray_attenuation(elelist)
        plt.show()


class Scatteringgui:
    """
    Simulate scattering of various forms
    """
    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Scattering %s' % xtl.name)
        #self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT,anchor=tk.N)
        
        # Variatbles
        self.energy_kev = tk.DoubleVar(frame, 8.0)
        self.edge = tk.StringVar(frame, 'Edge')
        self.type = tk.StringVar(frame,'X-Ray')
        self.orientation = tk.StringVar(frame,'Reflection')
        self.direction_h = tk.IntVar(frame,0)
        self.direction_k = tk.IntVar(frame,0)
        self.direction_l = tk.IntVar(frame,1)
        self.theta_offset = tk.DoubleVar(frame,0.0)
        self.theta_min = tk.DoubleVar(frame,-180.0)
        self.theta_max = tk.DoubleVar(frame,180.0)
        self.twotheta_min = tk.DoubleVar(frame,-180.0)
        self.twotheta_max = tk.DoubleVar(frame,180.0)
        self.powder_units = tk.StringVar(frame, 'Two-Theta')
        self.hkl_check = tk.StringVar(frame,'0 0 1')
        self.hkl_result = tk.StringVar(frame,'I:%10.0f TTH:%8.2f'%(0,0))
        self.val_i = tk.IntVar(frame,0)
        self.hkl_magnetic = tk.StringVar(frame,'0 0 1')
        self.azim_zero = tk.StringVar(frame,'1 0 0')
        self.isres = tk.BooleanVar(frame,True)
        self.psival = tk.DoubleVar(frame,0.0)
        self.polval = tk.StringVar(frame,u'\u03c3-\u03c0')
        self.resF0 = tk.DoubleVar(frame,0.0)
        self.resF1 = tk.DoubleVar(frame, 1.0)
        self.resF2 = tk.DoubleVar(frame, 0.0)
        self.magresult = tk.StringVar(frame, 'I = --')

        # X-ray edges:
        self.xr_edges, self.xr_energies = self.xtl.Properties.xray_edges()
        self.xr_edges.insert(0, 'Cu Ka')
        self.xr_edges.insert(1, 'Mo Ka')
        self.xr_energies.insert(0, fg.Cu)
        self.xr_energies.insert(1, fg.Mo)

        #---Line 1---
        line1 = tk.Frame(frame)
        line1.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        var = tk.Label(line1, text='Scattering', font=LF)
        var.pack(side=tk.LEFT)

        var = tk.Button(line1, text='Supernova', font=BF, command=self.fun_supernova, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        var = tk.Button(line1, text='Wish', font=BF, command=self.fun_wish, bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        var = tk.Button(line1, text='I16', font=BF, command=self.fun_i16, bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        
        #---Line 2---
        line2 = tk.Frame(frame)
        line2.pack(side=tk.TOP,fill=tk.X,pady=5)
        
        # Energy
        var = tk.Label(line2, text='Energy (keV):',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line2, self.edge, *self.xr_edges, command=self.fun_edge)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line2, textvariable=self.energy_kev, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # Type
        types = ['X-Ray', 'Neutron', 'XRay Magnetic', 'Neutron Magnetic', 'XRay Resonant']
        var = tk.Label(line2, text='Type:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line2, self.type, *types)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        
        # Orientation
        orients = ['None', 'Reflection', 'Transmission']
        var = tk.OptionMenu(line2, self.orientation, *orients)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        
        # Direction
        var = tk.Label(line2, text='Direction:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line2, textvariable=self.direction_h, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line2, textvariable=self.direction_k, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line2, textvariable=self.direction_l, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # --- Line 3 ---
        line3 = tk.Frame(frame)
        line3.pack(side=tk.TOP,fill=tk.X,pady=5)
        
        # Theta offset
        var = tk.Label(line3, text='Offset:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line3, textvariable=self.theta_offset, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # Theta min
        var = tk.Label(line3, text='Min Theta:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line3, textvariable=self.theta_min, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # Theta max
        var = tk.Label(line3, text='Max Theta:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line3, textvariable=self.theta_max, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # TwoTheta min
        var = tk.Label(line3, text='Min TwoTheta:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line3, textvariable=self.twotheta_min, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # TwoTheta max
        var = tk.Label(line3, text='Max TwoTheta:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line3, textvariable=self.twotheta_max, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # --- Line 4 ---
        line4 = tk.Frame(frame)
        line4.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        var = tk.Button(line4, text='Display Intensities', font=BF, command=self.fun_intensities, bg=btn2,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        
        var = tk.Button(line4, text='Plot Powder', font=BF, command=self.fun_powder, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        xaxistypes = ['two-theta', 'd-spacing', 'Q']
        var = tk.Label(line4, text='Units:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line4, self.powder_units, *xaxistypes)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        
        # hkl check
        hklbox = tk.LabelFrame(line4, text='Quick Check')
        hklbox.pack(side=tk.RIGHT)
        var = tk.Entry(hklbox, textvariable=self.hkl_check, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>',self.fun_hklcheck)
        var.bind('<KP_Enter>',self.fun_hklcheck)
        var = tk.Label(hklbox, textvariable=self.hkl_result,font=TF, width=22)
        var.pack(side=tk.LEFT)
        #var = tk.Button(hklbox, text='Check HKL', font=BF, command=self.fun_hklcheck, bg=btn,
        #                activebackground=btn_active)
        #var.pack(side=tk.LEFT, pady=2)
        
        # --- Line 5 ---
        line5 = tk.Frame(frame)
        line5.pack(side=tk.TOP,pady=5)

        # ---HKL Planes---
        # i value
        var = tk.Label(line5, text='i:',font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line5, textvariable=self.val_i, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        
        # directions
        vframe = tk.Frame(line5)
        vframe.pack(side=tk.LEFT, padx=3)
        var = tk.Button(vframe, text='HKi', font=BF, command=self.fun_hki,width=5, bg=btn, activebackground=btn_active)
        var.pack()
        var = tk.Button(vframe, text='HiL', font=BF, command=self.fun_hil,width=5, bg=btn, activebackground=btn_active)
        var.pack()
        
        vframe = tk.Frame(line5)
        vframe.pack(side=tk.LEFT)
        var = tk.Button(vframe, text='iKL', font=BF, command=self.fun_ikl,width=5, bg=btn, activebackground=btn_active)
        var.pack()
        var = tk.Button(vframe, text='HHi', font=BF, command=self.fun_hhi,width=5, bg=btn, activebackground=btn_active)
        var.pack()

        #---X-ray Magnetic scattering----
        if np.any(self.xtl.Structure.mxmymz()):
            resbox = tk.LabelFrame(line5, text='X-Ray Magnetic Scattering')
            resbox.pack(side=tk.LEFT, fill=tk.Y, padx=3)

            # Resonant HKL, azimuthal reference
            vframe = tk.Frame(resbox)
            vframe.pack(side=tk.LEFT, fill=tk.Y, padx=3)

            hframe = tk.Frame(vframe)
            hframe.pack()
            var = tk.Label(hframe, text='       HKL:', font=SF, width=11)
            var.pack(side=tk.LEFT)
            var = tk.Entry(hframe, textvariable=self.hkl_magnetic, font=TF, width=6, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)
            var.bind('<Return>',self.fun_hklmag)
            var.bind('<KP_Enter>',self.fun_hklmag)

            hframe = tk.Frame(vframe)
            hframe.pack()
            var = tk.Label(vframe, text='Azim. Ref.:', font=SF, width=11)
            var.pack(side=tk.LEFT)
            var = tk.Entry(vframe, textvariable=self.azim_zero, font=TF, width=6, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)

            # Resonant value
            vframe = tk.Frame(resbox)
            vframe.pack(side=tk.LEFT, fill=tk.Y, padx=3)

            hframe = tk.Frame(vframe)
            hframe.pack()
            var = tk.Label(hframe, text='F0:', font=SF)
            var.pack(side=tk.LEFT)
            var = tk.Entry(hframe, textvariable=self.resF0, font=TF, width=3, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)

            hframe = tk.Frame(vframe)
            hframe.pack()
            var = tk.Label(hframe, text='F1:', font=SF)
            var.pack(side=tk.LEFT)
            var = tk.Entry(hframe, textvariable=self.resF1, font=TF, width=3, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)

            hframe = tk.Frame(vframe)
            hframe.pack()
            var = tk.Label(hframe, text='F2:', font=SF)
            var.pack(side=tk.LEFT)
            var = tk.Entry(hframe, textvariable=self.resF2, font=TF, width=3, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)

            vframe = tk.Frame(resbox)
            vframe.pack(side=tk.LEFT, fill=tk.Y, padx=3)

            # Polarisation
            poltypes = [u'\u03c3-\u03c3', u'\u03c3-\u03c0', u'\u03c0-\u03c3', u'\u03c0-\u03c0']
            hframe = tk.Frame(vframe)
            hframe.pack()
            var = tk.Label(hframe, text='Polarisation:', font=SF)
            var.pack(side=tk.LEFT)
            var = tk.OptionMenu(hframe, self.polval, *poltypes)
            var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
            var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
            var.pack(side=tk.LEFT)

            hframe = tk.Frame(vframe)
            hframe.pack()

            # Resonant tickbox
            var = tk.Checkbutton(hframe, text='Resonant', variable=self.isres, font=SF)
            var.pack(side=tk.LEFT, padx=6)
            # psi
            var = tk.Label(hframe, text='psi:', font=SF, width=4)
            var.pack(side=tk.LEFT)
            var = tk.Entry(hframe, textvariable=self.psival, font=TF, width=4, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)
            var.bind('<Return>',self.fun_hklmag)
            var.bind('<KP_Enter>',self.fun_hklmag)

            vframe = tk.Frame(resbox)
            vframe.pack(side=tk.LEFT, fill=tk.Y, padx=3)

            # Azimuth Button
            var = tk.Button(vframe, text='Calc. Mag. Inten.', font=BF, command=self.fun_hklmag, bg=btn,
                            activebackground=btn_active)
            var.pack()
            # Magnetic Result
            var = tk.Label(vframe, textvariable=self.magresult, font=SF, width=12)
            var.pack(fill=tk.Y)

            # Azimuth Button
            var = tk.Button(resbox, text='Simulate\n Azimuth', font=BF, command=self.fun_azimuth, width=7, bg=btn,
                            activebackground=btn_active)
            var.pack(side=tk.LEFT)

    def fun_set(self):
        """"Set gui parameters from crystal"""
        
        self.type.set(self.xtl._scattering_type)
        #self.energy_kev.set(8)
        self.theta_offset.set(self.xtl._scattering_theta_offset)
        self.theta_min.set(self.xtl._scattering_min_theta)
        self.theta_max.set(self.xtl._scattering_max_theta)
        self.twotheta_min.set(self.xtl._scattering_min_two_theta)
        self.twotheta_max.set(self.xtl._scattering_max_two_theta)
        
        if self.orientation.get() == 'Reflection':
            self.direction_h.set(self.xtl._scattering_specular_direction[0])
            self.direction_k.set(self.xtl._scattering_specular_direction[1])
            self.direction_l.set(self.xtl._scattering_specular_direction[2])
        else:
            self.direction_h.set(self.xtl._scattering_parallel_direction[0])
            self.direction_k.set(self.xtl._scattering_parallel_direction[1])
            self.direction_l.set(self.xtl._scattering_parallel_direction[2])
    
    def fun_get(self):
        """Set crytal parameters from gui"""
        
        scat = self.xtl.Scatter
        scat._scattering_type = self.type.get()
        scat._energy_kev = self.energy_kev.get()
        scat._scattering_theta_offset = self.theta_offset.get()
        scat._scattering_min_theta = self.theta_min.get()
        scat._scattering_max_theta = self.theta_max.get()
        scat._scattering_min_two_theta = self.twotheta_min.get()
        scat._scattering_max_two_theta = self.twotheta_max.get()
        scat._powder_units = self.powder_units.get()
        
        if self.orientation.get() == 'Reflection':
            scat._scattering_specular_direction[0] = self.direction_h.get()
            scat._scattering_specular_direction[1] = self.direction_k.get()
            scat._scattering_specular_direction[2] = self.direction_l.get()
        elif self.orientation.get() == 'Transmission':
            scat._scattering_parallel_direction[0] = self.direction_h.get()
            scat._scattering_parallel_direction[1] = self.direction_k.get()
            scat._scattering_parallel_direction[2] = self.direction_l.get()
    
    def fun_i16(self):
        """"Add I16 parameters"""
        
        self.type.set('X-Ray')
        self.energy_kev.set(8.0)
        self.edge.set('Edge')
        self.powder_units.set('Two-Theta')
        self.theta_offset.set(0.0)
        self.theta_min.set(-20.0)
        self.theta_max.set(150.0)
        self.twotheta_min.set(0.0)
        self.twotheta_max.set(130.0)

    def fun_wish(self):
        """"Add Wish parameters"""

        self.type.set('Neutron')
        self.energy_kev.set(17.7)
        self.edge.set('Edge')
        self.powder_units.set('d-spacing')
        self.theta_offset.set(0.0)
        self.theta_min.set(-180.0)
        self.theta_max.set(180.0)
        self.twotheta_min.set(10.0)
        self.twotheta_max.set(170.0)

    def fun_supernova(self):
        """Add SuperNova parameters"""

        self.type.set('X-Ray')
        idx = self.xr_edges.index('Mo Ka')
        self.edge.set('Mo Ka')
        self.energy_kev.set(self.xr_energies[idx])
        self.powder_units.set('Two-Theta')
        self.theta_offset.set(0.0)
        self.theta_min.set(-180.0)
        self.theta_max.set(180.0)
        self.twotheta_min.set(-170.0)
        self.twotheta_max.set(170.0)

    def fun_edge(self, event=None):
        """X-ray edge option menu"""
        edge = self.edge.get()
        if self.edge.get() in self.xr_edges:
            idx = self.xr_edges.index(edge)
            self.energy_kev.set(self.xr_energies[idx])

    def fun_hklcheck(self, event=None):
        """"Show single hkl intensity"""
        
        self.fun_get()
        hkl = self.hkl_check.get()
        hkl = hkl.replace(',', ' ') # remove commas
        hkl = hkl.replace('(','').replace(')','') # remove brackets
        hkl = hkl.replace('[','').replace(']','') # remove brackets
        hkl = np.fromstring(hkl,sep=' ')
        I = self.xtl.Scatter.intensity(hkl)

        unit = self.powder_units.get()
        energy = self.energy_kev.get()
        tth = self.xtl.Cell.tth(hkl, energy)

        if unit.lower() in ['tth', 'angle', 'twotheta', 'theta', 'two-theta']:
            self.hkl_result.set('I:%10.0f TTH:%8.2f' % (I, tth))
        elif unit.lower() in ['d', 'dspace', 'd-spacing', 'dspacing']:
            q = fc.calQmag(tth, energy)
            d = fc.q2dspace(q)
            self.hkl_result.set('I:%10.0f   d:%8.2f A' % (I, d))
        else:
            q = fc.calQmag(tth, energy)
            self.hkl_result.set('I:%8.0f   Q:%8.2f A^-1' % (I, q))

    def fun_intensities(self):
        """Display intensities"""
        
        self.fun_get()
        if self.orientation.get() == 'Reflection':
            string = self.xtl.Scatter.print_ref_reflections(min_intensity=-1, max_intensity=None)
        elif self.orientation.get() == 'Transmission':
            string = self.xtl.Scatter.print_tran_reflections(min_intensity=-1, max_intensity=None)
        else:
            string = self.xtl.Scatter.print_all_reflections(min_intensity=-1, max_intensity=None)
        StringViewer(string, 'Intensities %s' % self.xtl.name)
    
    def fun_powder(self):
        """Plot Powder"""
        self.fun_get()
        energy = self.energy_kev.get()
        min_q = fc.calQmag(self.twotheta_min.get(), energy)
        max_q = fc.calQmag(self.twotheta_max.get(), energy)
        if min_q < 0: min_q=0.0

        if self.xtl.Scatter._powder_units.lower() in ['tth', 'angle', 'twotheta', 'theta', 'two-theta']:
            minx = fc.cal2theta(min_q, energy)
            maxx = fc.cal2theta(max_q, energy)
        elif self.xtl.Scatter._powder_units.lower() in ['d', 'dspace', 'd-spacing', 'dspacing']:
            if min_q < 0.01: min_q = 0.5
            minx = 0
            maxx = fc.q2dspace(min_q)
        else:
            minx = min_q
            maxx = max_q

        self.xtl.Plot.simulate_powder(energy)
        plt.xlim(minx, maxx)
        plt.show()
    
    def fun_hki(self):
        """Plot hki plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hk0(i)
        plt.show()
    
    def fun_hil(self):
        """Plot hil plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_h0l(i)
        plt.show()
    
    def fun_ikl(self):
        """Plot ikl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_0kl(i)
        plt.show()
    
    def fun_hhi(self):
        """Plot hhl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hhl(i)
        plt.show()

    def fun_hklmag(self, event=None):
        """"Magnetic scattering"""

        energy_kev = self.energy_kev.get()
        hkl = self.hkl_magnetic.get()
        hkl = hkl.replace(',', ' ')  # remove commas
        hkl = hkl.replace('(', '').replace(')', '')  # remove brackets
        hkl = hkl.replace('[', '').replace(']', '')  # remove brackets
        hkl = np.fromstring(hkl, sep=' ')

        azi = self.azim_zero.get()
        azi = azi.replace(',', ' ')  # remove commas
        azi = azi.replace('(', '').replace(')', '')  # remove brackets
        azi = azi.replace('[', '').replace(']', '')  # remove brackets
        azi = np.fromstring(azi, sep=' ')

        psi = self.psival.get()
        pol = self.polval.get()
        if pol == u'\u03c3-\u03c3':
            pol = 's-s'
        elif pol == u'\u03c3-\u03c0':
            pol = 's-p'
        elif pol == u'\u03c0-\u03c3':
            pol = 'p-s'
        else:
            pol = 'p-p'

        F0 = self.resF0.get()
        F1 = self.resF1.get()
        F2 = self.resF2.get()

        isres = self.isres.get()
        if isres:
            # Resonant scattering
            maginten = self.xtl.Scatter.xray_resonant_magnetic(
                hkl,
                energy_kev=energy_kev,
                azim_zero=azi, psi=psi,
                polarisation=pol,
                F0=F0, F1=F1, F2=F2)
        else:
            # Non-Resonant scattering
            maginten = self.xtl.Scatter.xray_nonresonant_magnetic(
                hkl,
                energy_kev=energy_kev,
                azim_zero=azi, psi=psi,
                polarisation=pol)

        self.magresult.set('I = %9.4g'%maginten)

    def fun_azimuth(self):
        """Simulate azimuthal magnetic scattering"""

        energy_kev = self.energy_kev.get()
        hkl = self.hkl_magnetic.get()
        hkl = hkl.replace(',', ' ')  # remove commas
        hkl = hkl.replace('(', '').replace(')', '')  # remove brackets
        hkl = hkl.replace('[', '').replace(']', '')  # remove brackets
        hkl = np.fromstring(hkl, sep=' ')

        azi = self.azim_zero.get()
        azi = azi.replace(',', ' ')  # remove commas
        azi = azi.replace('(', '').replace(')', '')  # remove brackets
        azi = azi.replace('[', '').replace(']', '')  # remove brackets
        azi = np.fromstring(azi, sep=' ')

        pol = self.polval.get()
        if pol == u'\u03c3-\u03c3':
            pol = 's-s'
        elif pol == u'\u03c3-\u03c0':
            pol = 's-p'
        elif pol == u'\u03c0-\u03c3':
            pol = 'p-s'
        else:
            pol = 'p-p'

        F0 = self.resF0.get()
        F1 = self.resF1.get()
        F2 = self.resF2.get()

        isres = self.isres.get()
        if isres:
            # Resonant scattering
            self.xtl.Plot.simulate_azimuth_resonant(
                hkl,
                energy_kev=energy_kev,
                azim_zero=azi,
                polarisation=pol,
                F0=F0, F1=F1, F2=F2)
            plt.show()
        else:
            # Non-Resonant scattering
            self.xtl.Plot.simulate_azimuth_nonresonant(
                hkl,
                energy_kev=energy_kev,
                azim_zero=azi,
                polarisation=pol)
            plt.show()


class Superstructuregui:
    """
    Generate a superstructure
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Properties %s' % xtl.name)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Variables
        if hasattr(xtl, 'P'):
            # self.P = xtl.P
            self.P = np.linalg.inv(xtl.P)
        else:
            self.P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.a1 = tk.DoubleVar(frame, self.P[0][0])
        self.a2 = tk.DoubleVar(frame, self.P[0][1])
        self.a3 = tk.DoubleVar(frame, self.P[0][2])
        self.b1 = tk.DoubleVar(frame, self.P[1][0])
        self.b2 = tk.DoubleVar(frame, self.P[1][1])
        self.b3 = tk.DoubleVar(frame, self.P[1][2])
        self.c1 = tk.DoubleVar(frame, self.P[2][0])
        self.c2 = tk.DoubleVar(frame, self.P[2][1])
        self.c3 = tk.DoubleVar(frame, self.P[2][2])
        self.parent_hkl = tk.StringVar(frame, '1 0 0')
        self.super_hkl = tk.StringVar(frame, '1 0 0')
        lp = self.xtl.Cell.lp()
        self.latpar1 = tk.StringVar(frame, '    a = %6.3f    b = %6.3f     c = %6.3f' % lp[:3])
        self.latpar2 = tk.StringVar(frame, 'alpha = %6.3f beta = %6.3f gamma = %6.3f' % lp[3:])

        # Supercell box
        box = tk.LabelFrame(frame, text='Superstructure Cell')
        box.pack(side=tk.TOP, padx=4)

        # ---Line B1---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, expand=tk.TRUE, pady=5)
        var = tk.Label(bline, text='a\' =')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.a1, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*a + ')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.a2, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*b + ')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.a3, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*c')
        var.pack(side=tk.LEFT)

        # ---Line B2---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, expand=tk.TRUE, pady=5)
        var = tk.Label(bline, text='b\' =')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.b1, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*a + ')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.b2, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*b + ')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.b3, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*c')
        var.pack(side=tk.LEFT)

        # ---Line B3---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, expand=tk.TRUE, pady=5)
        var = tk.Label(bline, text='c\' =')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.c1, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*a + ')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.c2, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*b + ')
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.c3, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_updatecell)
        var.bind('<KP_Enter>', self.fun_updatecell)
        var = tk.Label(bline, text='*c')
        var.pack(side=tk.LEFT)

        # Update
        var = tk.Button(frame, text='Update', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_updatecell)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)

        # Cell properties
        var = tk.Label(frame, textvariable=self.latpar1)
        var.pack(side=tk.TOP)
        var = tk.Label(frame, textvariable=self.latpar2)
        var.pack(side=tk.TOP)

        # Index hkl
        line = tk.Label(frame)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text='Parent (h k l)=')
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.parent_hkl, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_parent2super)
        var.bind('<KP_Enter>', self.fun_parent2super)
        var = tk.Label(line, text=' <-> Super (h\' k\' l\')=')
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.super_hkl, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_super2parent)
        var.bind('<KP_Enter>', self.fun_super2parent)

        # Generate
        var = tk.Button(frame, text='Generate Supercell', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_gencell)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.TRUE)

    def fun_updatecell(self, event=None):
        """Update cell parameters"""
        self.P = [[self.a1.get(), self.a2.get(), self.a3.get()],
                  [self.b1.get(), self.b2.get(), self.b3.get()],
                  [self.c1.get(), self.c2.get(), self.c3.get()]]
        newUV = self.xtl.Cell.calculateR(self.P)
        newLP = fc.UV2latpar(newUV)

        self.latpar1.set('    a = %6.3f    b = %6.3f     c = %6.3f' % newLP[:3])
        self.latpar2.set('alpha = %6.3f beta = %6.3f gamma = %6.3f' % newLP[3:])
        self.fun_parent2super()

    def fun_gencell(self, event=None):
        """Generate new supercell"""
        sup = self.xtl.generate_superstructure(self.P)
        Crystalgui(sup)
        self.root.destroy()

    def fun_parent2super(self, event=None):
        """Convert parent hkl to super hkl"""
        hkl = self.parent_hkl.get()
        hkl = hkl.replace(',', ' ')  # remove commas
        hkl = hkl.replace('(', '').replace(')', '')  # remove brackets
        hkl = hkl.replace('[', '').replace(']', '')  # remove brackets
        hkl = np.fromstring(hkl, sep=' ')

        newUV = self.xtl.Cell.calculateR(self.P)
        newUVstar = fc.RcSp(newUV)
        Q = self.xtl.Cell.calculateQ(hkl)
        newhkl = fc.indx(Q, newUVstar)
        newhkl = tuple([round(x, 3) for x in newhkl[0]])
        self.super_hkl.set('%1.3g %1.3g %1.3g' % newhkl)

    def fun_super2parent(self, event=None):
        """Convert super hkl to parent hkl"""
        hkl = self.super_hkl.get()
        hkl = hkl.replace(',', ' ')  # remove commas
        hkl = hkl.replace('(', '').replace(')', '')  # remove brackets
        hkl = hkl.replace('[', '').replace(']', '')  # remove brackets
        hkl = np.fromstring(hkl, sep=' ')

        newUV = self.xtl.Cell.calculateR(self.P)
        newUVstar = fc.RcSp(newUV)
        Q = np.dot(hkl, newUVstar)
        newhkl = self.xtl.Cell.indexQ(Q)
        newhkl = tuple([round(x, 3) for x in newhkl[0]])
        self.parent_hkl.set('%1.3g %1.3g %1.3g' % newhkl)


class RunFDMNESgui:
    """
    Create FDMNES indata files and run them
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        self.fdm = Fdmnes(xtl)
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('FDMNES %s' % xtl.name)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        default_width = 60
        default_height = 30

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Available x-ray edges
        self.xr_edges, self.xr_energies = self.xtl.Properties.xray_edges()
        # Polarisations
        poltypes = [u'\u03c3-\u03c3', u'\u03c3-\u03c0', u'\u03c0-\u03c3', u'\u03c0-\u03c0']

        # Variables
        if hasattr(xtl, 'P'):
            # Superstructure with parent
            pass
        else:
            # Not superstructure
            pass
        self.calculation_name = tk.StringVar(frame, self.fdm.output_name)
        self.calculation_path = tk.StringVar(frame, self.fdm.generate_input_path())
        self.calc_comment = tk.StringVar(frame, self.fdm.comment)
        self.calc_range = tk.StringVar(frame, self.fdm.range)
        self.calc_radius = tk.DoubleVar(frame, self.fdm.radius)
        self.calc_edge = tk.StringVar(frame, self.xr_edges[0])
        self.calc_green = tk.BooleanVar(frame, self.fdm.green)
        self.calc_scf = tk.BooleanVar(frame, self.fdm.scf)
        self.calc_quad = tk.BooleanVar(frame, self.fdm.quadrupole)
        self.calc_aziref = tk.StringVar(frame, str(self.fdm.azi_ref).strip('[]'))
        self.calc_reflist = tk.StringVar(frame, str(self.fdm.hkl_reflections))
        self.calc_addref = tk.StringVar(frame, '1, 0, 0')
        self.calc_addpol = tk.StringVar(frame,  poltypes[1])

        # ---Line 0---
        line = tk.Frame(frame)
        line.pack(expand=tk.YES, fill=tk.X)
        var = tk.Label(line, text='FDMNES Path: %s' % self.fdm.exe_path, font=SF)
        var.pack(side=tk.LEFT)

        # ---Line 1---
        line = tk.Frame(frame)
        line.pack(expand=tk.YES, fill=tk.X)
        var = tk.Label(line, text='Path:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.calculation_path, font=HF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
        var = tk.Button(line, text='Select', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_loadpath)
        var.pack(side=tk.LEFT, padx=5)

        # ---Line 2---
        line = tk.Frame(frame)
        line.pack(expand=tk.YES, fill=tk.X)
        var = tk.Label(line, text='Calculation Name:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.calculation_name, font=HF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text='Comment:', font=SF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.calc_comment, font=HF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X, padx=5)

        # ---Line 3---
        line = tk.Frame(frame)
        line.pack()

        # LLLLeftLLL
        left = tk.Frame(line)
        left.pack(side=tk.LEFT)

        # |||Parameters box|||
        box = tk.LabelFrame(left, text='Calculation Parameters', font=SF)
        box.pack(padx=4, pady=6)

        # ---Line B1---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(bline, text='Range (eV)', font=SF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(bline, textvariable=self.calc_range, font=HF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=5)

        # ---Line B2---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(bline, text='Radius (A)', font=SF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(bline, textvariable=self.calc_radius, font=HF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=5)

        # ---Line B3---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(bline, text='Edge:', font=SF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.OptionMenu(bline, self.calc_edge, *self.xr_edges)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT, padx=5)

        # ---Line B4---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Checkbutton(bline, text='Muffin-Tin', variable=self.calc_green, font=SF)
        var.pack(side=tk.LEFT, padx=6)
        var = tk.Checkbutton(bline, text='SCF', variable=self.calc_scf, font=SF)
        var.pack(side=tk.LEFT, padx=6)

        # ---Line B5---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Checkbutton(bline, text='Quadrupole', variable=self.calc_quad, font=SF)
        var.pack(side=tk.LEFT, padx=6)

        # ---Line B6---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(bline, text='Azimuthal Reference:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.calc_aziref, font=SF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, expand=tk.YES, padx=6)

        # ---Line B7---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(bline, text='h,k,l:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(bline, textvariable=self.calc_addref, font=SF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, expand=tk.YES, padx=6)
        """
        var = tk.OptionMenu(bline, self.calc_addpol, *poltypes)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        """
        var = tk.Button(bline, text='Add Ref', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_addref)
        var.pack(side=tk.LEFT)

        # ---Line B8---
        bline = tk.Frame(box)
        bline.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(bline, text='Reflections:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Label(bline, textvariable=self.calc_reflist, font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Button(bline, text='Clear', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_clearref)
        var.pack(side=tk.LEFT)

        # ---Line  B9---
        bline = tk.Frame(box)
        bline.pack(fill=tk.X)
        var = tk.Button(bline, text='Update input text', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_update)
        var.pack(side=tk.RIGHT)

        # LLLLine 2LLL
        lline = tk.Frame(left)
        lline.pack()
        var = tk.Button(lline, text='Create Directory + input files', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_writefiles)
        var.pack()
        var = tk.Button(lline, text='Write fdm start file', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_writefdmfile)
        var.pack()
        var = tk.Button(lline, text='Write Files & Run FDMNES', font=BF, bg=btn2, activebackground=btn_active,
                        command=self.fun_runfdmnes)
        var.pack()
        var = tk.Button(lline, text='Plot Results', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_analyse)
        var.pack()

        # |||Text Box|||
        box = tk.LabelFrame(line, text='FDMNES Input File')
        box.pack(side=tk.LEFT, padx=4, fill=tk.BOTH, expand=tk.YES)

        # Scrollbars
        scanx = tk.Scrollbar(box, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)

        scany = tk.Scrollbar(box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)

        # Editable string box
        self.text = tk.Text(box, width=default_width, height=default_height, font=HF, wrap=tk.NONE, bg=ety)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.gentxt()

        self.text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)

    def update(self):
        """Updates the internal fdm object"""

        filepath, filename = os.path.split(self.calculation_path.get())
        self.fdm.output_path = filepath
        self.fdm.input_name = filename
        self.fdm.output_name = self.calculation_name.get()
        self.fdm.comment = self.calc_comment.get()
        self.fdm.range = self.calc_range.get()
        self.fdm.radius = self.calc_radius.get()
        absedge = self.calc_edge.get()
        self.fdm.absorber, self.fdm.edge = absedge.split()
        self.fdm.green = self.calc_green.get()
        self.fdm.scf = self.calc_scf.get()
        self.fdm.quadrupole = self.calc_quad.get()
        self.fdm.azi_ref = [float(n) for n in self.calc_aziref.get().replace(',', ' ').split()]

    def gentxt(self):
        """Generate input file text"""
        parstr = self.fdm.generate_parameters_string()
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, parstr)

    def write_reflections(self):
        """ Write reflections string"""
        self.calc_reflist.set('%s'%self.fdm.hkl_reflections)

    def fun_loadpath(self, event=None):
        """Select folder"""
        filepath, filename = os.path.split(self.calculation_path.get())
        filepath = filedialog.askdirectory(initialdir=filepath)
        self.fdm.input_name = filename
        self.fdm.output_path = filepath
        self.calculation_path.set(self.fdm.generate_input_path())

    def fun_addref(self, event=None):
        """Add reflection to list"""
        newhkl = [int(n) for n in self.calc_addref.get().replace(',', ' ').split()]
        newpol = self.calc_addpol.get()
        self.fdm.hkl_reflections += [newhkl]
        self.write_reflections()

    def fun_clearref(self, event=None):
        """Clear reflection list"""
        self.fdm.hkl_reflections = []
        self.write_reflections()

    def fun_update(self, event=None):
        """Update values and generate text"""
        self.update()
        self.gentxt()

    def fun_writefiles(self, event=None):
        """Create FDMNES files"""
        self.update()
        text_string = self.text.get(1.0, tk.END)
        self.fdm.create_files(param_string=text_string)

    def fun_writefdmfile(self, event=None):
        """Create FDMNES files"""
        self.update()
        self.fdm.write_fdmfile()

    def fun_runfdmnes(self, event=None):
        """Run FDMNES"""
        self.update()
        text_string = self.text.get(1.0, tk.END)
        self.fdm.create_files(param_string=text_string)
        self.fdm.write_fdmfile()
        self.fdm.run_fdmnes()  # This will take a few mins, output should be printed to the console
        #self.fdm.info()

    def fun_analyse(self, event=None):
        """Start Analysis GUI"""
        self.update()
        AnaFDMNESgui(self.fdm.output_path, self.fdm.output_name)


class AnaFDMNESgui:
    """
    Read files from a completed FDMNES calculation folder
    """

    def __init__(self, output_path=None, output_name='out'):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('FDMNES %s' % output_path)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        default_width = 60
        default_height = 30

        if output_path is None or not os.path.isfile(os.path.join(output_path, output_name + '.txt')):
            self.fun_loadpath()

        self.fdm = FdmnesAnalysis(output_path, output_name)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # ---Line 1---
        line = tk.Frame(frame)
        line.pack(expand=tk.YES, fill=tk.X)
        var = tk.Label(line, text='Calculation Path: %s' % self.fdm.output_path, font=SF)
        var.pack(side=tk.LEFT, expand=tk.YES)
        var = tk.Button(line, text='Select', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_loadpath)
        var.pack(side=tk.LEFT, padx=5)

        # LLLLeftLLL
        left = tk.Frame(frame)
        left.pack(side=tk.LEFT)

        # ---Line 1---
        line = tk.Frame(left)
        line.pack(expand=tk.YES, fill=tk.X)
        var = tk.Button(line, text='XANES', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_xanes)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Button(line, text='DoS', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_dos)
        var.pack(side=tk.LEFT, padx=5)

        # ---Line 2---
        line = tk.Frame(left)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Reflections:', font=SF)
        var.pack(side=tk.LEFT)
        self.reflist = [ref.replace('(', '').replace(')', '').replace('-', '_') for ref in self.fdm.reflist]
        self.reflection = tk.StringVar(frame, self.reflist[0])
        var = tk.OptionMenu(line, self.reflection, *self.reflist)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # ---Line 3---
        line = tk.Frame(left)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Button(line, text='Plot All Azi/Energy', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_ref3d)
        var.pack(side=tk.LEFT, padx=5)
        # DoS projection button if sph_signal_rxs files available
        if any(['sph' in key for key in self.fdm.__dict__.keys()]):
            var = tk.Button(line, text='Plot DoS Projection', font=BF, bg=btn, activebackground=btn_active,
                            command=self.fun_refdos)
            var.pack(side=tk.LEFT, padx=5)

        # ---Line 4---
        line = tk.Frame(left)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Angle (Deg):', font=SF)
        var.pack(side=tk.LEFT)
        self.cutangle = tk.StringVar(frame, 'max')
        var = tk.Entry(line, textvariable=self.cutangle, font=SF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, expand=tk.YES, padx=6)
        var = tk.Button(line, text='Plot Energy', font=BF, bg=btn, activebackground=btn_active, width=15,
                        command=self.fun_refenergy)
        var.pack(side=tk.LEFT, padx=6)

        # ---Line 5---
        line = tk.Frame(left)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Energy (keV):', font=SF)
        var.pack(side=tk.LEFT)
        self.cutenergy = tk.StringVar(frame, 'max')
        var = tk.Entry(line, textvariable=self.cutenergy, font=SF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, expand=tk.YES, padx=6)
        var = tk.Button(line, text='Plot Azimuth', font=BF, bg=btn, activebackground=btn_active, width=15,
                        command=self.fun_refazim)
        var.pack(side=tk.LEFT, padx=6)

        # RRRRightRRR
        # |||Text Box|||
        box = tk.LabelFrame(frame, text='FDMNES _bav.txt File')
        box.pack(side=tk.LEFT, padx=4, fill=tk.BOTH, expand=tk.YES)

        # Scrollbars
        scanx = tk.Scrollbar(box, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)

        scany = tk.Scrollbar(box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)

        # Editable string box
        self.text = tk.Text(box, width=default_width, height=default_height, font=HF, wrap=tk.NONE, bg=ety)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, self.fdm.output_text)

        self.text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)

    def fun_loadpath(self, event=None):
        """Button Select - Open new folder"""
        filename = filedialog.askopenfilename(filetypes=[('FDMNES output file', '.txt'),
                                                         ('All files', '.*')])  # from tkFileDialog
        if filename:
            filepath, filename = os.path.split(filename)
            self.root.destroy()
            AnaFDMNESgui(filepath)

    def fun_xanes(self, event=None):
        """Button XANES"""
        self.fdm.xanes.plot()
        plt.show()

    def fun_dos(self, event=None):
        """Button DOS"""
        self.fdm.density.plot()
        plt.show()

    def fun_ref3d(self, event=None):
        """Button Plot all Azim/energy"""
        refn = self.reflection.get()
        refob = self.fdm.__getattribute__(refn)
        refob.plot3D()
        plt.show()

    def fun_refdos(self, event=None):
        """Button spherical tensors"""
        refn = self.reflection.get()
        sphrefname = 'sph_' + refn
        if sphrefname in self.fdm.__dict__.keys():
            refob = self.fdm.__getattribute__(sphrefname)
            refob.plot()
            plt.show()
        else:
            print('Spherical tensor file not available')

    def fun_refenergy(self, event=None):
        """Button Plot Energy"""
        refn = self.reflection.get()
        azim = self.cutangle.get()
        try:
            azim = float(azim)
        except ValueError:
            pass
        refob = self.fdm.__getattribute__(refn)
        refob.plot_eng(azim)
        plt.show()

    def fun_refazim(self, event=None):
        """Button Plot Azimuth"""
        refn = self.reflection.get()
        energy = self.cutenergy.get()
        try:
            energy = float(energy)
        except ValueError:
            pass
        refob = self.fdm.__getattribute__(refn)
        refob.plot_azi(energy)
        plt.show()


class StringViewer:
    """
    Simple GUI that displays strings
    """
    def __init__(self, string, title=''):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(title)
        #self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        # Textbox height
        height = string.count('\n')
        if height > 40: height = 40
        
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        
        # --- label ---
        #labframe = tk.Frame(frame,relief='groove')
        #labframe.pack(side=tk.TOP, fill=tk.X)
        #var = tk.Label(labframe, text=label_text,font=SF,justify='left')
        #var.pack(side=tk.LEFT)
        
        # --- Button ---
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.BOTTOM, fill=tk.X)
        var = tk.Button(frm1, text='Close', font=BF, command=self.fun_close, bg=btn, activebackground=btn_active)
        var.pack(fill=tk.X)
        
        # --- Text box ---
        frame_box = tk.Frame(frame)
        frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        
        # Scrollbars
        scanx = tk.Scrollbar(frame_box, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)
        
        scany = tk.Scrollbar(frame_box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Editable string box
        self.text = tk.Text(frame_box, width=40, height=height, font=HF, wrap=tk.NONE)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.text.insert(tk.END, string)
        
        self.text.config(xscrollcommand=scanx.set,yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)
    
    def fun_close(self):
        """close window"""
        self.root.destroy()


if __name__ == '__main__':
    from Dans_Diffraction import Crystal,MultiCrystal,Structures
    S = Structures()
    xtl = S.Diamond.build()
    Crystalgui(xtl)