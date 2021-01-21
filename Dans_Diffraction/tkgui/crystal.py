"""
Main crystal gui windows
"""

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
    from tkinter import messagebox

# Internal functions
from ..classes_crystal import Crystal
from ..classes_structures import Structures
from ..classes_fdmnes import fdmnes_checker
from .. import functions_general as fg
from .. import functions_plotting as fp
from .. import functions_crystallography as fc
from .basic_widgets import StringViewer, SelectionBox
from .basic_widgets import (TF, BF, SF, LF, HF, TTF,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)
from .scattering import ScatteringGui
from .multi_crystal import MultiCrystalGui
from .multiple_scattering import MultipleScatteringGui
from .tensor_scattering import TensorScatteringGui


class CrystalGui:
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

        self.structure_list = Structures()

        # Create Widget elements from top down
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Filename (variable)
        f_file = tk.Frame(frame)
        f_file.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
        self.file = tk.StringVar(frame, self.short_file())
        var = tk.Label(f_file, text='CIF file:', font=SF, width=10)
        var.pack(side=tk.LEFT, expand=tk.NO)
        var = tk.Label(f_file, textvariable=self.file, width=40, font=TF)
        var.pack(side=tk.LEFT, expand=tk.NO, padx=3)
        var = tk.Button(f_file, text='Load CIF', font=BF, bg=btn, activebackground=btn_active, command=self.fun_loadcif)
        var.pack(side=tk.RIGHT, expand=tk.NO, padx=5)

        # Name (variable)
        f_name = tk.Frame(frame)
        f_name.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
        self.name = tk.StringVar(frame, self.xtl.name)
        var = tk.Label(f_name, text='Name:', font=SF, width=10)
        var.pack(side=tk.LEFT)
        var = tk.Entry(f_name, textvariable=self.name, font=TF, width=40, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.update_name)
        var.bind('<KP_Enter>', self.update_name)

        # List of structures
        self.struct_list = tk.StringVar(frame, 'Structures')
        var = tk.OptionMenu(f_name, self.struct_list, *self.structure_list.list, command=self.fun_loadstruct)
        var.config(font=SF, width=20, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.RIGHT, fill=tk.X, padx=5)

        # Buttons 1
        f_but = tk.Frame(frame)
        f_but.pack(side=tk.TOP)
        var = tk.Button(f_but, text='Crystal\nInfo', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_info)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Lattice\nParameters', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_latpar)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Symmetric\nPositions', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_atoms)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Symmetry\nOperations', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_symmetry)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='General\nPositions', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_structure)
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
        var = tk.Button(f_but, text='Properties\n& Conversions', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_properties)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Multi\nCrystal', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_multicrystal)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Super\nStructure', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_superstructure)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Write\nCIF', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_writecif)
        var.pack(side=tk.LEFT)

        # Buttons 3
        f_but = tk.Frame(frame)
        f_but.pack(side=tk.TOP)
        var = tk.Button(f_but, text='Simulate\nStructure Factors', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_simulate)
        var.pack(side=tk.LEFT)
        var = tk.Button(f_but, text='Multiple\nScattering', bg=btn, activebackground=btn_active, font=BF,
                        command=self.fun_multiple_scattering)
        var.pack(side=tk.LEFT)
        # Remove Tensor-Scattering 26/05/20
        #var = tk.Button(f_but, text='Tensor\nScattering', bg=btn, activebackground=btn_active, font=BF,
        #                command=self.fun_tensor_scattering)
        #var.pack(side=tk.LEFT)
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
        self.file.set(self.short_file())
        self.name.set(self.xtl.name)

    def fun_get(self):
        self.xtl.name = self.name.get()

    def short_file(self):
        path, name = os.path.split(self.xtl.filename)
        path, short_path = os.path.split(path)
        return '/'.join([short_path, name])

    def fun_loadcif(self):
        # root = Tk().withdraw() # from Tkinter
        defdir = os.path.join(os.path.dirname(__file__), 'Structures')
        # defdir = os.path.expanduser('~')
        filename = filedialog.askopenfilename(
            title='Select CIF to open',
            initialdir=defdir,
            filetypes=[('cif file', '.cif'), ('magnetic cif', '.mcif'), ('All files', '.*')],
            parent=self.root
        )
        if filename:
            # Check cif
            cifvals = fc.readcif(filename)
            if fc.cif_check(cifvals):
                self.xtl = Crystal(filename)
                self.fun_set()
            else:
                messagebox.showinfo(
                    title='Dans_Diffraction',
                    message='This cif file is missing a key parameter',
                    parent=self.root
                )
                self.xtl = Crystal(filename)
                self.fun_set()

    def fun_writecif(self, inifile=None):
        if inifile is None:
            inifile = '%s.cif' % self.xtl.name
        defdir = os.path.join(os.path.dirname(__file__), 'Structures')
        filename = filedialog.asksaveasfilename(
            title='Save structure as CIF',
            initialfile=inifile,
            initialdir=defdir,
            filetypes=[('cif file', '.cif'), ('magnetic cif', '.mcif'), ('All files', '.*')],
            parent=self.root
        )
        if filename:
            self.xtl.write_cif(filename)
            messagebox.showinfo(
                title='Dans_Diffraction',
                message='Structure saved as:\n"%s"' % filename,
                parent=self.root
            )

    def fun_loadstruct(self, event=None):
        """Load from structure_list"""
        if self.struct_list.get() in self.structure_list.list:
            self.xtl = getattr(self.structure_list, self.struct_list.get()).build()
            self.fun_set()

    def update_name(self, event=None):
        newname = self.name.get()
        self.xtl.name = newname

    def fun_latpar(self):
        self.fun_set()
        LatparGui(self.xtl)

    def fun_atoms(self):
        self.fun_set()
        if np.any(self.xtl.Atoms.mxmymz()):
            AtomsGui(self.xtl, True, True)
        else:
            AtomsGui(self.xtl, True, False)

    def fun_structure(self):
        self.fun_set()
        if np.any(self.xtl.Structure.mxmymz()):
            AtomsGui(self.xtl, False, True)
        else:
            AtomsGui(self.xtl, False, False)

    def fun_symmetry(self):
        self.fun_set()
        SymmetryGui(self.xtl)

    def fun_info(self):
        """Display Crystal info"""
        string = '%s\n%s' % (self.xtl.filename, self.xtl.info())
        StringViewer(string, self.xtl.name, width=60)

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
        ScatteringGui(self.xtl)

    def fun_multicrystal(self):
        self.fun_set()
        rt = MultiCrystalGui([self.xtl])
        rt.root.focus_force()

    def fun_multiple_scattering(self):
        self.fun_set()
        rt = MultipleScatteringGui(self.xtl)
        rt.root.focus_force()

    def fun_tensor_scattering(self):
        self.fun_set()
        rt = TensorScatteringGui(self.xtl)
        rt.root.focus_force()

    def fun_properties(self):
        self.fun_set()
        PropertiesGui(self.xtl)

    def fun_superstructure(self):
        self.fun_set()
        SuperstructureGui(self.xtl)

    def fun_fdmnes(self):
        from .fdmnes import RunFDMNESgui
        RunFDMNESgui(self.xtl)

    def on_closing(self):
        """End mainloop on close window"""
        self.root.destroy()


class LatparGui:
    """
    View and edit the lattice parameters
    """

    def __init__(self, xtl):
        """"Initialise"""
        self.xtl = xtl
        self.Cell = xtl.Cell
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(xtl.name + ' Lattice Parameters')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Name
        # var = tk.Label(frame, text='Name: {}'.format(self.xtl.name))
        # var.pack(side=tk.TOP)

        # Lattice constants
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.TOP)

        self.a = tk.DoubleVar(frame, self.Cell.a)
        self.b = tk.DoubleVar(frame, self.Cell.b)
        self.c = tk.DoubleVar(frame, self.Cell.c)

        var = tk.Label(frm1, text='a:', width=10, font=SF)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Entry(frm1, textvariable=self.a, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2, pady=5)
        var = tk.Label(frm1, text='b:', width=10, font=SF)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Entry(frm1, textvariable=self.b, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2, pady=5)
        var = tk.Label(frm1, text='c:', width=10, font=SF)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Entry(frm1, textvariable=self.c, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2, pady=5)

        # Lattice angles
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.TOP)

        self.alpha = tk.DoubleVar(frame, self.Cell.alpha)
        self.beta = tk.DoubleVar(frame, self.Cell.beta)
        self.gamma = tk.DoubleVar(frame, self.Cell.gamma)

        var = tk.Label(frm1, text='Alpha:', width=10, font=SF)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Entry(frm1, textvariable=self.alpha, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2, pady=5)
        var = tk.Label(frm1, text='Beta:', width=10, font=SF)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Entry(frm1, textvariable=self.beta, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2, pady=5)
        var = tk.Label(frm1, text='Gamma:', width=10, font=SF)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Entry(frm1, textvariable=self.gamma, width=10, font=TF, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2, pady=5)

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

        self.Cell.latt([a, b, c, alpha, beta, gamma])

        # Close window
        self.root.destroy()


class AtomsGui:
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
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # --- label ---
        labframe = tk.Frame(frame, relief='groove')
        labframe.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(labframe, text=label_text, font=SF, justify='left')
        var.pack(side=tk.LEFT)

        # --- Button ---
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.BOTTOM, fill=tk.X)
        var = tk.Button(frm1, text='Update', font=BF, command=self.fun_update, bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        var = tk.Button(frm1, text=mag_button, font=BF, command=self.fun_magnets, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # --- Text box ---
        frame_box = tk.Frame(frame)
        frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # Scrollbars
        scanx = tk.Scrollbar(frame_box, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)

        scany = tk.Scrollbar(frame_box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)

        # Editable string box
        self.text = tk.Text(frame_box, width=default_width, height=10, font=HF, wrap=tk.NONE, bg=ety)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
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
            for n in range(0, len(self.Atoms.u)):
                str += fmt % (n, self.Atoms.type[n], self.Atoms.label[n],
                              self.Atoms.u[n], self.Atoms.v[n], self.Atoms.w[n],
                              self.Atoms.occupancy[n], self.Atoms.uiso[n],
                              self.Atoms.mx[n], self.Atoms.my[n], self.Atoms.mz[n])
        else:
            fmt = '%3.0f %4s %5s %7.4f %7.4f %7.4f   %4.2f %6.4f\n'
            for n in range(0, len(self.Atoms.u)):
                str += fmt % (n, self.Atoms.type[n], self.Atoms.label[n],
                              self.Atoms.u[n], self.Atoms.v[n], self.Atoms.w[n],
                              self.Atoms.occupancy[n], self.Atoms.uiso[n])

        # Insert string in text box
        self.text.insert(tk.END, str)

    def fun_magnets(self):
        """"ReOpen window with magnetic moments"""

        # Close window
        self.root.destroy()

        AtomsGui(self.xtl, self.symmetric_only, not self.magnetic_moments)

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


class SymmetryGui:
    """
    View and edit the symmetry operations
    """

    def __init__(self, xtl):
        """"Initialise"""
        self.xtl = xtl
        self.Symmetry = xtl.Symmetry
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(xtl.name + ' Symmetry Operations')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frm1 = tk.Frame(self.root)
        frm1.pack(side=tk.TOP)

        # Spacegroup entry
        var = tk.Label(frm1, text='Spacegroup: ', font=TTF)
        var.pack(side=tk.LEFT)
        self.spacegroup = tk.StringVar(frm1, 'P1')
        self.spacegroup_number = tk.StringVar(frm1, '1')
        var = tk.Entry(frm1, textvariable=self.spacegroup_number, font=TF, width=4, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_spacegroup)
        var.bind('<KP_Enter>', self.fun_spacegroup)
        var = tk.Label(frm1, textvariable=self.spacegroup, width=14)
        var.pack(side=tk.LEFT, padx=5)
        # Spacegroup Buttons
        var = tk.Button(frm1, text='Choose\nSpacegroup', command=self.fun_ch_spacegroup, height=2, font=BF, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Button(frm1, text='Choose\nSubgroup', command=self.fun_ch_subgroup, height=2,
                        font=BF, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Button(frm1, text='Choose Magnetic\nSpacegroup', command=self.fun_ch_maggroup, height=2,
                        font=BF, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5, pady=5)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP)

        # ---Labels---
        label_frame = tk.Frame(frame, relief=tk.GROOVE)
        label_frame.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(label_frame, text='n', font=SF, justify=tk.CENTER, width=4, height=2, relief=tk.RIDGE)
        var.pack(side=tk.LEFT, pady=1)
        var = tk.Label(label_frame, text='Symmetry Operations', font=SF, justify=tk.CENTER, width=25, height=2,
                       relief=tk.RIDGE)
        var.pack(side=tk.LEFT, pady=1)
        var = tk.Label(label_frame, text='Magnetic Operations', font=SF, justify=tk.CENTER, width=25, height=2,
                       relief=tk.RIDGE)
        var.pack(side=tk.LEFT, pady=1)

        # ---Edit Boxes---
        box_frame = tk.Frame(frame)
        box_frame.pack(side=tk.TOP, fill=tk.BOTH)

        # Vertical Scrollbar
        self.scany = tk.Scrollbar(box_frame, command=self.fun_scroll)
        self.scany.pack(side=tk.RIGHT, fill=tk.Y)

        # Box 1: n
        self.text_1 = tk.Text(box_frame, width=4, height=10, font=SF, bg=ety)
        self.text_1.pack(side=tk.LEFT, pady=1, padx=1)
        # Box 1: Symmetry operations
        self.text_2 = tk.Text(box_frame, width=28, height=10, font=SF, bg=ety)
        self.text_2.pack(side=tk.LEFT, pady=1)
        # Box 1: Magnetic operations
        self.text_3 = tk.Text(box_frame, width=28, height=10, font=SF, bg=ety)
        self.text_3.pack(side=tk.LEFT, pady=1)

        # Mouse Wheel scroll:
        self.text_1.bind("<MouseWheel>", self.fun_mousewheel)
        self.text_2.bind("<MouseWheel>", self.fun_mousewheel)
        self.text_3.bind("<MouseWheel>", self.fun_mousewheel)

        self.text_1.config(yscrollcommand=self.fun_move)
        self.text_2.config(yscrollcommand=self.fun_move)
        self.text_3.config(yscrollcommand=self.fun_move)
        # scany.config(command=self.text1.yview)

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

        self.fun_set()

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
        self.text_1.yview('moveto', args[0])
        self.text_2.yview('moveto', args[0])
        self.text_3.yview('moveto', args[0])

    def fun_scroll(self, *args):
        """Move scrollbar"""
        self.text_1.yview(*args)
        self.text_2.yview(*args)
        self.text_3.yview(*args)

    def fun_mousewheel(self, event):
        """Move scollbar"""
        self.text_1.yview("scroll", event.delta, "units")
        self.text_2.yview("scroll", event.delta, "units")
        self.text_3.yview("scroll", event.delta, "units")
        # this prevents default bindings from firing, which
        # would end up scrolling the widget twice
        return "break"

    def fun_set(self):
        """Generate the text box"""
        # Clear boxes
        self.text_1.delete('1.0', tk.END)
        self.text_2.delete('1.0', tk.END)
        self.text_3.delete('1.0', tk.END)
        # Build string
        Nops = len(self.Symmetry.symmetry_operations)
        symstr = ''
        magstr = ''
        for n in range(Nops):
            self.text_1.insert(tk.END, '%3.0f\n' % n)
            sym = self.Symmetry.symmetry_operations[n].strip('\"\'')
            mag = self.Symmetry.symmetry_operations_magnetic[n]
            symstr += '%25s\n' % (sym)
            magstr += '%25s\n' % (mag)

        # Insert string in text box
        self.text_2.insert(tk.END, symstr)
        self.text_3.insert(tk.END, magstr)

        # Update spacegroup name + number
        self.spacegroup.set(self.Symmetry.spacegroup)
        self.spacegroup_number.set(self.Symmetry.spacegroup_number)

    def fun_spacegroup(self, event=None):
        """Load spacegroup symmetry"""
        sgn = int(self.spacegroup_number.get())
        self.Symmetry.load_spacegroup(sgn)
        self.fun_set()

    def fun_ch_spacegroup(self):
        """Button Select Spacegroup"""
        current = int(self.spacegroup_number.get())
        sg_list = fc.spacegroup_list().split('\n')
        current_selection = [sg_list[current-1]]
        selection = SelectionBox(self.root, sg_list, current_selection, 'Select SpaceGroup', False).show()
        if len(selection) > 0:
            new_sg = int(selection[0].split()[0])
            self.Symmetry.load_spacegroup(new_sg)
            self.fun_set()

    def fun_ch_subgroup(self):
        """Button select subgroup"""
        current = int(float(self.spacegroup_number.get()))  # str > float > int
        sbg_list = fc.spacegroup_subgroups_list(current).split('\n')
        selection = SelectionBox(self.root, sbg_list, [], 'Select SpaceGroup Subgroup', False).show()
        if len(selection) > 0:
            new_sg = int(selection[0].split()[3])
            self.Symmetry.load_spacegroup(new_sg)
            self.fun_set()

    def fun_ch_maggroup(self):
        """Button Select Magnetic Spacegroup"""
        current = int(self.spacegroup_number.get())
        msg_list = fc.spacegroup_magnetic_list(current).split('\n')
        selection = SelectionBox(self.root, msg_list, [], 'Select Magnetic SpaceGroup', False).show()
        if len(selection) > 0:
            new_sg = selection[0].split()[3]
            self.Symmetry.load_magnetic_spacegroup(new_sg)
            self.fun_set()

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


class PropertiesGui:
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
        atoms = fc.arrange_atom_order(atoms)
        elements = fc.atom_properties(None, 'Element')

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

        # ---Line 0---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, pady=5)

        # Cell Name
        var = tk.Label(line, text=xtl.Properties.molname(), font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text=xtl.Properties.molcharge(), font=TF)
        var.pack(side=tk.LEFT, padx=15)

        # ---Line 1---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, pady=5)

        # Cell properties
        var = tk.Label(line, text='Weight = %8.2f g/mol' % xtl.Properties.weight(), font=TF)
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
        var.bind('<Return>', self.fun_energy2wave)
        var.bind('<KP_Enter>', self.fun_energy2wave)
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

        var = tk.Button(line, text='Magnetic\nForm Factor', font=BF, command=self.fun_magff, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='X-Ray\nAttenuation', font=BF, command=self.fun_xratten, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='X-Ray\nScattering Factor', font=BF, command=self.fun_xsf, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Atomic\nScattering Factor', font=BF, command=self.fun_asf, bg=btn,
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
        qmag = fc.calqmag(tth, energy)
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
        elements = elements.replace(',', ' ')
        elelist = elements.split()
        out = fc.print_atom_properties(elelist)
        width = 12 + 12*len(elelist)
        if width > 120: width=120
        StringViewer(out, 'Element Properties', width)

    def fun_magff(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',', ' ')
        elelist = elements.split()
        fp.plot_magnetic_form_factor(elelist)
        plt.show()

    def fun_xratten(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',', ' ')
        elelist = elements.split()
        fp.plot_xray_attenuation(elelist)
        plt.show()

    def fun_xsf(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',', ' ')
        elelist = elements.split()
        for el in elelist:
            fp.plot_xray_scattering_factor(el)
        plt.show()

    def fun_asf(self):
        """Properties button"""
        elements = self.atoms.get()
        elements = elements.replace(',', ' ')
        elelist = elements.split()
        for el in elelist:
            fp.plot_atomic_scattering_factor(el)
        plt.show()


class SuperstructureGui:
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
        CrystalGui(sup)
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

