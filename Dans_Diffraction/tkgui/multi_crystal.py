"""
GUI for MultiCrystal code
"""

import sys, os

import matplotlib.pyplot as plt
import numpy as np

from .. import functions_general as fg
from .. import functions_crystallography as fc
from ..classes_crystal import Crystal
from ..classes_structures import Structures
from ..classes_multicrystal import MultiCrystal
from .basic_widgets import StringViewer, tk, filedialog
from .basic_widgets import (TF, BF, SF, LF, HF, TTF, TTFG, TTBG,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)


class MultiCrystalGui:
    """
    Combine crystal objects and plot combined scattering
    """

    def __init__(self, crystal_list):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Multi-Crystal')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        self.xtls = MultiCrystal(crystal_list)
        self.structure_list = Structures()

        # Variatbles
        self.struct_list = tk.StringVar(frame, 'Add Structure')
        self.scale = tk.DoubleVar(frame, 1.0)
        self.energy_kev = tk.DoubleVar(frame, 8.0)
        self.edge = tk.StringVar(frame, 'Edge')
        self.type = tk.StringVar(frame, 'X-Ray')
        self.twotheta_min = tk.DoubleVar(frame, 5.0)
        self.twotheta_max = tk.DoubleVar(frame, 60.0)
        self.powder_units = tk.StringVar(frame, 'Two-Theta')

        # X-ray edges:
        self.xr_edges = []
        self.xr_energies = []
        for xtl in self.xtls.crystal_list:
            edges, energies = xtl.Properties.xray_edges()
            self.xr_edges += edges
            self.xr_energies += energies
        self.xr_edges.insert(0, 'Cu Ka')
        self.xr_edges.insert(1, 'Mo Ka')
        self.xr_energies.insert(0, fg.Cu)
        self.xr_energies.insert(1, fg.Mo)

        # ---Line 1---
        line = tk.Frame(frame, bg=TTBG)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        var = tk.Label(line, text='Multi-Crystal', font=TTF, fg=TTFG, bg=TTBG)
        var.pack()

        # ---Line 2---
        line = tk.LabelFrame(frame, text='Phases')
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        # Crystals List box
        self.listbox = tk.Listbox(line, selectmode=tk.EXTENDED, font=HF, width=60, bg=ety)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=5, pady=5)
        self.generate_listbox()

        # List box buttons
        right = tk.Frame(line)
        right.pack(side=tk.LEFT)

        # List of structures
        var = tk.OptionMenu(right, self.struct_list, *self.structure_list.list, command=self.fun_loadstruct)
        var.config(font=SF, width=20, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Browse button
        var = tk.Button(right, text='Browse', font=BF, command=self.fun_loadcif, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Scale
        line2 = tk.Frame(right)
        line2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=20)
        var = tk.Label(line2, text='Scale:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line2, textvariable=self.scale, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_scale)
        var.bind('<KP_Enter>', self.fun_scale)
        var = tk.Button(line2, text='Set', font=BF, command=self.fun_scale, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Remove button
        var = tk.Button(right, text='Remove', font=BF, command=self.fun_remove, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # ---Line 3---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Energy
        var = tk.Label(line, text='Energy (keV):', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.edge, *self.xr_edges, command=self.fun_edge)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.energy_kev, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Type
        types = ['X-Ray', 'Neutron', 'XRay Magnetic', 'Neutron Magnetic', 'XRay Resonant']
        var = tk.Label(line, text='Type:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.type, *types)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # Units
        xaxistypes = ['two-theta', 'd-spacing', 'Q']
        var = tk.Label(line, text='Units:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.powder_units, *xaxistypes)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # TwoTheta min
        var = tk.Label(line, text='Min:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.twotheta_min, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # TwoTheta max
        var = tk.Label(line, text='Max TwoTheta:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.twotheta_max, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # --- Line 4 ---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)

        var = tk.Button(line, text='Display Intensities', font=BF, command=self.fun_intensities, bg=btn2,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Plot Powder', font=BF, command=self.fun_powder, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

    def generate_listbox(self):
        """Add crytal info to listbox"""
        self.listbox.delete(0, tk.END)
        for xtl in self.xtls.crystal_list:
            name = xtl.name
            atoms = len(xtl.Structure.label)
            scale = xtl.scale
            txt = '%20s | Atoms: %4d | Scale: %s' % (name, atoms, scale)
            self.listbox.insert(tk.END, txt)

    def fun_loadcif(self):
        # root = Tk().withdraw() # from Tkinter
        defdir = os.path.join(os.path.dirname(__file__), 'Structures')
        # defdir = os.path.expanduser('~')
        filename = filedialog.askopenfilename(initialdir=defdir,
                                              filetypes=[('cif file', '.cif'), ('magnetic cif', '.mcif'),
                                                         ('All files', '.*')])  # from tkFileDialog
        if filename:
            xtl = Crystal(filename)
            self.xtls += xtl
            self.generate_listbox()

    def fun_loadstruct(self, event=None):
        """Load from structure_list"""
        if self.struct_list.get() in self.structure_list.list:
            xtl = getattr(self.structure_list, self.struct_list.get()).build()
            self.xtls += xtl
            self.generate_listbox()

    def fun_scale(self, event=None):
        """Set current crystal scale"""
        select = self.listbox.curselection()
        scale = self.scale.get()
        for idx in select:
            self.xtls.set_scale(idx, scale)
        self.generate_listbox()

    def fun_remove(self, event=None):
        """Remove selected crystal"""
        select = self.listbox.curselection()
        for idx in select:
            self.xtls.remove(idx)
        self.generate_listbox()

    def fun_edge(self, event=None):
        """X-ray edge option menu"""
        edge = self.edge.get()
        if self.edge.get() in self.xr_edges:
            idx = self.xr_edges.index(edge)
            self.energy_kev.set(self.xr_energies[idx])

    def set_scatter(self):
        """Set up scattering conditions"""
        scattering_type = self.type.get()
        energy_kev = self.energy_kev.get()
        min_two_theta = self.twotheta_min.get()
        max_two_theta = self.twotheta_max.get()
        powder_units = self.powder_units.get()
        self.xtls.setup_scatter(
            scattering_type=scattering_type,
            energy_kev=energy_kev,
            min_twotheta=min_two_theta,
            max_twotheta=max_two_theta,
            powder_units=powder_units
        )

    def fun_intensities(self, event=None):
        """Display intensities"""
        units = self.powder_units.get()
        self.set_scatter()
        string = self.xtls.print_all_reflections()
        StringViewer(string, 'Intensities')

    def fun_powder(self, event=None):
        """Plot powder pattern"""
        self.set_scatter()
        energy_kev = self.energy_kev.get()
        self.xtls.Plot.simulate_powder(energy_kev)
        plt.show()

