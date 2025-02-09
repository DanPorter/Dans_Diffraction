"""
Properties GUI
"""

import matplotlib.pyplot as plt
import numpy as np

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .. import functions_plotting as fp
from .basic_widgets import tk, StringViewer, SelectionBox, messagebox, topmenu
from .basic_widgets import (TF, BF, SF, LF, HF,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)


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

        # Variables
        self.zfraction = tk.DoubleVar(frame, 1)
        self.atoms = tk.StringVar(frame, ' '.join(atoms))
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

        var = tk.Button(line, text='Elements', font=BF, command=self.fun_element, bg=btn, activebackground=btn_active)
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

        var = tk.Button(line, text='Bond\nDistances', font=BF, command=self.fun_neighbours, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # ---Line 5---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, pady=5)

        var = tk.Button(line, text='X-Ray\nScattering Factor', font=BF, command=self.fun_xsf, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Atomic\nScattering Factor', font=BF, command=self.fun_asf, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='X-Ray\nInteractions', font=BF, command=self.fun_cxro, bg=btn,
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
        """Element button"""
        ele_list = ['%3s: %s' % (sym, nm) for sym, nm in fc.atom_properties(None, ['Element', 'Name'])]
        ele = self.atoms.get().split()
        cur_ele = ['%3s: %s' % (sym, nm) for sym, nm in fc.atom_properties(ele, ['Element', 'Name'])]
        choose = SelectionBox(
            parent=self.root,
            data_fields=ele_list,
            current_selection=cur_ele,
            multiselect=True,
            title='Select elements'
        ).show()
        ch_ele = [ele[:3].strip() for ele in choose]
        self.atoms.set(' '.join(ch_ele))

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

    def fun_neighbours(self):
        NeighborsGui(self.xtl)

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

    def fun_cxro(self):
        """Xray Interactions Button"""
        XrayInteractionsGui(self.xtl)


class XrayInteractionsGui:
    """
    Calculate X-Ray interactions with Matter
    """

    def __init__(self, xtl=None):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('X-Ray Interactions with Matter')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Crystal info
        name = xtl.name if xtl else 'Fe'
        formula = xtl.Properties.molname() if xtl else 'Fe'
        density = round(xtl.Properties.density(), 3) if xtl else 92.735

        # Variables
        self.chem_formula = tk.StringVar(frame, formula)
        self.chem_select = tk.StringVar(frame, 'Select Formula')
        self.density = tk.DoubleVar(frame, density)
        self.scan_type = tk.StringVar(frame, 'Photon Energy (keV)')
        self.scan_min = tk.DoubleVar(frame, 0.03)
        self.scan_max = tk.DoubleVar(frame, 15.0)
        self.scan_step = tk.DoubleVar(frame, 0.01)
        self.scan_units = tk.StringVar(frame, 'keV')
        self.grazing_angle = tk.DoubleVar(frame, 90.)
        self.slab_thickness = tk.DoubleVar(frame, 0.2)

        self.formulas = {
            name: (formula, density),
            'Silicon Nitide': ('SiN4', 3.44),
            'Mylar': ('C10H8O4', 1.4),
            'Sapphire': ('Al2O3', 3.98),
            'Silicon': ('Si', 2.33),
            'Beryllium': ('Be', 1.84),
            'Kapton': ('C22H10N2O5', 1.43),
            'Super Glue': ('C6H7NO2', 1.06),
            'Silver Paint': ('(2Ag)(C6H12O3)(C6H12O2)', 1.692),  # 50-75%Ag (10.49), 10-25% 2-methoxy-1-methylethyl acetate (0.962), 10-25% n-butyl acetate (0.882)
        }
        self.scan_types = {
            'Photon Energy (keV)': 'keV',
            'Photon Energy (eV)': 'eV',
            'Wavelength (Å)': 'Å',
            'Wavelength (nm)': 'nm'
        }

        # ---Menu---
        menu = {
            'Periodic Table': self.menu_info_table,
            'Unit Converter': self.menu_converter,
            'About': self.menu_about,
        }
        topmenu(self.root, menu)

        # ---Line 0---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)

        # Chemical Formula
        var = tk.Label(line, text='Chemical Formula:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.chem_formula, font=TF, width=20, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        #var.bind('<Return>', self.fun_wave2energy)
        #var.bind('<KP_Enter>', self.fun_wave2energy)
        var = tk.OptionMenu(line, self.chem_select, *self.formulas, command=self.fun_formulas)
        var.config(font=TF, width=15, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # Density
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)

        var = tk.Label(line, text='Density:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.density, font=TF, width=20, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text=' g cm^-3', font=TF)
        var.pack(side=tk.LEFT)

        # Energy
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)

        var = tk.OptionMenu(line, self.scan_type, *self.scan_types, command=self.fun_scantype)
        var.config(font=TF, width=20, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text='Min:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.scan_min, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Label(line, text='Max:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.scan_max, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Label(line, text='Step:', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.scan_step, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Label(line, textvariable=self.scan_units, font=TF)
        var.pack(side=tk.LEFT)

        # Grazing angle
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)

        var = tk.Label(line, text='Grazing angle:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='?', font=TF, command=self.button_help_angle, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=1)
        var = tk.Entry(line, textvariable=self.grazing_angle, font=TF, width=20, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text=' Deg', font=TF)
        var.pack(side=tk.LEFT)

        # slab thickness
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)

        var = tk.Label(line, text='Thickness:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='?', font=TF, command=self.button_help_thickness, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=1)
        var = tk.Entry(line, textvariable=self.slab_thickness, font=TF, width=20, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text=' μm', font=TF)
        var.pack(side=tk.LEFT)

        # Buttons
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, pady=5)

        var = tk.Button(line, text='Attenuation\nLength', height=2, font=BF, command=self.button_atten, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Thick Slab\nTransmission', font=BF, command=self.button_transmission, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Reflectivity', font=BF, command=self.button_reflectivity, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)

        var = tk.Button(line, text='Index of\nRefraction', font=BF, command=self.button_refraction, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        self.root.mainloop()

    def menu_info_table(self):
        from .periodic_table import PeriodTableGui
        PeriodTableGui()

    def menu_converter(self):
        """Open unit converter"""
        from .converter import UnitConverter
        UnitConverter()

    def menu_about(self):
        about = "Xray Interactions with Matter\nBy Dan Porter 2025\n\n"
        about += "Inspired by CXRO: https://henke.lbl.gov/optical_constants/\n\n"
        messagebox.showinfo('X-Ray Interactions with Matter', about)

    def get_scan(self):
        scan_type = self.scan_type.get()
        scan_min = self.scan_min.get()
        scan_max = self.scan_max.get()
        scan_step = self.scan_step.get()

        # Convert to keV
        if scan_type == 'Photon Energy (eV)':
            scan_min = scan_min / 1000.
            scan_max = scan_max / 1000.
            scan_step = scan_step / 1000.
        elif scan_type == 'Wavelength (Å)':
            scan_min = fc.wave2energy(scan_min)
            scan_max = fc.wave2energy(scan_max)
            scan_step = fc.wave2energy(scan_step)
        elif scan_type == 'Wavelength (nm)':
            scan_min = fc.wave2energy(scan_min * 10.)
            scan_max = fc.wave2energy(scan_max * 10.)
            scan_step = fc.wave2energy(scan_step * 10.)
        return np.arange(scan_min, scan_max + scan_step, scan_step)

    def fun_formulas(self, event=None):
        chemname = self.chem_select.get()
        if chemname in self.formulas:
            formula, density = self.formulas[chemname]
            self.chem_formula.set(formula)
            self.density.set(density)

    def fun_scantype(self, event=None):
        scan_type = self.scan_type.get()
        old_units = self.scan_units.get()

        scan_min = self.scan_min.get()
        scan_max = self.scan_max.get()
        scan_step = self.scan_step.get()

        # Convert to keV
        if old_units == 'eV':
            scan_min = scan_min / 1000.
            scan_max = scan_max / 1000.
            scan_step = scan_step / 1000.
        elif old_units == 'Å':
            scan_min = fc.wave2energy(scan_min)
            scan_max = fc.wave2energy(scan_max)
            scan_step = fc.wave2energy(scan_step)
        elif old_units == 'nm':
            scan_min = fc.wave2energy(scan_min * 10.)
            scan_max = fc.wave2energy(scan_max * 10.)
            scan_step = fc.wave2energy(scan_step * 10.)

        # Convert to units
        if scan_type == 'Photon Energy (eV)':
            scan_min = scan_min * 1000.
            scan_max = scan_max * 1000.
            scan_step = scan_step * 1000.
        elif scan_type == 'Wavelength (Å)':
            scan_min = fc.energy2wave(scan_min)
            scan_max = fc.energy2wave(scan_max)
            scan_step = fc.energy2wave(scan_step)
        elif scan_type == 'Wavelength (nm)':
            scan_min = fc.wave2energy(scan_min) / 10.
            scan_max = fc.wave2energy(scan_max) / 10.
            scan_step = fc.wave2energy(scan_step) / 10.

        self.scan_min.set(scan_min)
        self.scan_max.set(scan_max)
        self.scan_step.set(scan_step)
        self.scan_units.set(self.scan_types[scan_type])

    def button_help_angle(self):
        messagebox.showinfo(
            parent=self.root,
            title='X-Ray Interactions',
            message='Grazing angle: angle from surface to incident beam, in degrees\n' +
            'Typical values:\n\tTransmission: 90 Deg\n\tReflectivity: 0.2 Deg',
        )

    def button_help_thickness(self):
        messagebox.showinfo(
            parent=self.root,
            title='X-Ray Interactions',
            message='Thickness: distance beam travels through material in Transmission, in microns.',
        )

    def button_atten(self):
        formula = self.chem_formula.get()
        density = self.density.get()
        angle = self.grazing_angle.get()
        energy_range = self.get_scan()
        fp.plot_xray_attenuation_length(formula, density, energy_range, angle)
        plt.show()

    def button_transmission(self):
        formula = self.chem_formula.get()
        density = self.density.get()
        thickness = self.slab_thickness.get()
        energy_range = self.get_scan()
        fp.plot_xray_transmission(formula, density, energy_range, thickness)
        plt.show()

    def button_reflectivity(self):
        formula = self.chem_formula.get()
        density = self.density.get()
        angle = self.grazing_angle.get()
        energy_range = self.get_scan()
        fp.plot_xray_reflectivity(formula, density, energy_range, angle)
        plt.show()

    def button_refraction(self):
        formula = self.chem_formula.get()
        density = self.density.get()
        energy_range = self.get_scan()
        fp.plot_xray_refractive_index(formula, density, energy_range)
        plt.show()


class NeighborsGui:
    """
    Calculate bond distances
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Bond Distances')
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
        self.atom_list = [self._atom_str(atom) for atom in self.xtl.Structure]
        self.structure_index = tk.IntVar(frame, 0)
        self.atom_str = tk.StringVar(frame, self.atom_list[0])
        self.radius = tk.DoubleVar(frame, 2.5)
        self.exchange_type = tk.StringVar(frame, 'O')
        self.bond_angle = tk.DoubleVar(frame, 90)
        self.search_in_cell = tk.BooleanVar(frame, True)
        self.group_neighbors = tk.BooleanVar(frame, True)

        # Atom index
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)
        var = tk.Label(line, text='Atom Index:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.structure_index, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.atom_update)
        var.bind('<KP_Enter>', self.atom_update)
        var = tk.Label(line, textvariable=self.atom_str, font=TF, width=30, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='Select', font=TF, command=self.button_select_atom,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=1)

        # Radius
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)
        var = tk.Label(line, text='Radius:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.radius, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='?', font=TF, command=self.button_radius,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=1)

        # Exchange type
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)
        var = tk.Label(line, text='Exchange type:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.exchange_type, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # bond angle
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)
        var = tk.Label(line, text='Bond Angle:', width=20, font=TF, anchor=tk.E)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.bond_angle, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text=' Deg', font=TF)
        var.pack(side=tk.LEFT)

        # Options
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.X, pady=5)
        var = tk.Checkbutton(line, text='Search in cell', variable=self.search_in_cell, font=SF)
        var.pack(side=tk.LEFT, padx=6)
        var = tk.Checkbutton(line, text='Group Neighbors', variable=self.group_neighbors, font=SF)
        var.pack(side=tk.LEFT, padx=6)

        # Buttons
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, expand=tk.TRUE, pady=5)
        var = tk.Button(line, text='All Bond\nDistances', font=BF, command=self.button_allbonds, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='Atomic\nNeighbours', font=BF, command=self.button_neighbours, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='Exchange\nPaths', font=BF, command=self.button_exchange, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='Plot Exchange\nPaths', font=BF, command=self.button_plot_exchange, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

    def _atom_str(self, atom):
        s = '%4s [%2s] (%5.2f,%5.2f,%5.2f)'
        return s % (atom.label, atom.type, atom.u, atom.v, atom.w)

    def atom_update(self, event=None):
        idx = self.structure_index.get()
        self.atom_str.set(self.atom_list[idx])

    def button_select_atom(self):
        choose = SelectionBox(self.root, self.atom_list, multiselect=False, title='Select atom site').show()
        if choose:
            idx = self.atom_list.index(choose[0])
            self.structure_index.set(idx)
            self.atom_str.set(choose[0])

    def button_radius(self):
        messagebox.showinfo(
            parent=self.root,
            title='Radius',
            message='Nearest neighbors: search radius about central atom [2.5A]\n' +
            'Exchange paths: typical value [7.0A]'
        )

    def button_allbonds(self):
        out = ''
        # get atom sites
        for atom_str in self.atom_list:
            idx = self.atom_list.index(atom_str)
            bond_str = self.xtl.Properties.atomic_neighbours(
                structure_index=idx,
                radius=self.radius.get(),
                return_str=True
            )
            out += '%3d %s\n%s\n\n' % (idx, atom_str, bond_str)
        StringViewer(out, title='%s All Bonds' % self.xtl.name, width=60)

    def button_neighbours(self):
        out = self.xtl.Properties.atomic_neighbours(
            structure_index=self.structure_index.get(),
            radius=self.radius.get(),
            disp=True,
            return_str=True
        )
        StringViewer(out, title='%s %s' % (self.xtl.name, self.atom_str.get()), width=60)

    def button_exchange(self):
        exchange_paths, exchange_distances, outstr = self.xtl.Properties.exchange_paths(
            cen_idx=self.structure_index.get(),
            nearest_neighbor_distance=self.radius.get(),
            exchange_type=self.exchange_type.get(),
            bond_angle=self.bond_angle.get(),
            search_in_cell=self.search_in_cell.get(),
            group_neighbors=self.group_neighbors.get(),
            disp=True,
            return_str=True
        )
        StringViewer(outstr, title='%s %s' % (self.xtl.name, self.atom_str.get()), width=100)

    def button_plot_exchange(self):
        self.xtl.Plot.plot_exchange_paths(
            cen_idx=self.structure_index.get(),
            nearest_neighbor_distance=self.radius.get(),
            exchange_type=self.exchange_type.get(),
            bond_angle=self.bond_angle.get(),
            search_in_cell=self.search_in_cell.get(),
            group_neighbors=self.group_neighbors.get()
        )

