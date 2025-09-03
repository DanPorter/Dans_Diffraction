"""
Scattering GUI
"""

import sys, os
import re

import matplotlib.pyplot as plt
import numpy as np

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .. import fdmnes_checker
from .basic_widgets import tk, StringViewer, topmenu, messagebox
from .basic_widgets import (TF, BF, SF, LF, HF, MF,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)


class ScatteringGui:
    """
    Simulate scattering of various forms
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Scattering %s' % xtl.name)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol
        )

        # ---Menu---
        menu = {
            'Info': {
                'Crystal': self.menu_info_crystal,
                'Scattering Settings': self.menu_info_scattering,
            },
            'Defaults': {
                'Supernova': self.fun_supernova,
                'Wish': self.fun_wish,
                'I16': self.fun_i16,
            }
        }
        topmenu(self.root, menu)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Variables
        self.radiation = tk.StringVar(frame, 'X-Ray')
        self.check_magnetic = tk.BooleanVar(frame, False)
        self.energy = tk.DoubleVar(frame, 8.0)
        self.wavelength = tk.DoubleVar(frame, 1.5498)
        self.energy_unit = tk.StringVar(frame, 'keV')
        self.wavelength_unit = tk.StringVar(frame, '\u212B')
        self.edge = tk.StringVar(frame, 'Edge')
        self.orientation = tk.StringVar(frame, 'None')
        self.direction_h = tk.IntVar(frame, 0)
        self.direction_k = tk.IntVar(frame, 0)
        self.direction_l = tk.IntVar(frame, 1)
        self.theta_offset = tk.DoubleVar(frame, 0.0)
        self.theta_min = tk.DoubleVar(frame, -180.0)
        self.theta_max = tk.DoubleVar(frame, 180.0)
        self.twotheta_min = tk.DoubleVar(frame, -180.0)
        self.twotheta_max = tk.DoubleVar(frame, 180.0)
        self.powder_units = tk.StringVar(frame, 'Two-Theta')
        self.powderaverage = tk.BooleanVar(frame, True)
        self.powder_width = tk.DoubleVar(frame, 0.01)
        self.hkl_check = tk.StringVar(frame, '0 0 1')
        self.hkl_result = tk.StringVar(frame, 'I:%10.0f TTH:%8.2f' % (0, 0))
        self.close_twotheta = tk.DoubleVar(frame, 2)
        self.close_angle = tk.DoubleVar(frame, 10)
        self.val_i = tk.IntVar(frame, 0)
        self.cmin = tk.StringVar(frame, '')
        self.cmax = tk.StringVar(frame, '')
        self.polarisation = tk.StringVar(frame, '1 0 0')
        self.average_polarisation = tk.BooleanVar(frame, True)
        self.mag_form_factor = tk.BooleanVar(frame, True)

        # radiation
        radiations = ['X-Ray', 'Neutron', 'Electron']
        energy_units = ['keV', 'eV', 'meV']
        wavelength_units = [u'\u212B', 'nm']

        # X-ray edges:
        self.xr_edges, self.xr_energies = self.xtl.Properties.xray_edges()
        self.xr_edges.insert(0, 'Cu Ka')
        self.xr_edges.insert(1, 'Mo Ka')
        self.xr_energies.insert(0, fg.Cu)
        self.xr_energies.insert(1, fg.Mo)

        # --- Top Buttons ---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)

        var = tk.Label(line, text='Scattering', font=LF)
        var.pack(side=tk.LEFT)

        # ---Settings---
        frm = tk.Frame(frame)
        frm.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)
        box = tk.LabelFrame(frm, text='Settings')
        box.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        # Radiation
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Radiation:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.radiation, *radiations, command=self.fun_radiation)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        if self.xtl.Structure.ismagnetic():
            var = tk.Checkbutton(line, text='Magnetic', variable=self.check_magnetic, font=SF)
            var.pack(side=tk.LEFT, padx=6)

        # Energy
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Energy:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.edge, *self.xr_edges, command=self.fun_edge)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.energy, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_energy)
        var.bind('<KP_Enter>', self.fun_energy)
        var = tk.OptionMenu(line, self.energy_unit, *energy_units, command=self.fun_wavelength)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # Wavelength
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text=u'Wavelength:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.wavelength, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_wavelength)
        var.bind('<KP_Enter>', self.fun_wavelength)
        var = tk.OptionMenu(line, self.wavelength_unit, *wavelength_units, command=self.fun_energy)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # Scattering Type
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Units
        xaxistypes = ['two-theta', 'd-spacing', 'Q']
        var = tk.Label(line, text='Units:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.powder_units, *xaxistypes)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # Orientation
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Geometry:', font=SF)
        var.pack(side=tk.LEFT)
        orients = ['None', 'Reflection', 'Transmission']
        var = tk.OptionMenu(line, self.orientation, *orients)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # Direction
        var = tk.Label(line, text='Direction:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.direction_h, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.direction_k, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.direction_l, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Theta offset
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Offset:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.theta_offset, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Theta min
        var = tk.Label(line, text='Min Theta:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.theta_min, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Theta max
        var = tk.Label(line, text='Max Theta:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.theta_max, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # TwoTheta min
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Min TwoTheta:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.twotheta_min, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # TwoTheta max
        var = tk.Entry(line, textvariable=self.twotheta_max, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.RIGHT)
        var = tk.Label(line, text='Max TwoTheta:', font=SF)
        var.pack(side=tk.RIGHT)

        # Powder width
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Powder peak width:', font=SF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.powder_width, font=TF, width=5, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Powder average tickbox
        var = tk.Checkbutton(line, text='Powder average', variable=self.powderaverage, font=SF)
        var.pack(side=tk.LEFT, padx=6)

        # Polarisation
        if self.xtl.Structure.ismagnetic():
            line = tk.Frame(box)
            line.pack(side=tk.TOP, fill=tk.X, pady=5)

            var = tk.Label(line, text='Polarisation:', font=SF)
            var.pack(side=tk.LEFT, padx=3)
            var = tk.Entry(line, textvariable=self.polarisation, font=TF, width=5, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)
            var = tk.Checkbutton(line, text='average', variable=self.average_polarisation, font=SF)
            var.pack(side=tk.LEFT, padx=6)
            var = tk.Checkbutton(line, text='Mag. form factor', variable=self.mag_form_factor, font=SF)
            var.pack(side=tk.LEFT, padx=6)

        # ---Intensities---
        right = tk.Frame(frm)
        right.pack(side=tk.LEFT, fill=tk.BOTH)
        box = tk.LabelFrame(right, text='Intensities')
        box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)

        var = tk.Button(line, text='Display Intensities', font=BF, command=self.fun_intensities, bg=btn2,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Plot Powder', font=BF, command=self.fun_powder, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # ---hkl check---
        hklbox = tk.LabelFrame(box, text='HKL Check')
        hklbox.pack(side=tk.TOP, fill=tk.X, pady=5)
        line = tk.Frame(hklbox)
        line.pack(side=tk.TOP)
        var = tk.Entry(line, textvariable=self.hkl_check, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_hklcheck)
        var.bind('<KP_Enter>', self.fun_hklcheck)
        var = tk.Button(line, text='sym', font=TF, command=self.fun_hklsym, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)
        var = tk.Label(line, textvariable=self.hkl_result, font=TF, width=22)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='Check HKL', font=BF, command=self.fun_hklcheck, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        line = tk.Frame(hklbox)
        line.pack(side=tk.TOP)
        var = tk.Button(line, text='Atomic Contributions', font=BF, command=self.fun_hklatom, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)
        var = tk.Button(line, text='Symmetry Contributions', font=BF, command=self.fun_hklsymmetry, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        line = tk.Frame(hklbox)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text=u'\u0394 Two Theta:', font=SF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.close_twotheta, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text=u'\u0394 Angle:', font=SF)
        var.pack(side=tk.LEFT, padx=3)
        var = tk.Entry(line, textvariable=self.close_angle, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Button(line, text='Close Refs', font=TF, command=self.fun_closerefs, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        # ---HKL Planes---
        box = tk.LabelFrame(right, text='Reciprocal Space Planes')
        box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        side = tk.Frame(box)
        side.pack(side=tk.LEFT, pady=5)

        # i value
        var = tk.Label(side, text='i:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(side, textvariable=self.val_i, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # directions
        vframe = tk.Frame(side)
        vframe.pack(side=tk.LEFT, padx=3)
        var = tk.Button(vframe, text='HKi', font=BF, command=self.fun_hki, width=5, bg=btn, activebackground=btn_active)
        var.pack()
        var = tk.Button(vframe, text='HiL', font=BF, command=self.fun_hil, width=5, bg=btn, activebackground=btn_active)
        var.pack()

        vframe = tk.Frame(side)
        vframe.pack(side=tk.LEFT)
        var = tk.Button(vframe, text='iKL', font=BF, command=self.fun_ikl, width=5, bg=btn, activebackground=btn_active)
        var.pack()
        var = tk.Button(vframe, text='HHi', font=BF, command=self.fun_hhi, width=5, bg=btn, activebackground=btn_active)
        var.pack()

        side = tk.Frame(box)
        side.pack(side=tk.LEFT, padx=5, pady=5)
        frm = tk.Frame(side)
        frm.pack(side=tk.TOP)
        var = tk.Button(frm, text='3D', font=BF, command=self.fun_sf3d, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Arbitary Cut', font=BF, command=self.fun_cut, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)
        # clim
        frm = tk.Frame(side, relief=tk.RIDGE)
        frm.pack(side=tk.TOP, pady=10)
        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text='Clim:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.cmin, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_clim)
        var.bind('<KP_Enter>', self.fun_clim)
        var = tk.Entry(line, textvariable=self.cmax, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_clim)
        var.bind('<KP_Enter>', self.fun_clim)
        var = tk.Button(line, text='Update', font=TF, command=self.fun_clim, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        # ---X-ray Magnetic scattering---
        box = tk.Frame(right)
        box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)
        var = tk.Button(box, text='X-Ray Magnetic Scattering', font=BF, command=self.fun_rxs, bg=btn,
                        activebackground=btn_active)
        var.pack(fill=tk.X)

    ###################################################################################
    ################################# MENU ############################################
    ###################################################################################
    def menu_info_crystal(self):
        """Crystal info"""
        string = '%s\n%s' % (self.xtl.filename, self.xtl.info())
        StringViewer(string, self.xtl.name, width=60)

    def menu_info_scattering(self):
        """Scattering info"""
        self.fun_get()
        string = '%s\n%s' % (self.xtl.name, self.xtl.Scatter)
        StringViewer(string, self.xtl.name, width=60)

    ###################################################################################
    ############################## FUNCTIONS ##########################################
    ###################################################################################
    def fun_set(self):
        """"Set gui parameters from crystal"""

        # self.energy_kev.set(8)
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
        # scatering type
        radiation = self.radiation.get()
        magnetic = self.check_magnetic.get()
        av_pol = self.average_polarisation.get()
        if radiation == 'X-Ray':
            if magnetic and av_pol:
                scat._scattering_type = 'xray magnetic'
            elif magnetic:
                scat._scattering_type = 'xray polarised'
            else:
                scat._scattering_type = 'xray'
        elif radiation == 'Neutron':
            if magnetic and av_pol:
                scat._scattering_type = 'neutron magnetic'
            elif magnetic:
                scat._scattering_type = 'neutron polarised'
            else:
                scat._scattering_type = 'neutron'
        else:
            scat._scattering_type = 'electron'

        scat._energy_kev = fc.wave2energy(self.get_wavelength())
        scat._scattering_theta_offset = self.theta_offset.get()
        scat._scattering_min_theta = self.theta_min.get()
        scat._scattering_max_theta = self.theta_max.get()
        scat._scattering_min_twotheta = self.twotheta_min.get()
        scat._scattering_max_twotheta = self.twotheta_max.get()
        scat._powder_units = self.powder_units.get()
        scat._polarisation_vector_incident = fg.str2array(self.polarisation.get())
        scat._use_magnetic_form_factor = self.mag_form_factor.get()

        if self.orientation.get() == 'Reflection':
            scat._scattering_specular_direction[0] = self.direction_h.get()
            scat._scattering_specular_direction[1] = self.direction_k.get()
            scat._scattering_specular_direction[2] = self.direction_l.get()
        elif self.orientation.get() == 'Transmission':
            scat._scattering_parallel_direction[0] = self.direction_h.get()
            scat._scattering_parallel_direction[1] = self.direction_k.get()
            scat._scattering_parallel_direction[2] = self.direction_l.get()

    def fun_get_hkl(self):
        """Get quick check hkl"""
        self.fun_get()
        hkl = self.hkl_check.get()
        return fg.str2array(hkl)

    def get_energy(self):
        """Return energy in keV according to unit"""
        energy = self.energy.get()
        unit = self.energy_unit.get()
        if unit == 'eV':
            energy = energy / 1000.
        elif unit == 'meV':
            energy = energy / 1.0e6
        return energy

    def get_wavelength(self):
        """Return wavelength in A according to unit"""
        wavelength = self.wavelength.get()
        unit = self.wavelength_unit.get()
        if unit == 'nm':
            wavelength = wavelength / 10.
        return wavelength

    def set_energy(self, energy_kev):
        """set energy according to unit"""
        energy_unit = self.energy_unit.get()
        if energy_unit == 'meV':
            energy = energy_kev * 1e6
        elif energy_unit == 'eV':
            energy = energy_kev * 1e3
        else:
            energy = energy_kev
        self.energy.set(round(energy, 4))

    def set_wavelength(self, wavelength_a):
        """set wavelength according to unit"""
        wavelength_unit = self.wavelength_unit.get()
        if wavelength_unit == 'nm':
            wavelength = wavelength_a / 10.
        else:
            wavelength = wavelength_a
        self.wavelength.set(round(wavelength, 4))

    ###################################################################################
    ############################### DEFAULTS ##########################################
    ###################################################################################

    def fun_i16(self):
        """"Add I16 parameters"""

        self.radiation.set('X-Ray')
        self.energy_unit.set('keV')
        self.set_energy(8)
        self.fun_energy()
        self.edge.set('Edge')
        self.powder_units.set('Two-Theta')
        self.powderaverage.set(False)
        self.orientation.set('Reflection')
        self.theta_offset.set(0.0)
        self.theta_min.set(-20.0)
        self.theta_max.set(150.0)
        self.twotheta_min.set(0.0)
        self.twotheta_max.set(130.0)

    def fun_wish(self):
        """"Add Wish parameters"""

        self.radiation.set('Neutron')
        self.energy_unit.set('meV')
        self.set_wavelength(0.7)
        self.fun_wavelength()
        self.edge.set('Edge')
        self.powder_units.set('d-spacing')
        self.orientation.set('None')
        self.theta_offset.set(0.0)
        self.theta_min.set(-180.0)
        self.theta_max.set(180.0)
        self.twotheta_min.set(10.0)
        self.twotheta_max.set(170.0)

    def fun_supernova(self):
        """Add SuperNova parameters"""

        self.radiation.set('X-Ray')
        idx = self.xr_edges.index('Mo Ka')
        self.edge.set('Mo Ka')
        self.set_energy(self.xr_energies[idx])
        self.fun_energy()
        self.powder_units.set('Two-Theta')
        self.orientation.set('None')
        self.theta_offset.set(0.0)
        self.theta_min.set(-180.0)
        self.theta_max.set(180.0)
        self.twotheta_min.set(-170.0)
        self.twotheta_max.set(170.0)

    ###################################################################################
    ################################ BUTTONS ##########################################
    ###################################################################################

    def fun_radiation(self, event=None):
        """Set radiation"""
        radiation = self.radiation.get()
        if radiation == 'Neutron':
            self.energy_unit.set('meV')
        elif radiation == 'Electron':
            self.energy_unit.set('eV')
            self.check_magnetic.set(False)
        else:
            self.energy_unit.set('keV')
        self.fun_wavelength()

    def fun_energy(self, event=None):
        """Set wavelength"""
        radiation = self.radiation.get()
        energy_kev = self.get_energy()
        # calculate wavelength in A for radiation
        if radiation == 'Electron':
            wavelength_a = fc.electron_wavelength(energy_kev * 1000)
        elif radiation == 'Neutron':
            wavelength_a = fc.neutron_wavelength(energy_kev * 1e6)
        else:
            wavelength_a = fc.energy2wave(energy_kev)
        self.set_wavelength(wavelength_a)

    def fun_wavelength(self, event=None):
        """Set energy"""
        radiation = self.radiation.get()
        wavelength_a = self.get_wavelength()
        # calculate energy in keV for radiation
        if radiation == 'Electron':
            energy_kev = fc.electron_energy(wavelength_a) / 1000
        elif radiation == 'Neutron':
            energy_kev = fc.neutron_energy(wavelength_a) / 1e6
        else:
            energy_kev = fc.wave2energy(wavelength_a)
        self.set_energy(energy_kev)

    def fun_edge(self, event=None):
        """X-ray edge option menu"""
        edge = self.edge.get()
        if self.edge.get() in self.xr_edges:
            idx = self.xr_edges.index(edge)
            self.set_wavelength(fc.energy2wave(self.xr_energies[idx]))
            self.fun_wavelength()

    def fun_hklcheck(self, event=None):
        """"Show single hkl intensity"""

        hkl = self.fun_get_hkl()
        I = self.xtl.Scatter.intensity(hkl)

        unit = self.powder_units.get()
        wavelength = self.get_wavelength()
        tth = self.xtl.Cell.tth(hkl, wavelength_a=wavelength)

        if unit.lower() in ['tth', 'angle', 'twotheta', 'theta', 'two-theta']:
            self.hkl_result.set('I:%10.0f TTH:%8.2f' % (I, tth))
        elif unit.lower() in ['d', 'dspace', 'd-spacing', 'dspacing']:
            q = fc.calqmag(tth, wavelength_a=wavelength)
            d = fc.q2dspace(q)
            self.hkl_result.set(u'I:%10.0f   d:%8.2f \u212B' % (I, d))
        else:
            q = fc.calqmag(tth, wavelength_a=wavelength)
            self.hkl_result.set(u'I:%8.0f   Q:%8.2f \u212B\u207B\u00B9' % (I, q))

    def fun_hklsym(self):
        """Print symmetric reflections"""
        hkl = self.fun_get_hkl()
        out = self.xtl.Scatter.print_symmetric_reflections(hkl)
        StringViewer(out, 'Symmetric Reflections %s' % hkl)

    def fun_hklatom(self):
        """Print atomic contributions"""
        hkl = self.fun_get_hkl()
        out = self.xtl.Scatter.print_atomic_contributions(hkl)
        StringViewer(out, 'Atomic Contributions %s' % hkl, width=80)

    def fun_hklsymmetry(self):
        """Print symmetry contributions"""
        hkl = self.fun_get_hkl()
        out = self.xtl.Scatter.print_symmetry_contributions(hkl)
        StringViewer(out, 'Symmetric Contributions %s' % hkl, width=120)

    def fun_closerefs(self):
        hkl = self.fun_get_hkl()
        close_tth = self.close_twotheta.get()
        close_ang = self.close_angle.get()
        out = self.xtl.Scatter.find_close_reflections(hkl, max_twotheta=close_tth, max_angle=close_ang)
        StringViewer(out, 'Close Reflections %s' % hkl)

    def fun_intensities(self):
        """Display intensities"""

        self.fun_get()
        if self.orientation.get() == 'Reflection':
            string = self.xtl.Scatter.print_ref_reflections(min_intensity=-1, max_intensity=None)
        elif self.orientation.get() == 'Transmission':
            string = self.xtl.Scatter.print_tran_reflections(min_intensity=-1, max_intensity=None)
        else:
            units = self.powder_units.get()
            string = self.xtl.Scatter.print_all_reflections(min_intensity=-1, max_intensity=None, units=units)
        StringViewer(string, 'Intensities %s' % self.xtl.name)

    def fun_powder(self):
        """Plot Powder"""
        self.fun_get()
        wavelength_a = self.get_wavelength()
        energy_kev = fc.wave2energy(wavelength_a)
        pow_avg = self.powderaverage.get()
        pow_wid = self.powder_width.get()

        self.xtl.Plot.simulate_powder(energy_kev=energy_kev, peak_width=pow_wid, powder_average=pow_avg)
        plt.show()

    def fun_hki(self):
        """Plot hki plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hk0(i, peak_width=0.2)
        self.fun_clim()
        plt.show()

    def fun_hil(self):
        """Plot hil plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_h0l(i, peak_width=0.2)
        self.fun_clim()
        plt.show()

    def fun_ikl(self):
        """Plot ikl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_0kl(i, peak_width=0.2)
        self.fun_clim()
        plt.show()

    def fun_hhi(self):
        """Plot hhl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hhl(i, peak_width=0.2)
        self.fun_clim()
        plt.show()

    def fun_cut(self):
        self.fun_get()
        ArbitaryCutGui(self.xtl)

    def fun_sf3d(self):
        self.fun_get()
        self.xtl.Plot.plot_3Dintensity()
        plt.show()

    def fun_clim(self, event=None):
        """Update clim"""
        try:
            cmin = float(self.cmin.get())
        except ValueError:
            cmin = None
        try:
            cmax = float(self.cmax.get())
        except ValueError:
            cmax = None
        plt.gca()
        plt.clim(cmin, cmax)

    def fun_rxs(self):
        ResonantXrayGui(self)


class ArbitaryCutGui:
    """
    Simulate scattering of various forms
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Reciprocal Space Viewer %s' % xtl.name)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        self.hkl_plane_x = tk.StringVar(frame, '1 0 0')
        self.hkl_plane_y = tk.StringVar(frame, '0 1 0')
        self.hkl_plane_c = tk.StringVar(frame, '0 0 0')
        self.hkl_plane_q = tk.DoubleVar(frame, 4)
        self.hkl_plane_width = tk.DoubleVar(frame, 0.05)
        self.hkl_plane_bkg = tk.DoubleVar(frame, 0)
        self.hkl_plane_peak = tk.DoubleVar(frame, 0.2)
        self.lattice_points = tk.BooleanVar(frame, False)
        self.cmin = tk.StringVar(frame, '')
        self.cmax = tk.StringVar(frame, '')

        # --- Title ---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Arbitary HKL Cut', font=LF)
        var.pack(side=tk.LEFT)

        # --- Entry ---
        frm = tk.Frame(frame)
        frm.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text='x:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.hkl_plane_x, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text='y:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.hkl_plane_y, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text='c:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.hkl_plane_c, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text=u'Max Q [\u212B\u207B\u00B9]:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.hkl_plane_q, font=TF, width=4, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text='Cut width:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.hkl_plane_width, font=TF, width=4, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text='Background:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.hkl_plane_bkg, font=TF, width=4, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(line, text='Peak width:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.hkl_plane_peak, font=TF, width=4, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Checkbutton(line, text='Lattice points', variable=self.lattice_points, font=SF)
        var.pack(side=tk.LEFT, padx=5)

        # --- CLIM ---
        frm = tk.LabelFrame(frame, text='Colours', relief=tk.RIDGE)
        frm.pack(side=tk.TOP, pady=10)
        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Label(line, text='Clim:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.cmin, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_clim)
        var.bind('<KP_Enter>', self.fun_clim)
        var = tk.Entry(line, textvariable=self.cmax, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_clim)
        var.bind('<KP_Enter>', self.fun_clim)
        var = tk.Button(line, text='Update', font=TF, command=self.fun_clim, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        # --- Buttons ---
        frm = tk.Frame(frame)
        frm.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        line = tk.Frame(frm)
        line.pack(side=tk.TOP)
        var = tk.Button(line, text='Generate\nCut', font=TF, command=self.fun_generate, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        var = tk.Button(line, text='Generate\nEnvelope', font=TF, command=self.fun_envelope, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        var = tk.Button(line, text='Generate\n3D Lattice', font=TF, command=self.fun_3dlattice, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        var = tk.Button(line, text='Show\nCoverage', font=TF, command=self.fun_ewald, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

    def fun_clim(self, event=None):
        """Update clim"""
        try:
            cmin = float(self.cmin.get())
        except ValueError:
            cmin = None
        try:
            cmax = float(self.cmax.get())
        except ValueError:
            cmax = None
        plt.gca()
        plt.clim(cmin, cmax)

    def fun_generate(self):
        """Run code"""
        self.xtl.Plot.simulate_intensity_cut(
            x_axis=fg.str2array(self.hkl_plane_x.get()),
            y_axis=fg.str2array(self.hkl_plane_y.get()),
            centre=fg.str2array(self.hkl_plane_c.get()),
            q_max=self.hkl_plane_q.get(),
            cut_width=self.hkl_plane_width.get(),
            background=self.hkl_plane_bkg.get(),
            peak_width=self.hkl_plane_peak.get()
        )
        self.fun_clim()
        if self.lattice_points.get():
            self.xtl.Plot.axis_reciprocal_lattice_points(
                axes=None,
                x_axis=fg.str2array(self.hkl_plane_x.get()),
                y_axis=fg.str2array(self.hkl_plane_y.get()),
                centre=fg.str2array(self.hkl_plane_c.get()),
                q_max=self.hkl_plane_q.get(),
                cut_width=self.hkl_plane_width.get()
            )
        plt.show()

    def fun_envelope(self):
        """Run code"""
        self.xtl.Plot.simulate_envelope_cut(
            x_axis=fg.str2array(self.hkl_plane_x.get()),
            y_axis=fg.str2array(self.hkl_plane_y.get()),
            centre=fg.str2array(self.hkl_plane_c.get()),
            q_max=self.hkl_plane_q.get(),
            background=self.hkl_plane_bkg.get(),
            pixels=301,
        )
        self.fun_clim()
        if self.lattice_points.get():
            self.xtl.Plot.axis_reciprocal_lattice_points(
                axes=None,
                x_axis=fg.str2array(self.hkl_plane_x.get()),
                y_axis=fg.str2array(self.hkl_plane_y.get()),
                centre=fg.str2array(self.hkl_plane_c.get()),
                q_max=self.hkl_plane_q.get(),
                cut_width=self.hkl_plane_width.get()
            )
        plt.show()

    def fun_ewald(self):
        """Button plot ewald coverage"""
        self.xtl.Plot.simulate_ewald_coverage(
            energy_kev=self.xtl.Scatter.get_energy(),
            sample_normal=fg.str2array(self.hkl_plane_y.get()),
            sample_para=fg.str2array(self.hkl_plane_x.get()),
            cut_width=self.hkl_plane_width.get(),
            peak_width=self.hkl_plane_peak.get()
        )
        plt.show()

    def fun_3dlattice(self):
        """Button plot 3D lattice"""
        self.xtl.Plot.plot_3Dlattice(
            q_max=self.hkl_plane_q.get(),
            x_axis=fg.str2array(self.hkl_plane_x.get()),
            y_axis=fg.str2array(self.hkl_plane_y.get()),
            centre=fg.str2array(self.hkl_plane_c.get()),
            cut_width=self.hkl_plane_width.get()
        )
        plt.show()


class ResonantXrayGui:
    """
    Simulate scattering of various forms
    """

    def __init__(self, parent: ScatteringGui):
        self.parent = parent
        self.xtl = parent.xtl
        # Create Tk inter instance
        self.root = tk.Toplevel(parent.root)
        self.root.wm_title('REXS')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol
        )

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        self.energy_kev = tk.DoubleVar(frame, round(parent.get_energy(), 4))
        self.wavelenth_a = tk.DoubleVar(frame, round(parent.get_wavelength(), 4))
        self.edge = tk.StringVar(frame, parent.edge.get())
        self.hkl_magnetic = tk.StringVar(frame, '0 0 1')
        self.azim_zero = tk.StringVar(frame, '1 0 0')
        self.isres = tk.BooleanVar(frame, True)
        self.psival = tk.DoubleVar(frame, 0.0)
        self.polval = tk.StringVar(frame, u'\u03c3-\u03c0')
        self.resF0 = tk.DoubleVar(frame, 0.0)
        self.resF1 = tk.DoubleVar(frame, 1.0)
        self.resF2 = tk.DoubleVar(frame, 0.0)
        self.magresult = tk.StringVar(frame, 'I = --')

        # --- Title ---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='X-Ray Magnetic Scattering', font=LF)
        var.pack(side=tk.LEFT)

        # --- Energy ---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Energy (keV):', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.edge, *self.parent.xr_edges, command=self.fun_edge)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.energy_kev, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_energy)
        var.bind('<KP_Enter>', self.fun_energy)
        # Wavelength
        var = tk.Label(line, text=u'Wavelength (\u212B):', font=SF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.wavelenth_a, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_wavelength)
        var.bind('<KP_Enter>', self.fun_wavelength)

        # --- Resonant HKL ---
        box = tk.Frame(frame)
        box.pack(side=tk.TOP, fill=tk.BOTH, padx=3)

        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        # Resonant HKL, azimuthal reference
        vframe = tk.Frame(line)
        vframe.pack(side=tk.LEFT, fill=tk.Y, padx=3)

        hframe = tk.Frame(vframe)
        hframe.pack()
        var = tk.Label(hframe, text='       HKL:', font=SF, width=11)
        var.pack(side=tk.LEFT)
        var = tk.Entry(hframe, textvariable=self.hkl_magnetic, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_hklmag)
        var.bind('<KP_Enter>', self.fun_hklmag)

        hframe = tk.Frame(vframe)
        hframe.pack()
        var = tk.Label(vframe, text='Azim. Ref.:', font=SF, width=11)
        var.pack(side=tk.LEFT)
        var = tk.Entry(vframe, textvariable=self.azim_zero, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Resonant value
        vframe = tk.Frame(line)
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

        vframe = tk.Frame(line)
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
        var.bind('<Return>', self.fun_hklmag)
        var.bind('<KP_Enter>', self.fun_hklmag)

        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        vframe = tk.Frame(line)
        vframe.pack(side=tk.LEFT, fill=tk.Y, padx=3)

        # Mag. Inten List button
        var = tk.Button(vframe, text='Calc. List', font=BF, command=self.fun_hklmag_list, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5)
        # Mag. Inten button
        var = tk.Button(vframe, text='Calc. Mag. Inten.', font=BF, command=self.fun_hklmag, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5)
        # Magnetic Result
        var = tk.Label(vframe, textvariable=self.magresult, font=SF, width=12)
        var.pack(side=tk.LEFT, fill=tk.Y)

        # Azimuth Button
        var = tk.Button(line, text='Simulate\n Azimuth', font=BF, command=self.fun_azimuth, width=7, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        # Energy Button
        var = tk.Button(line, text='Simulate\n Resonance', font=BF, command=self.fun_dispersion, width=10, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        # FDMNES Button
        if fdmnes_checker():
            var = tk.Button(line, text='FDMNES', font=BF, command=self.fun_fdmnes_ref, height=2, width=10, bg=btn,
                            activebackground=btn_active)
            var.pack(side=tk.RIGHT)

    def fun_energy(self, event=None):
        """Set wavelength"""
        energy_kev = self.energy_kev.get()
        self.wavelenth_a.set(round(fc.energy2wave(energy_kev), 4))

    def fun_wavelength(self, event=None):
        """Set energy"""
        wavelength_a = self.wavelenth_a.get()
        self.energy_kev.set(round(fc.wave2energy(wavelength_a), 4))

    def fun_edge(self, event=None):
        """X-ray edge option menu"""
        edge = self.edge.get()
        if self.edge.get() in self.parent.xr_edges:
            idx = self.parent.xr_edges.index(edge)
            self.energy_kev.set(self.parent.xr_energies[idx])
            self.fun_energy()

    def fun_hklmag(self, event=None):
        """"Magnetic scattering"""

        energy_kev = self.energy_kev.get()
        hkl = fg.str2array(self.hkl_magnetic.get())
        azi = fg.str2array(self.azim_zero.get())
        psi = self.psival.get()
        pol = self.polval.get().replace('\u03c3', 's').replace('\u03c0', 'p')

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

        self.magresult.set('I = %9.4g' % maginten)

    def fun_fdmnes_ref(self):
        """Run FDMNES calculation"""
        from ..classes_fdmnes import Fdmnes, sim_folder
        edge = self.edge.get()
        if edge == 'Edge':
            messagebox.showinfo(
                parent=self.root,
                title='FDMNES Calculation',
                message='Please choose an absorption edge'
            )
            return
        absorber, edge = edge.split()
        hkl = fg.str2array(self.hkl_magnetic.get())
        azi = fg.str2array(self.azim_zero.get())
        pol = self.polval.get().replace('\u03c3', 's').replace('\u03c0', 'p')

        # Create FDMNES calculation
        fdm = Fdmnes(self.xtl)
        fdm.setup(
            output_path=sim_folder('.Dans_Diffraction'),
            comment='Scattering GUI calculation',
            energy_range='-10. 1. -5 0.1 -2. 0.05 5 0.1 10. 0.5 25 1 31.',
            radius=4.0,
            edge=edge,
            absorber=absorber,
            scf=False,
            quadrupole=True,
            azi_ref=azi,
            hkl_reflections=[hkl]
        )
        answer = messagebox.askyesnocancel(
            parent=self.root,
            title='FDMNES Calculation',
            message='Run FDMNES calculation?\nThis will take a few minutes.\nClick No to reload a previous calcualtion'
        )
        if answer is None:
            return
        elif answer:
            # Create files and run FDMNES
            fdm.create_files()
            fdm.write_fdmfile()
            fdm.run_fdmnes()
        # Analyse data
        ana = fdm.analyse()
        for ref in ana:
            ref.plot3D()
        plt.show()

    def fun_hklmag_list(self, event=None):
        """"Magnetic scattering"""

        energy_kev = self.energy_kev.get()
        azi = fg.str2array(self.azim_zero.get())
        psi = self.psival.get()
        F0 = self.resF0.get()
        F1 = self.resF1.get()
        F2 = self.resF2.get()

        if self.parent.orientation.get() == 'Reflection':
            hkl = self.xtl.Scatter.get_hkl(energy_kev=energy_kev, reflection=True)
        elif self.parent.orientation.get() == 'Transmission':
            hkl = self.xtl.Scatter.get_hkl(energy_kev=energy_kev, transmission=True)
        else:
            hkl = self.xtl.Scatter.get_hkl(energy_kev=energy_kev, remove_symmetric=True)

        isres = self.isres.get()
        if isres:
            # Resonant scattering
            magstr = self.xtl.Scatter.print_xray_resonant(
                hkl,
                energy_kev=energy_kev,
                azim_zero=azi, psi=psi,
                F0=F0, F1=F1, F2=F2
            )
            ttl = 'X-Ray Resonant Magnetic Scattering'
        else:
            # Non-Resonant scattering
            magstr = self.xtl.Scatter.print_xray_nonresonant(
                hkl,
                energy_kev=energy_kev,
                azim_zero=azi, psi=psi
            )
            ttl = 'X-Ray Non-Resonant Magnetic Scattering'
        StringViewer(magstr, title=ttl, width=80)

    def fun_azimuth(self):
        """Simulate azimuthal magnetic scattering"""

        energy_kev = self.energy_kev.get()
        hkl = fg.str2array(self.hkl_magnetic.get())
        azi = fg.str2array(self.azim_zero.get())
        pol = self.polval.get().replace('\u03c3', 's').replace('\u03c0', 'p')

        F0 = self.resF0.get()
        F1 = self.resF1.get()
        F2 = self.resF2.get()

        if self.isres.get():
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

    def fun_dispersion(self):
        """Simulate energy resonance"""

        energy_kev = self.energy_kev.get()
        hkl = fg.str2array(self.hkl_magnetic.get())
        hkl = np.array([hkl, -hkl])
        self.xtl.Plot.plot_xray_resonance(hkl, energy_kev=energy_kev, width=1.0, npoints=200)
        plt.show()


class ReflectionSelectionBox:
    """
    Displays all data fields and returns a selection
    Making a selection returns a list of field strings

    out = ReflectionSelectionBox(['field1','field2','field3'], current_selection=['field2'], title='', multiselect=False).show()
    # Make selection and press "Select" > box disappears
    out = ['list','of','strings']
    """

    REF_FMT = '%14s %8.2f  %12.4g'

    def __init__(self, xtl, parent, title='Reflections', multiselect=True,
                 radiation=None, wavelength_a=None):
        self.xtl = xtl
        self.hkl_list = []
        self.tth_list = []
        self.sf_list = []
        self.str_list = []
        self.current_selection = []
        self.output = []

        # Create Tk inter instance
        self.root = tk.Toplevel(parent)
        self.root.wm_title(title)
        self.root.minsize(width=100, height=300)
        self.root.maxsize(width=1200, height=1200)
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol
        )

        # tk variables
        radiations = ['X-Ray', 'Neutron', 'Electron']
        wavelength_types = ['Energy [keV]', 'Energy [eV]', 'Energy [meV]', u'Wavelength [\u212B]', 'Wavelength [nm]']
        max_types = [u'Max Q [\u212B\u207B\u00B9]', u'Max 2\u03B8 [Deg]', u'min d [\u212B]']
        self.radiation_type = tk.StringVar(self.root, radiations[0] if radiation is None else radiation)
        self.check_magnetic = tk.BooleanVar(self.root, False)
        self.wavelength_type = tk.StringVar(self.root, 'Energy [keV]')
        self._prev_wavelength_type = tk.StringVar(self.root, 'Energy [keV]')
        self.wavelength_val = tk.DoubleVar(self.root, 8)
        self.edge = tk.StringVar(self.root, 'Edge')
        self.max_gen_type = tk.StringVar(self.root, u'Max Q [\u212B\u207B\u00B9]')
        self._prev_max_gen_type = tk.StringVar(self.root, u'Max Q [\u212B\u207B\u00B9]')
        self.max_val = tk.DoubleVar(self.root, 4)
        self.min_sf = tk.DoubleVar(self.root, 0)
        self.max_sf = tk.DoubleVar(self.root, np.inf)
        self.add_hkl = tk.StringVar(self.root, '')

        self.fun_radiation()
        self.set_wavelength(1.5 if wavelength_a is None else wavelength_a)

        # X-ray edges:
        self.xr_edges, self.xr_energies = self.xtl.Properties.xray_edges()
        self.xr_edges.insert(0, 'Cu Ka')
        self.xr_edges.insert(1, 'Mo Ka')
        self.xr_energies.insert(0, fg.Cu)
        self.xr_energies.insert(1, fg.Mo)

        # tk Frames
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, anchor=tk.N)

        "---------------------------Scattering------------------------"
        OPT_WIDTH = 14
        frm = tk.LabelFrame(frame, text='Radiation', relief=tk.RIDGE)
        frm.pack(fill=tk.X, padx=4, pady=4)

        # Radiation
        if radiation is None:
            ln = tk.Frame(frm)
            ln.pack(side=tk.TOP, fill=tk.X)
            var = tk.Label(ln, text='Radiation:', font=SF)
            var.pack(side=tk.LEFT)
            var = tk.OptionMenu(ln, self.radiation_type, *radiations, command=self.fun_radiation)
            var.config(font=SF, width=OPT_WIDTH, bg=opt, activebackground=opt_active)
            var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
            var.pack(side=tk.LEFT)
            var = tk.Checkbutton(ln, text='Magnetic', variable=self.check_magnetic, font=SF)
            var.pack(side=tk.LEFT, padx=6)

        # Wavelength / Energy
        ln = tk.Frame(frm)
        ln.pack(side=tk.TOP, fill=tk.X)
        if wavelength_a is None:
            var = tk.OptionMenu(ln, self.wavelength_type, *wavelength_types, command=self.fun_wavelength)
            var.config(font=SF, width=OPT_WIDTH, bg=opt, activebackground=opt_active)
            var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
            var.pack(side=tk.LEFT)
            var = tk.OptionMenu(ln, self.edge, *self.xr_edges, command=self.fun_edge)
            var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
            var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
            var.pack(side=tk.LEFT)
            var = tk.Entry(ln, textvariable=self.wavelength_val, font=TF, width=8, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)
            var.bind('<Return>', self.fun_wavelength)
            var.bind('<KP_Enter>', self.fun_wavelength)
        else:
            var = tk.Label(ln, textvariable=self.wavelength_type, font=SF)
            var.pack(side=tk.LEFT)
            var = tk.Label(ln, textvariable=self.wavelength_val, font=SF)
            var.pack(side=tk.LEFT)

        # Max Q
        ln = tk.Frame(frm)
        ln.pack(side=tk.TOP, fill=tk.X)
        var = tk.OptionMenu(ln, self.max_gen_type, *max_types, command=self.get_max_q)
        var.config(font=SF, width=OPT_WIDTH, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(ln, textvariable=self.max_val, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_gen_refs)
        var.bind('<KP_Enter>', self.fun_gen_refs)
        var = tk.Button(ln, text='?', font=TF, command=self.help_max_q, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        # Intensities
        ln = tk.Frame(frm)
        ln.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(ln, text=u'Min |SF|\u00B2:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(ln, textvariable=self.min_sf, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(ln, text=u'    Max |SF|\u00B2:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(ln, textvariable=self.max_sf, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Add hkl
        ln = tk.Frame(frm)
        ln.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(ln, text=u'hkl', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(ln, textvariable=self.add_hkl, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_add_hkl)
        var.bind('<KP_Enter>', self.fun_add_hkl)
        var = tk.Button(ln, text='Add ref', font=TF, command=self.fun_add_hkl, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        # Generate Buttons
        ln = tk.Frame(frm)
        ln.pack(side=tk.TOP, fill=tk.X)
        var = tk.Button(ln, text='Gen Sym', font=TF, command=self.fun_gen_sym_refs, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)
        var = tk.Button(ln, text='Rem Sym', font=TF, command=self.fun_rem_sym_refs, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)
        var = tk.Button(ln, text='Gen Refs', font=TF, command=self.fun_gen_refs, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT, pady=2)
        var = tk.Button(ln, text='Clear', font=TF, command=self.clear_reflections, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT, pady=2)

        "---------------------------ListBox---------------------------"
        ln = tk.Frame(frame)
        ln.pack(side=tk.TOP)
        var = tk.Label(ln, text='hkl    Two-Theta    Intensity', font=SF)
        var.pack(side=tk.LEFT)

        # Eval box with scroll bar
        frm = tk.Frame(frame)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        sclx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        sclx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scly = tk.Scrollbar(frm)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.lst_data = tk.Listbox(frm, font=MF, selectmode=tk.SINGLE, width=40, height=20, bg=ety,
                                   xscrollcommand=sclx.set, yscrollcommand=scly.set)
        self.lst_data.configure(exportselection=True)
        if multiselect:
            self.lst_data.configure(selectmode=tk.EXTENDED)
        self.lst_data.bind('<<ListboxSelect>>', self.fun_listboxselect)
        self.lst_data.bind('<Double-Button-1>', self.fun_exitbutton)
        self.lst_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        sclx.config(command=self.lst_data.xview)
        scly.config(command=self.lst_data.yview)

        # self.txt_data.config(xscrollcommand=scl_datax.set,yscrollcommand=scl_datay.set)

        "----------------------------Search Field-----------------------------"
        frm = tk.LabelFrame(frame, text='Search', relief=tk.RIDGE)
        frm.pack(fill=tk.X, expand=tk.YES, padx=2, pady=2)

        self.searchbox = tk.StringVar(self.root, '')
        var = tk.Entry(frm, textvariable=self.searchbox, font=TF, bg=ety, fg=ety_txt)
        var.bind('<Key>', self.fun_search)
        var.pack(fill=tk.X, expand=tk.YES, padx=2, pady=2)

        "----------------------------Exit Button------------------------------"
        frm_btn = tk.Frame(frame)
        frm_btn.pack(fill=tk.X, expand=tk.YES)

        self.numberoffields = tk.StringVar(self.root, '%3d Selected Fields' % 0)
        var = tk.Label(frm_btn, textvariable=self.numberoffields, width=20)
        var.pack(side=tk.LEFT)
        btn_exit = tk.Button(frm_btn, text='Select', font=BF, command=self.fun_exitbutton, bg=btn,
                             activebackground=btn_active)
        btn_exit.pack(side=tk.RIGHT)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        # self.root.mainloop()

    "------------------------------------------------------------------------"
    "------------------------Scattering Functions----------------------------"
    "------------------------------------------------------------------------"

    def get_scattering_type(self):
        """Get scattering type"""
        radiation = self.radiation_type.get()
        magnetic = self.check_magnetic.get()
        if radiation == 'X-Ray':
            if magnetic:
                scattering_type = 'xray magnetic'
            else:
                scattering_type = 'xray'
        elif radiation == 'Neutron':
            if magnetic:
                scattering_type = 'neutron magnetic'
            else:
                scattering_type = 'neutron'
        elif radiation == 'Electron':
            scattering_type = 'electron'
        else:
            scattering_type = radiation
        return scattering_type

    def get_wavelength(self):
        """Return wavelength in A according to unit"""
        val = self.wavelength_val.get()
        rad = self.radiation_type.get()
        unit = self._prev_wavelength_type.get()

        if unit == 'Energy [keV]':
            if 'Electron' in rad:
                wavelength_a = fc.electron_wavelength(val * 1000)
            elif 'Neutron' in rad:
                wavelength_a = fc.neutron_wavelength(val * 1e6)
            else:
                wavelength_a = fc.energy2wave(val)
        elif unit == 'Energy [meV]':
            if 'Electron' in rad:
                wavelength_a = fc.electron_wavelength(val / 1000.)
            elif 'Neutron' in rad:
                wavelength_a = fc.neutron_wavelength(val)
            else:
                wavelength_a = fc.energy2wave(val / 1.0e6)
        elif unit == 'Energy [eV]':
            if 'Electron' in rad:
                wavelength_a = fc.electron_wavelength(val)
            elif 'Neutron' in rad:
                wavelength_a = fc.neutron_wavelength(val * 1000)
            else:
                wavelength_a = fc.energy2wave(val / 1000.)
        elif unit == 'Wavelength [nm]':
            wavelength_a = val / 10.
        else:
            wavelength_a = val
        return wavelength_a

    def set_wavelength(self, wavelength_a):
        """set wavelength according to unit"""
        rad = self.radiation_type.get()
        unit = self.wavelength_type.get()

        if unit == 'Energy [keV]':
            if 'Electron' in rad:
                val = fc.electron_energy(wavelength_a) / 1000.
            elif 'Neutron' in rad:
                val = fc.neutron_energy(wavelength_a) / 1.0e6
            else:
                val = fc.wave2energy(wavelength_a)
        elif unit == 'Energy [meV]':
            if 'Electron' in rad:
                val = fc.electron_energy(wavelength_a) * 1000
            elif 'Neutron' in rad:
                val = fc.neutron_energy(wavelength_a)
            else:
                val = fc.wave2energy(wavelength_a) / 1.0e6
        elif unit == 'Energy [eV]':
            if 'Electron' in rad:
                val = fc.electron_energy(wavelength_a)
            elif 'Neutron' in rad:
                val = fc.neutron_energy(wavelength_a) * 1000
            else:
                val = fc.wave2energy(wavelength_a) / 1000.
        elif unit == 'Wavelength [nm]':
            val = wavelength_a * 10.,
        else:
            val = wavelength_a
        self.wavelength_val.set(round(val, 4))
        self._prev_wavelength_type.set(unit)
        # Set max Q
        max_q = fc.calqmag(180, wavelength_a=wavelength_a)
        max_gen = self.max_gen_type.get()
        if max_gen == u'Max Q [\u212B\u207B\u00B9]':
            self.max_val.set(round(max_q, 4))
        # elif max_gen == u'Max 2\u03B8 [Deg]':
        #     energy_kev = fc.wave2energy(wavelength_a)
        #     max_val.set(round(fc.cal2theta(max_q, energy_kev), 4))
        elif max_gen == u'min d [\u212B]':
            self.max_val.set(round(fc.q2dspace(max_q), 4))

    def fun_radiation(self, event=None):
        """Set radiation"""
        rad = self.radiation_type.get()
        wavelength_a = self.get_wavelength()
        if 'Neutron' in rad:
            self.wavelength_type.set('Energy [meV]')
        elif 'Electron' in rad:
            self.wavelength_type.set('Energy [eV]')
            self.check_magnetic.set(False)
        else:
            self.wavelength_type.set('Energy [keV]')
        self.set_wavelength(wavelength_a)

    def fun_wavelength(self, event=None):
        """Convert previous unit"""
        wavelength_a = self.get_wavelength()
        self.set_wavelength(wavelength_a)

    def fun_edge(self, event=None):
        """X-ray edge option menu"""
        edge_name = self.edge.get()
        if edge_name in self.xr_edges:
            idx = self.xr_edges.index(edge_name)
            self.set_wavelength(fc.energy2wave(self.xr_energies[idx]))

    def get_max_q(self, event=None):
        """Return max val in inverse angstroms, convert if changed"""
        val = self.max_val.get()
        old_max_gen = self._prev_max_gen_type.get()
        max_gen = self.max_gen_type.get()
        if old_max_gen == u'Max Q [\u212B\u207B\u00B9]':
            max_q = val
        elif old_max_gen == u'Max 2\u03B8 [Deg]':
            wavelength_a = self.get_wavelength()
            max_q = fc.calqmag(twotheta=val, wavelength_a=wavelength_a)
        else:  # max_gen == u'min d [\u212B]'
            max_q = fc.dspace2q(val)
        # Convert if changed
        if max_gen != old_max_gen:
            if max_gen == u'Max Q [\u212B\u207B\u00B9]':
                self.max_val.set(round(max_q, 4))
            elif max_gen == u'Max 2\u03B8 [Deg]':
                wavelength_a = self.get_wavelength()
                tth = fc.cal2theta(max_q, wavelength_a=wavelength_a)
                tth = 180. if np.isnan(tth) else tth
                self.max_val.set(round(tth, 4))
            else:  # max_gen == u'min d [\u212B]'
                self.max_val.set(round(fc.q2dspace(max_q), 4))
            self._prev_max_gen_type.set(max_gen)
        return max_q

    def help_max_q(self):
        msg = "Calculate reflection list upto this value\n  (lower angle is less reflections)."
        messagebox.showinfo(
            parent=self.root,
            title='max-Q',
            message=msg
        )

    def fun_add_hkl(self, event=None):
        """Add additional hkl"""
        hkl = fg.str2array(self.add_hkl.get())
        if len(hkl) > 0:
            self.add_reflection(hkl)

    def fun_gen_sym_refs(self):
        new_list = []
        for hkl in self.hkl_list:
            if not fg.vectorinvector(hkl, new_list):
                new_list += [hkl]
            sym_list = self.xtl.Symmetry.symmetric_reflections_unique(hkl)
            for sym_hkl in sym_list:
                if not fg.vectorinvector(sym_hkl, new_list):
                    new_list += [sym_hkl]
        self.add_reflection_list(new_list)

    def fun_rem_sym_refs(self):
        new_list = self.xtl.Symmetry.remove_symmetric_reflections(self.hkl_list)
        self.add_reflection_list(new_list)

    def fun_gen_refs(self):
        """Generate reflections"""
        maxq = self.get_max_q()
        hkl = self.xtl.Cell.all_hkl(maxq=maxq)
        hkl = self.xtl.Cell.sort_hkl(hkl)[1:]  # remove [0,0,0]
        self.add_reflection_list(hkl)

    def clear_reflections(self):
        """Clear reflection list"""
        self.hkl_list = []
        self.tth_list = []
        self.sf_list = []
        self.str_list = []
        self.lst_data.delete(0, tk.END)

    def add_reflection(self, hkl):
        """Add reflection to list"""
        if not fg.vectorinvector(hkl, self.hkl_list):
            wavelength_a = self.get_wavelength()
            scattering_type = self.get_scattering_type()
            tth = self.xtl.Cell.tth(hkl, wavelength_a=wavelength_a)[0]
            intensity = self.xtl.Scatter.intensity(
                hkl=hkl,
                scattering_type=scattering_type,
                wavelength_a=wavelength_a
            )[0]
            intensity = round(intensity, 2)
            hkl_str = fc.hkl2str(hkl)
            ref_str = self.REF_FMT % (hkl_str, tth, intensity)
            self.hkl_list += [hkl]
            self.tth_list += [tth]
            self.sf_list += [intensity]
            self.str_list += [ref_str]
            self.lst_data.insert(tk.END, ref_str)

    def add_reflection_list(self, hkl_list):
        """Replace reflection list"""
        min_sf = self.min_sf.get()
        max_sf = self.max_sf.get()
        self.clear_reflections()
        wavelength_a = self.get_wavelength()
        scattering_type = self.get_scattering_type()
        tth = self.xtl.Cell.tth(hkl_list, wavelength_a=wavelength_a)
        intensity = self.xtl.Scatter.intensity(
            hkl=hkl_list,
            scattering_type=scattering_type,
            wavelength_a=wavelength_a
        )
        for n in range(len(hkl_list)):
            if min_sf < intensity[n] < max_sf:
                ii = round(intensity[n], 2)
                hkl_str = fc.hkl2str(hkl_list[n])
                ref_str = self.REF_FMT % (hkl_str, tth[n], ii)
                self.hkl_list += [hkl_list[n]]
                self.tth_list += [tth[n]]
                self.sf_list += [ii]
                self.str_list += [ref_str]
                self.lst_data.insert(tk.END, ref_str)

    "------------------------------------------------------------------------"
    "--------------------------ListBox Functions-----------------------------"
    "------------------------------------------------------------------------"

    def show(self):
        """Run the selection box, wait for response"""

        # self.root.deiconify()  # show window
        self.root.wait_window()  # wait for window
        return self.output

    def fun_search(self, event=None):
        """Search the selection for string"""
        search_str = self.searchbox.get()
        search_str = search_str + event.char
        search_str = search_str.strip().lower()
        if not search_str: return

        # Clear current selection
        self.lst_data.select_clear(0, tk.END)
        view_idx = None
        # Search for whole words first
        for n, item in enumerate(self.str_list):
            if re.search(r'\b%s\b' % search_str, item.lower()):  # whole word search
                self.lst_data.select_set(n)
                view_idx = n
        # if nothing found, search anywhere
        if view_idx is None:
            for n, item in enumerate(self.str_list):
                if search_str in item.lower():
                    self.lst_data.select_set(n)
                    view_idx = n
        if view_idx is not None:
            self.lst_data.see(view_idx)
        self.fun_listboxselect()

    def fun_listboxselect(self, event=None):
        """Update label on listbox selection"""
        self.numberoffields.set('%3d Selected Fields' % len(self.lst_data.curselection()))

    def fun_exitbutton(self, event=None):
        """Closes the current data window and generates output"""
        selection = self.lst_data.curselection()
        self.output = {
            'hkl': np.array([self.hkl_list[n] for n in selection], dtype=int),
            'tth': np.array([self.tth_list[n] for n in selection]),
            'sf2': np.array([self.sf_list[n] for n in selection]),
        }
        self.root.destroy()

    def f_exit(self, event=None):
        """Closes the current data window"""
        self.output = {
            'hkl': np.array([]),
            'tth': np.array([]),
            'sf2': np.array([])
        }
        self.root.destroy()

