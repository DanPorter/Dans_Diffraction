"""
Scattering GUI
"""

import sys, os

import matplotlib.pyplot as plt # Plotting
import numpy as np
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .basic_widgets import StringViewer, topmenu
from .basic_widgets import (TF, BF, SF, LF, HF,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)

# Scattering functions
"""




plot_xray_resonance(self, hkl, energy_kev=None, width=1.0, npoints=200)
plot_3Dpolarisation(self, hkl, energy_kev=None, polarisation='sp', azim_zero=[1,0,0], psi=0)



ms_azimuth(self, hkl, energy_kev, azir=[0, 0, 1], pv=[1, 0], numsteps=3, peak_width=0.1,
                   full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False)
diff6circle_intensity(self, phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0,
                              energy_kev=None, wavelength=1.0, fwhm=0.5)
multiple_scattering(self, hkl, azir=[0, 0, 1], pv=[1, 0], energy_range=[7.8, 8.2], numsteps=60,
                            full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False)
"""


class ScatteringGuiOLD:
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

        # Variatbles
        self.energy_kev = tk.DoubleVar(frame, 8.0)
        self.wavelenth_a = tk.DoubleVar(frame, 1.5498)
        self.neutron_wl = tk.DoubleVar(frame, False)
        self.electron_wl = tk.DoubleVar(frame, False)
        self.edge = tk.StringVar(frame, 'Edge')
        self.type = tk.StringVar(frame, 'X-Ray')
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

        # Energy
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text='Energy (keV):', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.edge, *self.xr_edges, command=self.fun_edge)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.energy_kev, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_energy)
        var.bind('<KP_Enter>', self.fun_energy)

        # Wavelength
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        var = tk.Label(line, text=u'Wavelength (\u212B):', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.wavelenth_a, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_wavelength)
        var.bind('<KP_Enter>', self.fun_wavelength)
        var = tk.Checkbutton(line, text='Neutron', variable=self.neutron_wl, font=SF, command=self.fun_wavelength)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Checkbutton(line, text='Electron', variable=self.electron_wl, font=SF, command=self.fun_wavelength)
        var.pack(side=tk.LEFT, padx=5)

        # Scattering Type
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        types = ['X-Ray', 'Neutron', 'Electron', 'XRay Magnetic', 'Neutron Magnetic', 'XRay Resonant', 'XRay Dispersion']
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
        var = tk.Button(side, text='Arbitary Cut', font=BF, command=self.fun_cut, bg=btn,
                        activebackground=btn_active)
        var.pack()

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

        self.type.set(self.xtl._scattering_type)
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
        scat._scattering_type = self.type.get()
        scat._energy_kev = self.energy_kev.get()
        scat._scattering_theta_offset = self.theta_offset.get()
        scat._scattering_min_theta = self.theta_min.get()
        scat._scattering_max_theta = self.theta_max.get()
        scat._scattering_min_twotheta = self.twotheta_min.get()
        scat._scattering_max_twotheta = self.twotheta_max.get()
        scat._powder_units = self.powder_units.get()

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

    def fun_i16(self):
        """"Add I16 parameters"""

        self.type.set('X-Ray')
        self.energy_kev.set(8.0)
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

        self.type.set('Neutron')
        self.energy_kev.set(17.7)
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

        self.type.set('X-Ray')
        idx = self.xr_edges.index('Mo Ka')
        self.edge.set('Mo Ka')
        self.energy_kev.set(self.xr_energies[idx])
        self.powder_units.set('Two-Theta')
        self.orientation.set('None')
        self.theta_offset.set(0.0)
        self.theta_min.set(-180.0)
        self.theta_max.set(180.0)
        self.twotheta_min.set(-170.0)
        self.twotheta_max.set(170.0)

    def fun_energy(self, event=None):
        """Set wavelength"""
        energy_kev = self.energy_kev.get()
        if self.neutron_wl.get():
            self.wavelenth_a.set(round(fc.neutron_wavelength(1e6 * energy_kev), 4))
        elif self.electron_wl.get():
            self.wavelenth_a.set(round(fc.electron_wavelength(1000 * energy_kev), 4))
        else:
            self.wavelenth_a.set(round(fc.energy2wave(energy_kev), 4))

    def fun_wavelength(self, event=None):
        """Set energy"""
        wavelength_a = self.wavelenth_a.get()
        if self.neutron_wl.get():
            self.electron_wl.set(False)
            self.energy_kev.set(round(fc.neutron_energy(wavelength_a) / 1e6, 4))
        elif self.electron_wl.get():
            self.neutron_wl.set(False)
            self.energy_kev.set(round(fc.electron_energy(wavelength_a) / 1000, 4))
        else:
            self.energy_kev.set(round(fc.wave2energy(wavelength_a), 4))

    def fun_edge(self, event=None):
        """X-ray edge option menu"""
        edge = self.edge.get()
        if self.edge.get() in self.xr_edges:
            idx = self.xr_edges.index(edge)
            self.energy_kev.set(self.xr_energies[idx])
            self.fun_energy()

    def fun_hklcheck(self, event=None):
        """"Show single hkl intensity"""

        hkl = self.fun_get_hkl()
        I = self.xtl.Scatter.intensity(hkl)

        unit = self.powder_units.get()
        energy = self.energy_kev.get()
        tth = self.xtl.Cell.tth(hkl, energy)

        if unit.lower() in ['tth', 'angle', 'twotheta', 'theta', 'two-theta']:
            self.hkl_result.set('I:%10.0f TTH:%8.2f' % (I, tth))
        elif unit.lower() in ['d', 'dspace', 'd-spacing', 'dspacing']:
            q = fc.calqmag(tth, energy)
            d = fc.q2dspace(q)
            self.hkl_result.set(u'I:%10.0f   d:%8.2f \u00c5' % (I, d))
        else:
            q = fc.calqmag(tth, energy)
            self.hkl_result.set(u'I:%8.0f   Q:%8.2f \u00c5\u207B\u00B9' % (I, q))

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
        energy = self.energy_kev.get()
        min_q = fc.calqmag(self.twotheta_min.get(), energy)
        max_q = fc.calqmag(self.twotheta_max.get(), energy)
        pow_avg = self.powderaverage.get()
        pow_wid = self.powder_width.get()
        #if min_q < 0: min_q = 0.0

        self.xtl.Plot.simulate_powder(energy, peak_width=pow_wid, powder_average=pow_avg)
        plt.show()

    def fun_hki(self):
        """Plot hki plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hk0(i, peak_width=0.2)
        plt.show()

    def fun_hil(self):
        """Plot hil plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_h0l(i, peak_width=0.2)
        plt.show()

    def fun_ikl(self):
        """Plot ikl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_0kl(i, peak_width=0.2)
        plt.show()

    def fun_hhi(self):
        """Plot hhl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hhl(i, peak_width=0.2)
        plt.show()

    def fun_cut(self):
        self.fun_get()
        ArbitaryCutGui(self.xtl)

    def fun_rxs(self):
        ResonantXrayGui(self)


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

        # Variatbles
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
        var = tk.Button(side, text='Arbitary Cut', font=BF, command=self.fun_cut, bg=btn,
                        activebackground=btn_active)
        var.pack()

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
        if radiation == 'X-Ray':
            if magnetic:
                scat._scattering_type = 'xray magnetic'
            else:
                scat._scattering_type = 'xray'
        elif radiation == 'Neutron':
            if magnetic:
                scat._scattering_type = 'neutron magnetic'
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
        energy = self.get_energy()
        tth = self.xtl.Cell.tth(hkl, energy)

        if unit.lower() in ['tth', 'angle', 'twotheta', 'theta', 'two-theta']:
            self.hkl_result.set('I:%10.0f TTH:%8.2f' % (I, tth))
        elif unit.lower() in ['d', 'dspace', 'd-spacing', 'dspacing']:
            q = fc.calqmag(tth, energy)
            d = fc.q2dspace(q)
            self.hkl_result.set(u'I:%10.0f   d:%8.2f \u00c5' % (I, d))
        else:
            q = fc.calqmag(tth, energy)
            self.hkl_result.set(u'I:%8.0f   Q:%8.2f \u00c5\u207B\u00B9' % (I, q))

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
        energy = self.get_energy()
        min_q = fc.calqmag(self.twotheta_min.get(), energy)
        max_q = fc.calqmag(self.twotheta_max.get(), energy)
        pow_avg = self.powderaverage.get()
        pow_wid = self.powder_width.get()
        #if min_q < 0: min_q = 0.0

        self.xtl.Plot.simulate_powder(energy, peak_width=pow_wid, powder_average=pow_avg)
        plt.show()

    def fun_hki(self):
        """Plot hki plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hk0(i, peak_width=0.2)
        plt.show()

    def fun_hil(self):
        """Plot hil plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_h0l(i, peak_width=0.2)
        plt.show()

    def fun_ikl(self):
        """Plot ikl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_0kl(i, peak_width=0.2)
        plt.show()

    def fun_hhi(self):
        """Plot hhl plane"""
        self.fun_get()
        i = self.val_i.get()
        self.xtl.Plot.simulate_hhl(i, peak_width=0.2)
        plt.show()

    def fun_cut(self):
        self.fun_get()
        ArbitaryCutGui(self.xtl)

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
        var = tk.Label(line, text='Arbitary HKl Cut', font=LF)
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
