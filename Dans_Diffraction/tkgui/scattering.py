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
            activeForeground=txtcol)

        # ---Menu---
        menu = {
            'Info': {
                'Crystal': self.menu_info_crystal,
                'Scattering Settings': self.menu_info_scattering,
            },
        }
        topmenu(self.root, menu)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Variatbles
        self.energy_kev = tk.DoubleVar(frame, 8.0)
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
        self.val_i = tk.IntVar(frame, 0)
        self.hkl_magnetic = tk.StringVar(frame, '0 0 1')
        self.azim_zero = tk.StringVar(frame, '1 0 0')
        self.isres = tk.BooleanVar(frame, True)
        self.psival = tk.DoubleVar(frame, 0.0)
        self.polval = tk.StringVar(frame, u'\u03c3-\u03c0')
        self.resF0 = tk.DoubleVar(frame, 0.0)
        self.resF1 = tk.DoubleVar(frame, 1.0)
        self.resF2 = tk.DoubleVar(frame, 0.0)
        self.magresult = tk.StringVar(frame, 'I = --')

        # X-ray edges:
        self.xr_edges, self.xr_energies = self.xtl.Properties.xray_edges()
        self.xr_edges.insert(0, 'Cu Ka')
        self.xr_edges.insert(1, 'Mo Ka')
        self.xr_energies.insert(0, fg.Cu)
        self.xr_energies.insert(1, fg.Mo)


        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)

        var = tk.Label(line, text='Scattering', font=LF)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Supernova', font=BF, command=self.fun_supernova, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        var = tk.Button(line, text='Wish', font=BF, command=self.fun_wish, bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT)
        var = tk.Button(line, text='I16', font=BF, command=self.fun_i16, bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT)

        # ---Settings---
        box = tk.LabelFrame(frame, text='Settings')
        box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

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

        # Type
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
        box = tk.LabelFrame(frame, text='Intensities')
        box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)

        var = tk.Button(line, text='Display Intensities', font=BF, command=self.fun_intensities, bg=btn2,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(line, text='Plot Powder', font=BF, command=self.fun_powder, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # hkl check
        line = tk.Frame(box)
        line.pack(side=tk.TOP, fill=tk.X, pady=5)
        hklbox = tk.LabelFrame(line, text='Quick Check')
        hklbox.pack(side=tk.RIGHT)
        var = tk.Entry(hklbox, textvariable=self.hkl_check, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_hklcheck)
        var.bind('<KP_Enter>', self.fun_hklcheck)
        var = tk.Label(hklbox, textvariable=self.hkl_result, font=TF, width=22)
        var.pack(side=tk.LEFT)
        var = tk.Button(hklbox, text='Check HKL', font=BF, command=self.fun_hklcheck, bg=btn,
                       activebackground=btn_active)
        var.pack(side=tk.LEFT, pady=2)

        # ---Planes---
        box = tk.LabelFrame(frame, text='Reciprocal Space Planes')
        box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        line = tk.Frame(box)
        line.pack(side=tk.TOP, pady=5)

        # ---HKL Planes---
        # i value
        var = tk.Label(line, text='i:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.val_i, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # directions
        vframe = tk.Frame(line)
        vframe.pack(side=tk.LEFT, padx=3)
        var = tk.Button(vframe, text='HKi', font=BF, command=self.fun_hki, width=5, bg=btn, activebackground=btn_active)
        var.pack()
        var = tk.Button(vframe, text='HiL', font=BF, command=self.fun_hil, width=5, bg=btn, activebackground=btn_active)
        var.pack()

        vframe = tk.Frame(line)
        vframe.pack(side=tk.LEFT)
        var = tk.Button(vframe, text='iKL', font=BF, command=self.fun_ikl, width=5, bg=btn, activebackground=btn_active)
        var.pack()
        var = tk.Button(vframe, text='HHi', font=BF, command=self.fun_hhi, width=5, bg=btn, activebackground=btn_active)
        var.pack()

        # ---X-ray Magnetic scattering----
        if np.any(self.xtl.Structure.mxmymz()):
            box = tk.LabelFrame(frame, text='X-Ray Magnetic Scattering')
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
        hkl = hkl.replace(',', ' ')  # remove commas
        hkl = hkl.replace('(', '').replace(')', '')  # remove brackets
        hkl = hkl.replace('[', '').replace(']', '')  # remove brackets
        hkl = np.fromstring(hkl, sep=' ')
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

        self.magresult.set('I = %9.4g' % maginten)

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

