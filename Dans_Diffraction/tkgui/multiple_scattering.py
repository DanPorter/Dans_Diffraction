"""
GUI for MultipleScattering code
"""

import sys, os

import matplotlib.pyplot as plt
import numpy as np

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .basic_widgets import tk
from .basic_widgets import (TF, BF, SF, LF, HF, TTF, TTFG, TTBG,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)


class MultipleScatteringGui:
    """
    Simulate multiple scattering
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Multiple Scattering %s' % xtl.name)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # Variatbles
        # hkl, azir=[0, 0, 1], pv=[1, 0], energy_range=[7.8, 8.2], numsteps=60,
        # full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False
        self.reflection_h = tk.IntVar(frame, 1)
        self.reflection_k = tk.IntVar(frame, 0)
        self.reflection_l = tk.IntVar(frame, 0)
        self.azir_h = tk.IntVar(frame, 0)
        self.azir_k = tk.IntVar(frame, 0)
        self.azir_l = tk.IntVar(frame, 1)
        self.pv_x = tk.IntVar(frame, 1)
        self.pv_y = tk.IntVar(frame, 0)
        self.energy_value = tk.DoubleVar(frame, 8)
        self.energy_range_width = tk.DoubleVar(frame, 0.1)
        self.numsteps = tk.IntVar(frame, 101)
        self.run_modes = ["full", "pv1", "pv2", "sfonly", "pv1xsf1"]
        self.run_mode = tk.StringVar(frame, self.run_modes[3])

        # ---Line 1---
        line = tk.Frame(frame, bg=TTBG)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        var = tk.Label(line, text='Multiple Scattering by Gareth Nisbet', font=TTF, fg=TTFG, bg=TTBG)
        var.pack()

        # ---Line 2---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        # Reflection
        var = tk.Label(line, text='hkl:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.reflection_h, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.focus() # set initial selection
        var.selection_range(0, tk.END)
        var = tk.Entry(line, textvariable=self.reflection_k, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.reflection_l, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Azir
        var = tk.Label(line, text='azir:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.azir_h, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.azir_k, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.azir_l, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # pv
        var = tk.Label(line, text='pv:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.pv_x, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Entry(line, textvariable=self.pv_y, font=TF, width=2, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # ---Line 3---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, pady=5)

        # run options
        for mode in self.run_modes:
            b = tk.Radiobutton(line, text=mode, variable=self.run_mode, value=mode, font=LF)
            b.pack(side=tk.LEFT, padx=3)

        # ---Line 4---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        # energy
        var = tk.Label(line, text='Energy:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.energy_value, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Run button
        var = tk.Button(line, text='Plot Azimuth', font=BF, command=self.fun_azimuth, width=10, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT, padx=5)

        # ---Line 5---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        # energy
        var = tk.Label(line, text='Energy Width:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.energy_range_width, font=TF, width=6, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # numsteps
        var = tk.Label(line, text='Steps:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.numsteps, font=TF, width=4, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=5)

        # Run button
        var = tk.Button(line, text='Plot Energy', font=BF, command=self.fun_energy, width=10, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT, padx=5)

    def fun_energy(self):
        """Run multiple scattering code"""

        refh = self.reflection_h.get()
        refk = self.reflection_k.get()
        refl = self.reflection_l.get()
        hkl = [refh, refk, refl]
        azih = self.azir_h.get()
        azik = self.azir_k.get()
        azil = self.azir_l.get()
        azir = [azih, azik, azil]
        pv = [self.pv_x.get(), self.pv_y.get()]
        en = self.energy_value.get()
        wid = self.energy_range_width.get()
        erange = [en-wid/2, en+wid/2]
        numsteps = self.numsteps.get()
        modes = {mode: False for mode in self.run_modes}
        mode = self.run_mode.get()
        modes[mode] = True

        self.xtl.Plot.plot_multiple_scattering(hkl, azir, pv, erange, numsteps, **modes)
        plt.show()

    def fun_azimuth(self):
        """Run multiple scattering code"""

        refh = self.reflection_h.get()
        refk = self.reflection_k.get()
        refl = self.reflection_l.get()
        hkl = [refh, refk, refl]
        azih = self.azir_h.get()
        azik = self.azir_k.get()
        azil = self.azir_l.get()
        azir = [azih, azik, azil]
        pv = [self.pv_x.get(), self.pv_y.get()]
        energy = self.energy_value.get()
        modes = {mode: False for mode in self.run_modes}
        mode = self.run_mode.get()
        modes[mode] = True

        self.xtl.Plot.plot_ms_azimuth(hkl, energy, azir, pv, **modes)
        plt.show()
