"""
GUI for Tensor Scattering code
"""

import sys, os

import matplotlib.pyplot as plt
import numpy as np
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .basic_widgets import StringViewer
from .basic_widgets import (TF, BF, SF, LF, HF, TTF, TTFG, TTBG,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)


class TensorScatteringGui:
    """
    Simulate Tensor scattering
    """

    def __init__(self, xtl):
        """Initialise"""
        self.xtl = xtl
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Tensor Scattering %s' % xtl.name)
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
        sites = self.xtl.Atoms.label
        self.atom_site = tk.StringVar(frame, sites[0])
        processes = ['Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag']
        self.process = tk.StringVar(frame, 'E1E1')
        self.reflection_h = tk.IntVar(frame, 1)
        self.reflection_k = tk.IntVar(frame, 0)
        self.reflection_l = tk.IntVar(frame, 0)
        self.azir_h = tk.IntVar(frame, 0)
        self.azir_k = tk.IntVar(frame, 0)
        self.azir_l = tk.IntVar(frame, 1)
        self.energy_value = tk.DoubleVar(frame, 8)
        # X-ray edges:
        self.xr_edges, self.xr_energies = self.xtl.Properties.xray_edges()
        self.edge = tk.StringVar(frame, 'Edge')
        self.psi_value = tk.DoubleVar(frame, 0)
        self.rank = tk.IntVar(frame, 2)
        options_time = ['+1', '-1', '0']
        self.time = tk.IntVar(frame, 1)
        options_parity = ['+1', '-1', '0']
        self.parity = tk.IntVar(frame, 1)
        self.mk = tk.StringVar(frame, '')
        self.lk = tk.StringVar(frame, '')
        self.sk = tk.StringVar(frame, '')

        # ---Line 1---
        line = tk.Frame(frame, bg=TTBG)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        var = tk.Label(line, text='Tensor Scattering by Steve Collins', font=TTF, fg=TTFG, bg=TTBG)
        var.pack()

        # ---Line 2---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        # Atom label
        var = tk.Label(line, text='Atom Site:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.OptionMenu(line, self.atom_site, *sites)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT, padx=5)

        # Process
        var = tk.Label(line, text='Process:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.OptionMenu(line, self.process, *processes)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT, padx=5)

        # ---Line 3---
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

        # energy
        var = tk.Label(line, text='Energy (keV):', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.energy_value, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(line, self.edge, *self.xr_edges, command=self.fun_edge)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # ---Line 4---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=5)

        # Psi
        var = tk.Label(line, text='psi:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.psi_value, font=TF, width=4, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=5)

        # Rank
        var = tk.Label(line, text='Rank (K):', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.rank, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # Time
        var = tk.Label(line, text='Time:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.OptionMenu(line, self.time, *options_time)
        var.config(font=SF, width=4, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT, padx=5)

        # Parity
        var = tk.Label(line, text='Parity:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.OptionMenu(line, self.parity, *options_parity)
        var.config(font=SF, width=4, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT, padx=5)

        # ---Line 5---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, pady=5)

        # mk
        var = tk.Label(line, text='mk:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.mk, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        # lk
        var = tk.Label(line, text='lk:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.lk, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        # sk
        var = tk.Label(line, text='sk:', font=LF)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Entry(line, textvariable=self.sk, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # ---Line 6---
        line = tk.Frame(frame)
        line.pack(side=tk.TOP, pady=5)

        # Run buttons
        var = tk.Button(line, text='Plot\nAzimuth', font=BF, command=self.fun_azimuth, width=10, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Button(line, text='Plot\nAnalyser', font=BF, command=self.fun_analyser, width=10, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Button(line, text='Display\nTensor', font=BF, command=self.fun_tensor, width=10, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5)
        var = tk.Button(line, text='Display\nReflections', font=BF, command=self.fun_refs, width=10, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5)

    def fun_edge(self, event=None):
        """X-ray edge option menu"""
        edge = self.edge.get()
        if self.edge.get() in self.xr_edges:
            idx = self.xr_edges.index(edge)
            self.energy_value.set(self.xr_energies[idx])

    def fun_azimuth(self):
        """Run tensor scattering code for varying azimuth"""

        atom_label = self.atom_site.get()
        process = self.process.get()
        refh = self.reflection_h.get()
        refk = self.reflection_k.get()
        refl = self.reflection_l.get()
        hkl = [refh, refk, refl]
        azih = self.azir_h.get()
        azik = self.azir_k.get()
        azil = self.azir_l.get()
        azir = [azih, azik, azil]
        energy_kev = self.energy_value.get()
        rank = self.rank.get()
        time = self.time.get()
        parity = self.parity.get()
        try:
            mk = eval(self.mk.get())
        except SyntaxError:
            mk = None
        try:
            lk = eval(self.lk.get())
        except SyntaxError:
            lk = None
        try:
            sk = eval(self.sk.get())
        except SyntaxError:
            sk = None

        self.xtl.Plot.tensor_scattering_azimuth(atom_label, hkl, energy_kev, azir,
                                                process, rank, time, parity, mk, lk, sk)
        plt.show()

    def fun_analyser(self):
        """Run tensor scattering code for varying stokes angle"""

        atom_label = self.atom_site.get()
        process = self.process.get()
        refh = self.reflection_h.get()
        refk = self.reflection_k.get()
        refl = self.reflection_l.get()
        hkl = [refh, refk, refl]
        azih = self.azir_h.get()
        azik = self.azir_k.get()
        azil = self.azir_l.get()
        azir = [azih, azik, azil]
        energy_kev = self.energy_value.get()
        psi = self.psi_value.get()
        rank = self.rank.get()
        time = self.time.get()
        parity = self.parity.get()
        try:
            mk = eval(self.mk.get())
        except SyntaxError:
            mk = None
        try:
            lk = eval(self.lk.get())
        except SyntaxError:
            lk = None
        try:
            sk = eval(self.sk.get())
        except SyntaxError:
            sk = None

        self.xtl.Plot.tensor_scattering_stokes(atom_label, hkl, energy_kev, azir, psi, 45,
                                               process, rank, time, parity, mk, lk, sk)
        plt.show()

    def fun_tensor(self):
        """Display information about tensor scattering on this reflection"""

        atom_label = self.atom_site.get()
        process = self.process.get()
        refh = self.reflection_h.get()
        refk = self.reflection_k.get()
        refl = self.reflection_l.get()
        hkl = [refh, refk, refl]
        azih = self.azir_h.get()
        azik = self.azir_k.get()
        azil = self.azir_l.get()
        azir = [azih, azik, azil]
        energy_kev = self.energy_value.get()
        psi = self.psi_value.get()
        rank = self.rank.get()
        time = self.time.get()
        parity = self.parity.get()
        try:
            mk = eval(self.mk.get())
        except SyntaxError:
            mk = None
        try:
            lk = eval(self.lk.get())
        except SyntaxError:
            lk = None
        try:
            sk = eval(self.sk.get())
        except SyntaxError:
            sk = None

        outstr = self.xtl.Scatter.print_tensor_scattering(atom_label, hkl, energy_kev, azir, psi,
                                                          process, rank, time, parity, mk, lk, sk)
        StringViewer(outstr, 'Tensor Intensities %s' % self.xtl.name)

    def fun_refs(self):
        """Display tensor intensities for all reflections at this energy"""

        atom_label = self.atom_site.get()
        process = self.process.get()
        refh = self.reflection_h.get()
        refk = self.reflection_k.get()
        refl = self.reflection_l.get()
        hkl = [refh, refk, refl]
        azih = self.azir_h.get()
        azik = self.azir_k.get()
        azil = self.azir_l.get()
        azir = [azih, azik, azil]
        energy_kev = self.energy_value.get()
        psi = self.psi_value.get()
        rank = self.rank.get()
        time = self.time.get()
        parity = self.parity.get()
        try:
            mk = eval(self.mk.get())
        except SyntaxError:
            mk = None
        try:
            lk = eval(self.lk.get())
        except SyntaxError:
            lk = None
        try:
            sk = eval(self.sk.get())
        except SyntaxError:
            sk = None

        outstr = self.xtl.Scatter.print_tensor_scattering_refs(atom_label, energy_kev, azir, psi,
                                                               process, rank, time, parity, mk, lk, sk)
        #outstr = self.xtl.Scatter.print_tensor_scattering_refs_max(atom_label, energy_kev, azir,
        #                                                           process, rank, time, parity, mk, lk, sk)
        StringViewer(outstr, 'Tensor Intensities %s' % self.xtl.name)

