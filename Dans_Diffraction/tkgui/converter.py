"""
Unit converter GUI
"""

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .. import Crystal
from .basic_widgets import tk, SelectionBox
from .basic_widgets import (TF, BF, SF, LF, HF, TTF,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)
from .crystal import LatparGui

# Mass in kg
MASSES = {
    'X-Ray': 0,
    'Electron': fg.me,
    'Neutron': fg.mn
}
# Mass units in kg
M_UNITS = {
    'MeV': (fg.e * 1e6) / fg.c ** 2,
    'kg': 1.0,
}
# Energy units in keV
E_UNITS = {
    'keV': 1.0,
    'eV': 1.0e-3,
    'meV': 1.0e-6,
    'Hz': fg.h / (fg.e * 1000),
    'GHz': 1e9 * fg.h / (fg.e * 1000),
    'EHz': 1e18 * fg.h / (fg.e * 1000),
}
# Wavelength units in Angstrom
W_UNITS = {
    u'\u212B': 1.0,
    'nm': 10.
}
# Entry format
FMT = '%.5g'


class UnitConverter:
    """
    Convert various units
    """

    def __init__(self):
        """"Initialise"""

        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Unit Converter')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.radiation = tk.StringVar(self.root, 'X-Ray')
        self.mass = tk.StringVar(self.root, '0')
        self.energy = tk.StringVar(self.root, '8.0')
        self.wavelength = tk.StringVar(self.root, '1.5498')
        self.mass_unit = tk.StringVar(self.root, 'MeV')
        self.energy_unit = tk.StringVar(self.root, 'keV')
        self.wavelength_unit = tk.StringVar(self.root, u'\u212B')
        self.edge = tk.StringVar(self.root, 'Edge')

        self._mass_unit = tk.StringVar(self.root, 'MeV')
        self._energy_unit = tk.StringVar(self.root, 'keV')
        self._wavelength_unit = tk.StringVar(self.root, u'\u212B')

        self.xtl = Crystal()
        self.xtl.Cell.latt(5.431)
        self.hkl = tk.StringVar(self.root, '1 0 0')
        self.latt = tk.StringVar(self.root, '5.431')
        self.tth = tk.StringVar(self.root, '16.407')
        self.dspace = tk.StringVar(self.root, '5.4307')
        self.qmag = tk.StringVar(self.root, '1.157')

        self.fwhm_deg = tk.StringVar(self.root, '0.1')
        self.fwhm_q = tk.StringVar(self.root, '0.008')

        self.dom_size = tk.StringVar(self.root, '807.44')
        self.dom_size_unit = tk.StringVar(self.root, u'\u212B')
        self._dom_size_unit = tk.StringVar(self.root, u'\u212B')

        self.refine_hkl = False  # used in self.update_hkl, altered in self.but_hkl_switch

        # --- Mass / Wavelength / Energy ---
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=6, pady=2)
        var = tk.Label(frame, text='Mass: ', font=TTF, width=10)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.mass, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_mass)
        var.bind('<KP_Enter>', self.fun_mass)
        var = tk.OptionMenu(frame, self.mass_unit, *M_UNITS, command=self.fun_mass_unit)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(frame, self.radiation, *MASSES, command=self.fun_mass_selection)
        var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=6, pady=2)
        var = tk.Label(frame, text='Wavelength: ', font=TTF, width=10)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.wavelength, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_wavelength)
        var.bind('<KP_Enter>', self.fun_wavelength)
        var = tk.OptionMenu(frame, self.wavelength_unit, *W_UNITS, command=self.fun_wavelength_unit)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=6, pady=2)
        var = tk.Label(frame, text='Energy: ', font=TTF, width=10)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.energy, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_energy)
        var.bind('<KP_Enter>', self.fun_energy)
        var = tk.OptionMenu(frame, self.energy_unit, *E_UNITS, command=self.fun_energy_unit)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frame, textvariable=self.edge, command=self.fun_edge, font=BF, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Reflection / Angle ---
        frame = tk.Frame(self.root, relief=tk.RIDGE)  # horizontal line
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=4)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=6, pady=6)
        var = tk.Label(frame, text='hkl: ', font=TTF, width=4, borderwidth=0, relief="ridge")
        var.pack(side=tk.LEFT)
        var.bind("<Button-1>", self.but_hkl_switch)
        self.hkl_label = var
        var = tk.Entry(frame, textvariable=self.hkl, font=TF, width=20, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_hkl)
        var.bind('<KP_Enter>', self.fun_hkl)
        var = tk.Label(frame, text='Lattice: ', font=TTF, borderwidth=2, relief="ridge")
        var.pack(side=tk.LEFT, padx=6)
        var.bind("<Button-1>", self.but_latt_switch)
        self.lattice_label = var
        var = tk.Entry(frame, textvariable=self.latt, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_setlat)
        var.bind('<KP_Enter>', self.fun_setlat)
        var = tk.Label(frame, text=u'\u212B', font=TF)
        var.pack(side=tk.LEFT)
        var = tk.Button(frame, text='Latt', command=self.fun_latt, font=BF, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=4)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=6, pady=6)
        var = tk.Label(frame, text=u'2\u03B8:', font=TTF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.tth, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_tth)
        var.bind('<KP_Enter>', self.fun_tth)
        var = tk.Label(frame, text='Deg', font=TF)
        var.pack(side=tk.LEFT)

        var = tk.Label(frame, text='d-space:', font=TTF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.dspace, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_dspace)
        var.bind('<KP_Enter>', self.fun_dspace)
        var = tk.Label(frame, text=u'\u212B', font=TF)
        var.pack(side=tk.LEFT)

        var = tk.Label(frame, text=u'Q:', font=TTF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.qmag, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_qmag)
        var.bind('<KP_Enter>', self.fun_qmag)
        var = tk.Label(frame, text=u'\u212B\u207B\u00B9', font=TF)
        var.pack(side=tk.LEFT)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=6, pady=6)
        var = tk.Label(frame, text='Peak Width:', font=TTF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.fwhm_deg, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=6)
        var.bind('<Return>', self.fun_fwhm_deg)
        var.bind('<KP_Enter>', self.fun_tth)
        var = tk.Label(frame, text='Deg   ', font=TF)
        var.pack(side=tk.LEFT)

        var = tk.Entry(frame, textvariable=self.fwhm_q, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=6)
        var.bind('<Return>', self.fun_fwhm_q)
        var.bind('<KP_Enter>', self.fun_tth)
        var = tk.Label(frame, text=u'\u212B\u207B\u00B9', font=TF)
        var.pack(side=tk.LEFT)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=6, pady=6)
        var = tk.Label(frame, text='Domain Size:', font=TTF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.dom_size, font=TF, width=8, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=6)
        var.bind('<Return>', self.fun_fwhm_dom)
        var.bind('<KP_Enter>', self.fun_tth)
        var = tk.OptionMenu(frame, self.dom_size_unit, *W_UNITS, command=self.fun_dom_size_unit)
        var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

    def get_mass_kg(self):
        unit = self._mass_unit.get()
        return float(self.mass.get()) * M_UNITS[unit]

    def get_energy_kev(self):
        unit = self._energy_unit.get()
        return float(self.energy.get()) * E_UNITS[unit]

    def get_wavelength_a(self):
        unit = self._wavelength_unit.get()
        return float(self.wavelength.get()) * W_UNITS[unit]

    def set_mass(self, mass_kg):
        unit = self.mass_unit.get()
        self.mass.set(FMT % (mass_kg / M_UNITS[unit]))
        self._mass_unit.set(unit)

    def set_energy(self, energy_kev):
        unit = self.energy_unit.get()
        self.energy.set(FMT % (energy_kev / E_UNITS[unit]))
        self._energy_unit.set(unit)

    def set_wavelength(self, wavelength_a):
        unit = self.wavelength_unit.get()
        self.wavelength.set(FMT % (wavelength_a / W_UNITS[unit]))
        self._wavelength_unit.set(unit)

    def fun_mass_selection(self, event=None):
        new_radiation = self.radiation.get()
        self.set_mass(MASSES[new_radiation])
        self.fun_mass()

    def fun_mass(self, event=None):
        mass_kg = self.get_mass_kg()
        wavelength_a = self.get_wavelength_a()
        if mass_kg == 0:
            energy_kev = fc.wave2energy(wavelength_a)
        else:
            energy_kev = fc.debroglie_energy(wavelength_a, mass_kg)
        self.set_energy(energy_kev)
        self.fun_hkl()

    def fun_mass_unit(self, event=None):
        self.set_mass(self.get_mass_kg())
        self.fun_mass()

    def fun_wavelength(self, event=None):
        self.fun_mass()

    def fun_wavelength_unit(self, event=None):
        self.set_wavelength(self.get_wavelength_a())
        self.fun_mass()

    def fun_energy(self, event=None):
        mass_kg = self.get_mass_kg()
        energy_kev = self.get_energy_kev()
        if mass_kg == 0:
            wavelength_a = fc.energy2wave(energy_kev)
        else:
            wavelength_a = fc.debroglie_wavelength(energy_kev, mass_kg)
        self.set_wavelength(wavelength_a)
        self.fun_hkl()

    def fun_energy_unit(self, event=None):
        self.set_energy(self.get_energy_kev())
        self.fun_energy()

    def fun_edge(self):
        """Edge button"""
        edge_list = ['K', 'L3', 'L2', 'L1', 'M5', 'M4', 'M3', 'M2', 'M1']
        elements = fc.atom_properties(None, ['Element', 'Name'] + edge_list)
        ele_list = []
        ele = '%3s: %12s: %2s %7.3f keV'
        for element in elements:
            ele_list += [ele % (element[0], element[1], edge, element[2+n])
                         for n, edge in enumerate(edge_list) if element[2+n] > 0]

        choose = SelectionBox(self.root, ele_list, multiselect=False, title='Select Absorption Edge (keV)').show()
        if choose:
            self.set_energy(float(choose[0][-12:-3].strip()))
            self.fun_energy()
            self.edge.set('%s %s' % (choose[0][1:3], choose[0][19:21]))

    def fun_latt(self):
        LatparGui(self.xtl)

    def fun_setlat(self, event=None):
        latt = float(self.latt.get())
        self.xtl.Cell.latt(latt)  # set cubic lattice
        self.fun_hkl()

    def but_hkl_switch(self, event=None):
        self.lattice_label.config(borderwidth=0)
        self.hkl_label.config(borderwidth=2)
        self.refine_hkl = True

    def but_latt_switch(self, event=None):
        self.lattice_label.config(borderwidth=2)
        self.hkl_label.config(borderwidth=0)
        self.refine_hkl = False

    def fun_hkl(self, event=None):
        hkl = fg.str2array(self.hkl.get())
        wavelength_a = self.get_wavelength_a()
        tth = self.xtl.Cell.tth(hkl, wavelength_a=wavelength_a)[0]
        dspace = self.xtl.Cell.dspace(hkl)[0]
        qmag = self.xtl.Cell.Qmag(hkl)[0]
        self.tth.set(FMT % tth)
        self.dspace.set(FMT % dspace)
        self.qmag.set(FMT % qmag)
        self.fun_fwhm_dom()

    def update_hkl(self, qmag):
        old_hkl = fg.str2array(self.hkl.get())
        oldq = self.xtl.Cell.calculateQ(old_hkl)[0]
        oldqmag = fg.mag(oldq)
        newq = oldq * qmag / oldqmag
        new_hkl = self.xtl.Cell.indexQ(newq)[0]

        if self.refine_hkl:
            # Update hkl
            hkl_str = ' '.join([FMT % h for h in new_hkl])
            self.hkl.set(hkl_str)
            self.fun_hkl()
        else:
            # update lattice
            old_lat = self.xtl.Cell.lp()
            new_lat = [old_lat[n] * old_hkl[n] / new_hkl[n] if abs(old_hkl[n]) > 0 else old_lat[n] for n in range(3)]
            new_lat.extend(old_lat[3:])
            self.xtl.Cell.latt(new_lat)
            self.latt.set('%.3f' % new_lat[0])
            self.fun_hkl()

    def fun_tth(self, event=None):
        wavelength_a = self.get_wavelength_a()
        tth = float(self.tth.get())
        qmag = fc.calqmag(tth, wavelength_a=wavelength_a)
        self.update_hkl(qmag)

    def fun_dspace(self, event=None):
        qmag = fc.dspace2q(float(self.dspace.get()))
        self.update_hkl(qmag)

    def fun_qmag(self, event=None):
        self.update_hkl(float(self.qmag.get()))

    def fun_fwhm_deg(self, event=None):
        wavelength_a = self.get_wavelength_a()
        tth = float(self.tth.get())
        fwhm_deg = float(self.fwhm_deg.get())
        size_a = fc.scherrer_size(fwhm_deg, twotheta=tth, wavelength_a=wavelength_a)
        fwhm_q = fc.dspace2q(size_a)
        self.fwhm_q.set(FMT % fwhm_q)
        self.dom_size.set(FMT % size_a)

    def fun_fwhm_q(self, event=None):
        wavelength_a = self.get_wavelength_a()
        tth = float(self.tth.get())
        fwhm_q = float(self.fwhm_q.get())
        size_a = fc.dspace2q(fwhm_q)
        fwhm_deg = fc.scherrer_fwhm(size_a, twotheta=tth, wavelength_a=wavelength_a)
        self.fwhm_deg.set(FMT % fwhm_deg)
        self.dom_size.set(FMT % size_a)

    def get_dom_size(self):
        unit = self._dom_size_unit.get()
        return float(self.dom_size.get()) * W_UNITS[unit]

    def fun_fwhm_dom(self, event=None):
        wavelength_a = self.get_wavelength_a()
        tth = float(self.tth.get())
        size_a = self.get_dom_size()
        fwhm_q = fc.q2dspace(size_a)
        fwhm_deg = fc.scherrer_fwhm(size_a, twotheta=tth, wavelength_a=wavelength_a)
        self.fwhm_deg.set(FMT % fwhm_deg)
        self.fwhm_q.set(FMT % fwhm_q)

    def fun_dom_size_unit(self, event=None):
        new_unit = self.dom_size_unit.get()
        size_a = self.get_dom_size()
        new_val = size_a / W_UNITS[new_unit]
        self.dom_size.set(FMT % new_val)
        self._dom_size_unit.set(new_unit)
        self.fun_fwhm_dom()


