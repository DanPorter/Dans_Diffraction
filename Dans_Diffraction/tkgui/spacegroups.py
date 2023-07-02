"""
Sacegroup Viewer GUI
"""

import matplotlib.pyplot as plt
import numpy as np

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .. import functions_plotting as fp
from ..classes_crystal import Symmetry
from .basic_widgets import tk, StringViewer, SelectionBox, messagebox
from .basic_widgets import (TF, BF, SF, LF, HF, TTF,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)


class SpaceGroupGui:
    """
    View and edit the symmetry operations
    """

    def __init__(self, symmetry=None, xtl=None):
        """"Initialise"""
        if symmetry is None:
            self.Symmetry = Symmetry()
        else:
            self.Symmetry = symmetry
        self.xtl = xtl
        self.sg = self.Symmetry.spacegroup_dict

        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Spacegroup Viewer')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.spacegroup_symbol = tk.StringVar(self.root, 'P1')
        self.spacegroup_number = tk.StringVar(self.root, '1')
        self.nops = tk.IntVar(self.root, 1)  # number of operations
        self.ncentre = tk.IntVar(self.root, 1)  # number of centring operations
        self.nspecial = tk.IntVar(self.root, 1)  # number of special positions

        # --- Spacegroup entry ---
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP)
        var = tk.Label(frame, text='Spacegroup: ', font=TTF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(frame, textvariable=self.spacegroup_number, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_spacegroup)
        var.bind('<KP_Enter>', self.fun_spacegroup)
        var = tk.Entry(frame, textvariable=self.spacegroup_symbol, font=TF, width=14, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=5)
        var.bind('<Return>', self.fun_spacegroup_symbol)
        var.bind('<KP_Enter>', self.fun_spacegroup_symbol)
        # Spacegroup Buttons
        var = tk.Button(frame, text='Choose\nSpacegroup', command=self.fun_ch_spacegroup, height=2, font=BF, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Button(frame, text='Choose\nSubgroup', command=self.fun_ch_subgroup, height=2,
                        font=BF, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5, pady=5)
        var = tk.Button(frame, text='Choose Magnetic\nSpacegroup', command=self.fun_ch_maggroup, height=2,
                        font=BF, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Spacegroup info ---
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, pady=4)
        var = tk.Label(frame, text='Operations: ', font=TTF, relief=tk.RIDGE)
        var.pack(side=tk.LEFT)
        var = tk.Label(frame, textvariable=self.nops, font=TTF, width=3, relief=tk.RIDGE)
        var.pack(side=tk.LEFT)
        var = tk.Label(frame, text='Centring: ', font=TTF, relief=tk.RIDGE)
        var.pack(side=tk.LEFT)
        var = tk.Label(frame, textvariable=self.ncentre, font=TTF, width=3, relief=tk.RIDGE)
        var.pack(side=tk.LEFT)
        var = tk.Label(frame, text='Special Positions: ', font=TTF, relief=tk.RIDGE)
        var.pack(side=tk.LEFT)
        var = tk.Label(frame, textvariable=self.nspecial, font=TTF, width=3, relief=tk.RIDGE)
        var.pack(side=tk.LEFT)
        # Wyckoff Button
        var = tk.Button(frame, text='Wyckoff Sites', command=self.fun_wyckoff, font=BF, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.RIGHT, padx=2)

        # --- Symmetry operations table ---
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

        # --- Options ---
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP)

        if self.xtl is not None:
            var = tk.Button(frame, text='Update', command=self.fun_update, height=2, font=BF, bg=btn,
                            activebackground=btn_active)
            var.pack(side=tk.LEFT)

        # uvw entry
        var = tk.Label(frame, text='(u v w)=')
        var.pack(side=tk.LEFT)
        self.uvw = tk.StringVar(frame, '0.5 0 0')
        var = tk.Entry(frame, textvariable=self.uvw, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_symuvw)
        var.bind('<KP_Enter>', self.fun_symuvw)
        var = tk.Button(frame, text='Symmetric\nPositions', font=BF, bg=btn, activebackground=btn_active,
                        command=self.fun_symuvw)
        var.pack(side=tk.LEFT)

        # hkl entry
        var = tk.Label(frame, text='(h k l)=')
        var.pack(side=tk.LEFT)
        self.hkl = tk.StringVar(frame, '1 0 0')
        var = tk.Entry(frame, textvariable=self.hkl, font=TF, width=12, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_symhkl)
        var.bind('<KP_Enter>', self.fun_symhkl)
        var = tk.Button(frame, text='Symmetric\nReflections', font=BF, bg=btn, activebackground=btn_active,
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
            messagebox.showwarning(
                parent=self.root,
                title='Spacegroups',
                message='Warning: The number of symmetry and magnetic operations are not the same!'
            )
            return

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
        if self.xtl is not None:
            self.xtl.generate_structure()

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
            symstr += '%25s\n' % sym
            magstr += '%25s\n' % mag

        # Insert string in text box
        self.text_2.insert(tk.END, symstr)
        self.text_3.insert(tk.END, magstr)

        # Update spacegroup name + number
        self.spacegroup_symbol.set(self.Symmetry.spacegroup)
        self.spacegroup_number.set(self.Symmetry.spacegroup_number)
        self.nops.set(len(self.Symmetry.symmetry_operations))
        if 'positions centring' in self.sg:
            self.ncentre.set(len(self.sg['positions centring']))
        else:
            self.ncentre.set(0)
        if 'positions coordinates' in self.sg:
            self.nspecial.set(len(self.sg['positions coordinates']))
        else:
            self.nspecial.set(0)

    def fun_spacegroup(self, event=None):
        """Load spacegroup symmetry"""
        sgn = int(float(self.spacegroup_number.get()))
        self.Symmetry.load_spacegroup(sgn)
        self.sg = self.Symmetry.spacegroup_dict
        self.fun_set()

    def fun_spacegroup_symbol(self, event=None):
        """Load spacegroup symmetry"""
        sgs = self.spacegroup_symbol.get()
        check = fc.find_spacegroup(sgs)
        if check:
            self.Symmetry.load_spacegroup(sg_dict=check)
            self.sg = self.Symmetry.spacegroup_dict
            self.fun_set()
        else:
            messagebox.showwarning(
                parent=self.root,
                title='Spacegroup Viewer',
                message="Can't find spacegroup '%s'" % sgs
            )

    def fun_ch_spacegroup(self):
        """Button Select Spacegroup"""
        current = int(float(self.spacegroup_number.get()))
        sg_list = fc.spacegroup_list().split('\n')
        current_selection = [sg_list[current-1]]
        selection = SelectionBox(self.root, sg_list, current_selection, 'Select SpaceGroup', False).show()
        if len(selection) > 0:
            new_sg = int(selection[0].split()[0])
            self.Symmetry.load_spacegroup(new_sg)
            self.sg = self.Symmetry.spacegroup_dict
            self.fun_set()

    def fun_ch_subgroup(self):
        """Button select subgroup"""
        current = int(float(self.spacegroup_number.get()))  # str > float > int
        sbg_list = fc.spacegroup_subgroups_list(current).split('\n')
        selection = SelectionBox(self.root, sbg_list, [], 'Select SpaceGroup Subgroup', False).show()
        if len(selection) > 0:
            new_sg = int(selection[0].split()[3])
            self.Symmetry.load_spacegroup(new_sg)
            self.sg = self.Symmetry.spacegroup_dict
            self.fun_set()

    def fun_ch_maggroup(self):
        """Button Select Magnetic Spacegroup"""
        current = int(float(self.spacegroup_number.get()))
        msg_list = fc.spacegroup_magnetic_list(current).split('\n')
        selection = SelectionBox(self.root, msg_list, [], 'Select Magnetic SpaceGroup', False).show()
        if len(selection) > 0:
            new_sg = selection[0].split()[3]
            self.Symmetry.load_magnetic_spacegroup(new_sg)
            self.sg = self.Symmetry.spacegroup_dict
            self.fun_set()

    def fun_wyckoff(self):
        """Button display Wyckoff sites"""
        out = self.Symmetry.print_wyckoff_sites()
        StringViewer(out, 'Wyckoff Sites', width=80)

    def fun_symuvw(self, event=None):
        """create symmetric uvw position"""
        self.update()
        uvw = self.uvw.get()
        uvw = uvw.replace(',', ' ')  # remove commas
        uvw = uvw.replace('(', '').replace(')', '')  # remove brackets
        uvw = uvw.replace('[', '').replace(']', '')  # remove brackets
        uvw = np.fromstring(uvw, sep=' ')
        out = self.Symmetry.print_symmetric_coordinate_operations(uvw)
        StringViewer(out, 'Symmetric Positions', width=80)

    def fun_symhkl(self, event=None):
        """create symmetric hkl reflections"""
        self.update()
        hkl = self.hkl.get()
        hkl = hkl.replace(',', ' ')  # remove commas
        hkl = hkl.replace('(', '').replace(')', '')  # remove brackets
        hkl = hkl.replace('[', '').replace(']', '')  # remove brackets
        hkl = np.fromstring(hkl, sep=' ')
        out = self.Symmetry.print_symmetric_vectors(hkl)
        StringViewer(out, 'Symmetric Reflections', width=80)

    def fun_update(self):
        """Update button"""

        self.update()

        # Close window
        self.root.destroy()
