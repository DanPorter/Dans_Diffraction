"""
Basic widgets and gui parameters
"""

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

# Fonts
TF = ["Times", 12]
BF = ["Times", 14]
SF = ["Times New Roman", 14]
LF = ["Times", 14]
MF = ["Courier", 8]  # fixed distance format
HF = ['Courier',12]
TTF = ("Helvetica", 10, "bold italic")
# Colours - background
bkg = 'snow'
ety = 'white'
btn = 'azure' #'light slate blue'
opt = 'azure' #'light slate blue'
btn2 = 'gold'
TTBG = 'light grey'
# Colours - active
btn_active = 'grey'
opt_active = 'grey'
# Colours - Fonts
txtcol = 'black'
btn_txt = 'black'
ety_txt = 'black'
opt_txt = 'black'
TTFG = 'red'


class StringViewer:
    """
    Simple GUI that displays strings
        StringViewer(string, title, width)
    """

    def __init__(self, string, title='', width=40):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(title)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        # Textbox height
        height = string.count('\n')
        if height > 40: height = 40

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # --- label ---
        # labframe = tk.Frame(frame,relief='groove')
        # labframe.pack(side=tk.TOP, fill=tk.X)
        # var = tk.Label(labframe, text=label_text,font=SF,justify='left')
        # var.pack(side=tk.LEFT)

        # --- Button ---
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.BOTTOM, fill=tk.X)
        var = tk.Button(frm1, text='Close', font=BF, command=self.fun_close, bg=btn, activebackground=btn_active)
        var.pack(fill=tk.X)

        # --- Text box ---
        frame_box = tk.Frame(frame)
        frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # Scrollbars
        scanx = tk.Scrollbar(frame_box, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)

        scany = tk.Scrollbar(frame_box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)

        # Editable string box
        self.text = tk.Text(frame_box, width=width, height=height, font=HF, wrap=tk.NONE, background=ety)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.text.insert(tk.END, string)

        self.text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)

    def fun_close(self):
        """close window"""
        self.root.destroy()


"------------------------------------------------------------------------"
"----------------------------Selection Box-------------------------------"
"------------------------------------------------------------------------"


class SelectionBox:
    """
    Displays all data fields and returns a selection
    Making a selection returns a list of field strings

    out = SelectionBox(['field1','field2','field3'], current_selection=['field2'], title='', multiselect=False)
    # Make selection and press "Select" > box disappears
    out.output = ['list','of','strings']

    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initilisation-----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self, parent, data_fields, current_selection=[], title='Make a selection', multiselect=True):
        self.data_fields = data_fields
        self.initial_selection = current_selection

        # Create Tk inter instance
        self.root = tk.Toplevel(parent)
        self.root.wm_title(title)
        self.root.minsize(width=100, height=300)
        self.root.maxsize(width=1200, height=1200)
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        self.output = []

        # Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, anchor=tk.N)

        "---------------------------ListBox---------------------------"
        # Eval box with scroll bar
        frm = tk.Frame(frame)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        sclx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        sclx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scly = tk.Scrollbar(frm)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.lst_data = tk.Listbox(frm, font=MF, selectmode=tk.SINGLE, width=60, height=20, bg=ety,
                                   xscrollcommand=sclx.set, yscrollcommand=scly.set)
        self.lst_data.configure(exportselection=True)
        if multiselect:
            self.lst_data.configure(selectmode=tk.EXTENDED)

        # Populate list box
        for k in self.data_fields:
            # if k[0] == '_': continue # Omit _OrderedDict__root/map
            strval = '{}'.format(k)
            self.lst_data.insert(tk.END, strval)

        self.lst_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        for select in current_selection:
            if select in data_fields:
                idx = data_fields.index(select)
                self.lst_data.select_set(idx)

        sclx.config(command=self.lst_data.xview)
        scly.config(command=self.lst_data.yview)

        # self.txt_data.config(xscrollcommand=scl_datax.set,yscrollcommand=scl_datay.set)

        "----------------------------Exit Button------------------------------"
        frm_btn = tk.Frame(frame)
        frm_btn.pack()

        btn_exit = tk.Button(frm_btn, text='Select', font=BF, command=self.f_exit, bg=btn, activebackground=btn_active)
        btn_exit.pack()

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        #self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def show(self):
        """Run the selection box, wait for response"""

        #self.root.deiconify()  # show window
        self.root.wait_window()  # wait for window
        return self.output

    def f_exit(self):
        """Closes the current data window"""
        selection = self.lst_data.curselection()
        self.output = [self.data_fields[n] for n in selection]
        self.root.destroy()

