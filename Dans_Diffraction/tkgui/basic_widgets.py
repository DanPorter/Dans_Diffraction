"""
Basic widgets and gui parameters
"""

import sys
import re

if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
else:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox

# Fonts
TF = ['Palatino', 12]  # ["Times", 12]
BF = ['Palatino', 14]  # ["Times", 14]
SF = ['Palatino', 14]  # ["Times New Roman", 14]
LF = ['Palatino', 14]  # ["Times", 14]
MF = ["Courier", 8]  # fixed distance format
HF = ['Courier', 12]
TTF = ("Helvetica", 10, "bold italic")
# Colours - background
bkg = 'snow'  # background colour
ety = 'white'  # entry box
btn = 'azure'  # buttons 'light slate blue'
opt = 'azure'  # option background  'light slate blue'
btn2 = 'gold'  # main button
TTBG = 'light grey'  # Title background
# Colours - active
btn_active = 'grey'
opt_active = 'grey'  # activeBackground
# Colours - Fonts
txtcol = 'black'  # activeForeground, foreground
btn_txt = 'black'
ety_txt = 'black'
opt_txt = 'black'
TTFG = 'red'


def dark_mode():
    """Change the colour scheme"""
    global bkg, txtcol, opt_active, opt, ety, ety_txt, btn, btn_active, btn2
    bkg = '#2b2b2b'
    txtcol = '#afb1b3'  # activeForeground, foreground
    opt_active = '#3c3f41'  # activeBackground
    opt = '#3c3f41'
    ety = '#3c3f41'
    ety_txt = 'white'
    btn = '#3b3e40'
    btn_active = '#81888c'
    btn2 = 'black'


def light_mode():
    """Change the colour scheme"""
    global bkg, txtcol, opt_active, opt, ety, ety_txt, btn, btn_active, btn2
    bkg = 'snow'
    txtcol = 'black'  # activeForeground, foreground
    opt_active = 'grey'  # activeBackground
    opt = 'azure'
    ety = 'white'
    ety_txt = 'black'
    btn = 'azure'
    btn_active = 'grey'
    btn2 = 'gold'


def popup_message(parent, title, message):
    """Create a message box"""
    root = tk.Toplevel(parent)
    root.title(title)
    # frame = tk.Frame(root)
    tk.Label(root, text=message, padx=20, pady=20).pack()
    # root.after(2000, root.destroy)
    return root


def popup_about():
    """Create about message"""
    from Dans_Diffraction import version_info, module_info
    msg = "%s\n\n" \
          "A Python package for loading crystal structures from " \
          "cif files and calculating diffraction information.\n\n" \
          "Module Info:\n%s\n\n" \
          "By Dan Porter, Diamond Light Source Ltd" % (version_info(), module_info())
    messagebox.showinfo('About', msg)


def popup_help():
    """Create help message"""
    from Dans_Diffraction import doc_str
    return StringViewer(doc_str(), 'Dans_Diffraction Help', width=121)


def topmenu(root, menu_dict):
    """
    Add a menubar to tkinter frame
    :param root: tkinter root
    :param menu_dict: {Menu name: {Item name: function}}
    :return: None
    """
    # Setup menubar
    menubar = tk.Menu(root)

    for item, obj in menu_dict.items():
        if isinstance(obj, dict):
            men = tk.Menu(menubar, tearoff=0)
            for label, function in obj.items():
                men.add_command(label=label, command=function)
            menubar.add_cascade(label=item, menu=men)
        else:
            menubar.add_command(label=item,command=obj)
    root.config(menu=menubar)


def menu_docs():
    """Open local docs"""
    import os
    import webbrowser
    docs_dir = os.path.join(os.path.split(__file__)[0], '../../docs/Dans_Diffraction.html')
    docs_dir = os.path.abspath(docs_dir)
    webbrowser.open_new_tab(docs_dir)


def menu_github():
    """Open GitHub page"""
    import webbrowser
    webbrowser.open_new_tab("https://github.com/DanPorter/Dans_Diffraction#dans_diffaction")


def search_re(pattern, text):
    matches = []
    text = text.splitlines()
    for i, line in enumerate(text):
        for match in re.finditer(pattern, line):
            if match.groups():
                matches.append((f"{i + 1}.{match.span(1)[0]}", f"{i + 1}.{match.span(1)[1]}"))
            else:
                matches.append((f"{i + 1}.{match.start()}", f"{i + 1}.{match.end()}"))
    return matches


def text_search(text: tk.Text, search_pattern: str, highlight_colour='yellow'):
    """
    Find string patterns in tk Text object and highlight them
    :param text: tk.Text object
    :param search_pattern: str text to search for or Regular expression
    :param highlight_colour: str color spec
    :return: None
    """
    # Remove all tags so they can be redrawn
    for tag in text.tag_names():
        text.tag_remove(tag, "1.0", tk.END)

    if not search_pattern: return

    # Add tags where the search_re function found the pattern
    for i, (start, end) in enumerate(search_re(search_pattern, text.get('1.0', tk.END))):
        text.tag_add(f'{i}', start, end)
        text.tag_config(f'{i}', background=highlight_colour)
    # Scroll to first instance
    index = text.search(search_pattern, '1.0')
    if index:
        text.see(index)


class StringViewer:
    """
    Simple GUI that displays strings
        StringViewer(string, title, width)
    """

    def __init__(self, string, title='', width=40):
        """Initialise"""
        # Create Tk inter instance
        self.title = title
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

        # --- Search ---
        self.search_str = tk.StringVar(self.root, '')
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.BOTTOM, fill=tk.X)
        var = tk.Entry(frm1, textvariable=self.search_str)
        var.pack(side=tk.LEFT)
        var.bind('<Return>', self.fun_search)
        var.bind('<KP_Enter>', self.fun_search)
        var = tk.Button(frm1, text='Search', font=BF, command=self.fun_search, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm1, text='SaveAs', font=BF, command=self.fun_saveas, bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT)

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

    def fun_search(self, event=None):
        """Search"""
        text_search(self.text, self.search_str.get())

    def fun_saveas(self):
        """Save as text file"""
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title='Save as text file',
            initialfile='%s.txt' % self.title,
            filetypes=[('Text file', '*.txt')],
        )
        if filename:
            with open(filename, 'wt') as f:
                f.write(self.text.get('1.0', tk.END))
            print('File written: %s' % filename)

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

    out = SelectionBox(['field1','field2','field3'], current_selection=['field2'], title='', multiselect=False).show()
    # Make selection and press "Select" > box disappears
    out = ['list','of','strings']
    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initilisation-----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self, parent, data_fields, current_selection=(), title='Make a selection', multiselect=True):
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
        self.lst_data.bind('<<ListboxSelect>>', self.fun_listboxselect)
        self.lst_data.bind('<Double-Button-1>', self.fun_exitbutton)

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

        self.numberoffields = tk.StringVar(self.root, '%3d Selected Fields' % len(self.initial_selection))
        var = tk.Label(frm_btn, textvariable=self.numberoffields, width=20)
        var.pack(side=tk.LEFT)
        btn_exit = tk.Button(frm_btn, text='Select', font=BF, command=self.fun_exitbutton, bg=btn,
                             activebackground=btn_active)
        btn_exit.pack(side=tk.RIGHT)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        # self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
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
        for n, item in enumerate(self.data_fields):
            if re.search(r'\b%s\b' % search_str, item.lower()):  # whole word search
                self.lst_data.select_set(n)
                view_idx = n
        # if nothing found, search anywhere
        if view_idx is None:
            for n, item in enumerate(self.data_fields):
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
        self.output = [self.data_fields[n] for n in selection]
        self.root.destroy()

    def f_exit(self, event=None):
        """Closes the current data window"""
        self.output = self.initial_selection
        self.root.destroy()
