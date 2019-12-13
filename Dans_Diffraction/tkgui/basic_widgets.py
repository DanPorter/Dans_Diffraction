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
    """

    def __init__(self, string, title=''):
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
        self.text = tk.Text(frame_box, width=40, height=height, font=HF, wrap=tk.NONE, background=ety)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.text.insert(tk.END, string)

        self.text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)

    def fun_close(self):
        """close window"""
        self.root.destroy()
