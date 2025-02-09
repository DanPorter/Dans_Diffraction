"""
Dans_Diffraction
Python package for loading crystal structures from cif files and calculating diffraction information

Usage:
    ***From Terminal***
    cd /location/of/file
    ipython -i -m --matplotlib tk Dans_Diffraction

For GUI use:
    ipython -m Dans_Diffraction gui

To Parse a cif:
    ipython -m Dans_Diffraction 'somefile.cif'

By Dan Porter, PhD
Diamond
2023
"""
if __name__ == '__main__':

    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import Dans_Diffraction as dif

    print('\nDans_Diffraction version %s, %s\n By Dan Porter, Diamond Light Source Ltd.'%(dif.__version__, dif.__date__))
    print('See help(dif.Crystal) for info, or dif.start_gui() to get started!')
    xtl = dif.Crystal()

    for arg in sys.argv:
        if 'cif' in arg.lower():
            xtl = dif.Crystal(arg)
            print(xtl.info())
        elif arg in dif.structure_list.list:
            xtl = getattr(dif.structure_list, arg)()
            print(xtl.info())
        elif 'gui' in arg.lower():
            xtl.start_gui()
        elif 'properties' in arg.lower():
            from Dans_Diffraction.tkgui.properties import XrayInteractionsGui
            XrayInteractionsGui()
        elif 'fdmnes' in arg.lower():
            from Dans_Diffraction.tkgui.fdmnes import AnaFDMNESgui
            AnaFDMNESgui()

