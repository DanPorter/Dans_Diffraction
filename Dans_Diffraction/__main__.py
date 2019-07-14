"""
Dans_Diffraction
Python package for loading crystal structures from cif files and calculating diffraction information

Usage:
    ***From Terminal***
    cd /location/of/file
    ipython -i -m -matplotlib tk Dans_Diffraction

By Dan Porter, PhD
Diamond
2017
"""
if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import Dans_Diffraction as dif

    print('\nDans_Diffraction version %s, %s\n By Dan Porter, Diamond Light Source Ltd.'%(dif.__version__, dif.__date__))
    print('See help(dif.Crystal) for info, or dif.Startgui() to get started!')
    xtl = dif.Crystal()

