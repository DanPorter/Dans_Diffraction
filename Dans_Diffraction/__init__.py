"""
Dans_Diffraction
Python package for loading crystal structures from cif files and calculating diffraction information

Usage:
    ***In Python***
    from Dans_Diffraction import Crystal
    f = '/location/of/file.cif'
    xtl = Crystal(f)
    
Usage:
    ***From Terminal***
    cd /location/of/file
    ipython -i -m -matplotlib tk Dans_Diffraction

By Dan Porter, PhD
Diamond
2017

Version 1.6
Last updated: 20/02/20

Version History:
02/03/18 1.0    Version History started.
30/05/18 1.1    Fdmnes added
08/06/18 1.2    Python3 now fully supported
23/02/19 1.3    Graphical user intrface and magnetic x-ray scattering now implemented
13/07/19 1.4    FDMNES GUI functionality added
13/12/19 1.5    Multiple Scattering added, tkGUI refactored, Startgui changed to start_gui
20/02/20 1.6    Tensor Scattering added
"""

# Set TkAgg environment
#import matplotlib
#matplotlib.use('TkAgg')

# Dans Diffraction
from . import functions_general as fg
from . import functions_plotting as fp
from . import functions_crystallography as fc
from .classes_crystal import Crystal, MultiCrystal
from .classes_structures import Structures

# FDMNES
from .classes_fdmnes import fdmnes_checker
if fdmnes_checker():
    from .classes_fdmnes import Fdmnes, FdmnesAnalysis


__version__ = '1.6'
__date__ = '20/02/20'


# Build
structure_list = Structures()


# tkGUI Activation
def start_gui(xtl=None):
    """Start GUI window (requires tkinter)"""
    try:
        from .tkgui import CrystalGui
        CrystalGui(xtl)
    except ImportError:
        print('GUI functionality not available, you need to install tkinter.')


# FDMNES Activation
def activate_fdmnes():
    """To activate FDMNES functionality"""
    fdmnes_checker(activate=True)
    if fdmnes_checker():
        from .classes_fdmnes import Fdmnes, FdmnesAnalysis
