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

Version 1.4
Last updated: 13/07/19

Version History:
02/03/18 1.0    Version History started.
30/05/18 1.1    Fdmnes added
08/06/18 1.2    Python3 now fully supported
23/02/19 1.3    Graphical user intrface and magnetic x-ray scattering now implemented
13/07/19 1.4    FDMNES GUI functionality added
"""

# Set TkAgg environment
import matplotlib
matplotlib.use('TkAgg')
# Dans Diffraction
from . import functions_general as fg
from . import functions_plotting as fp
from . import functions_crystallography as fc
from .classes_crystal import Crystal, MultiCrystal
from .classes_structures import Structures
from .classes_fdmnes import fdmnes_checker
if fdmnes_checker():
    from .classes_fdmnes import Fdmnes, FdmnesAnalysis

# GUI (requires tkinter)
try:
    from .classes_gui import Crystalgui as Startgui
except ImportError:
    print('GUI functionality not available, you need to install tkinter.')

__version__ = '1.4'
__date__ = '13/07/19'


# Build
structure_list = Structures()


# FDMNES Activation
def activate_fdmnes():
    """To activate FDMNES functionality"""
    fdmnes_checker(activate=True)
    if fdmnes_checker():
        from .classes_fdmnes import Fdmnes, FdmnesAnalysis
