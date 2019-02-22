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
    ipython -i -pyplot Dans_Diffraction

By Dan Porter, PhD
Diamond
2017

Version 1.2
Last updated: 08/06/18

Version History:
02/03/18 1.0    Version History started.
30/05/18 1.1    Fdmnes added
08/06/18 1.2    Python3 now fully supported
"""

import sys

from . import functions_general as fg
from . import functions_plotting as fp
from . import functions_crystallography as fc
from .classes_crystal import Crystal, MultiCrystal
from .classes_structures import Structures
from .classes_fdmnes import Fdmnes, FdmnesAnalysis

# GUI (requires tkinter)
try:
    from .classes_gui import Crystalgui as Startgui
except ImportError:
    print('GUI functionality not available, you need to install tkinter.')

__version__ = '1.2'


# Build 
structure_list = Structures()

if __name__ == '__main__':
    
    xtl = Crystal()
