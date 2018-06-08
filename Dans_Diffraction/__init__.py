"""
Dans_Diffraction
Python package for loading crystal structures from cif files and calculating diffraction

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

Version 1.1
Last updated: 30/05/18

Version History:
02/03/18 1.0    Version History started.
30/05/18 1.1    Fdmnes added
08/06/18 1.2
"""

from . import functions_general as fg
from . import functions_plotting as fp
from . import functions_crystallography as fc
from .classes_crystal import Crystal, MultiCrystal
from .classes_structures import Structures
from .classes_fdmnes import Fdmnes, FdmnesAnalysis

__version__ = '1.1'

# Build 
structure_list = Structures()

if __name__ == '__main__':
    
    xtl = Crystal()
