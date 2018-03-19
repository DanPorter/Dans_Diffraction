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

Version 1.0
Last updated: 02/03/18

Version History:
02/03/18 1.0    Version History started.
"""

from Dans_Diffraction import functions_general as fg
from Dans_Diffraction import functions_plotting as fp
from Dans_Diffraction import functions_crystallography as fc
from Dans_Diffraction.classes_crystal import Crystal,Multi_Crystal
from Dans_Diffraction.classes_structures import Structures

# Build 
structure_list = Structures()

if __name__ == '__main__':
    
    xtl = Crystal()