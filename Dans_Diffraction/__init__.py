"""
Dans_Diffraction
Python package for loading crystal structures from cif files and calculating diffraction information

Usage:
    ***In Python***
    import Dans_Diffraction as dif
    f = '/location/of/file.cif'
    xtl = dif.Crystal(f)
    
Usage:
    ***From Terminal***
    cd /location/of/file
    ipython -i -m -matplotlib tk Dans_Diffraction

By Dan Porter, PhD
Diamond
2017

Version 1.8.2
Last updated: 26/05/20

Version History:
02/03/18 1.0    Version History started.
30/05/18 1.1    Fdmnes added
08/06/18 1.2    Python3 now fully supported
23/02/19 1.3    Graphical user intrface and magnetic x-ray scattering now implemented
13/07/19 1.4    FDMNES GUI functionality added
13/12/19 1.5    Multiple Scattering added, tkGUI refactored, Startgui changed to start_gui
20/02/20 1.6    Tensor Scattering added
31/03/20 1.7    Refactored multicrystal methods, other minor changes, improved powder diffraction
19/04/20 1.7.1  Added write_cif + spacegroup file + functions
02/05/20 1.8    Updated readcif, added heavy atom properties, added magnetic spacegroups
12/05/20 1.8.1  Updated readcif, added atomic_scattering_factors and classes_orbitals
26/05/20 1.8.2  Updated copyright, removed tensor scattering. Updated magnetic spacegroups

-----------------------------------------------------------------------------
   Copyright 2020 Diamond Light Source Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Files in this package covered by this licence:
    classes_crystal.py
    classes_scattering.py
    classes_plotting.py
    classes_properties.py
    classes_multicrystal.py
    classes_orbitals.py
    functions_general.py
    functions_plotting.py
    functions_crystallography.py
    tkgui/*.py
Other files are either covered by their own licence or not licenced for other use.

 Dr Daniel G Porter, dan.porter@diamond.ac.uk
 www.diamond.ac.uk
 Diamond Light Source, Chilton, Didcot, Oxon, OX11 0DE, U.K.
"""

# Set TkAgg environment
#import matplotlib
#matplotlib.use('TkAgg')

# Dans Diffraction
from . import functions_general as fg
from . import functions_plotting as fp
from . import functions_crystallography as fc
from .classes_crystal import Crystal
from .classes_multicrystal import MultiCrystal
from .classes_structures import Structures
from .functions_crystallography import readcif

# FDMNES
from .classes_fdmnes import fdmnes_checker
if fdmnes_checker():
    from .classes_fdmnes import Fdmnes, FdmnesAnalysis


__version__ = '1.8.2'
__date__ = '27/05/20'


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
