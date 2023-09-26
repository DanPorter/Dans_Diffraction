"""
Dans_Diffraction
Python package for loading crystal structures from cif files and calculating diffraction information.

Installation:
$ python -m pip install Dans-Diffraction
or
$ git clone https://github.com/DanPorter/Dans_Diffraction.git
or
$ python -m pip install git+https://github.com/DanPorter/Dans_Diffraction.git

Requirements:
Python 2.7+/3+ with packages: Numpy, Matplotlib, Tkinter
BuiltIn packages used: sys, os, re, glob, warnings, json, itertools

Usage:
    ***In Python/ script***
    import Dans_Diffraction as dif
    f = '/location/of/file.cif'
    xtl = dif.Crystal(f)
    
Usage:
    ***From Terminal***
    $ cd /location/of/file
    $ ipython -i -m -matplotlib tk Dans_Diffraction

GitHub Repo: https://github.com/DanPorter/Dans_Diffraction
Citation DOI: https://doi.org/10.5281/zenodo.8106031

By Dan Porter, PhD
Diamond
2017

Version 3.1.1
Last updated: 26/09/23

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
09/06/20 1.9    Chaneged magnetic symmetry calculation, now correctly generates mag symmetry from operations
10/06/20 1.9.1  Added time symmetry to Symmetry class
16/06/20 1.9.2  Made change in simpulate_powder to remove linspace error
29/06/20 1.9.3  Removed import of scipy.convolve2d due to import errrors, new method slower but more accurate
02/09/20 1.9.4  Added string methods to Crystal classes
26/11/20 1.9.5  Various improvements and corrections
04/01/21 1.9.6  Added Scattering.structure_factor function
26/01/21 1.9.7  Added xray dispersion correction functions, plus x-ray interactions calculations
15/02/21 1.9.8  Added plotting options for out-of-plane recirprocal lattice lines
10/06/21 1.9.9  Corrected error in calculation of DebyeWaller factor (APDs), added x-ray factors by Waasmaier and Kirfel
09/07/21 1.9.9  Added new scattering factors to scattering factor settings
20/08/21 2.0.0  Added functions_scattering module with updated, more consistent scattering formula
27/09/21 2.1.0  Added classes_orientation and diffractometer orientation functions
15/11/21 2.1.1  Updated Plot.plot_crystal, added additional orientation functions
07/02/22 2.1.2  Corrected error in classes_scattering.powder() of wrong tth values. Thanks Mirko!
14/03/22 2.2.0  Scatter.powder() updated for new inputs and outputs for pVoight and custom peak shapes. Thanks yevgenyr!
23/07/22 2.2.1  Fixed error in MultiCrystal.Scatter
06/01/23 2.2.2  Removed redundent references to np.float
14/01/23 2.2.3  Corrected background error in xtl.Scatter.powder
08/05/23 2.3.0  Merged pull request for non-integer hkl option on SF and electron form factors. Thanks Prestipino!
25/06/23 3.0.0  Added new GUI elements including new Scattering UI and diffractomter simulator, plus other updates
11/07/23 3.0.1  Some fixes for plotting and additions to diffractometer orientation. Thanks asteppke!
20/07/23 3.1.0  Refactored FDMNES wrapper with new methods and new defaults. Thanks YvesJoly!
26/09/23 3.1.1  Minor changes and improvments. Added hkl1, hkl2 = xtl.scatter.orientation_reflections()

Acknoledgements:
    2018        Thanks to Hepesu for help with Python3 support and ideas about breaking up calculations
    Dec 2019    Thanks to Gareth Nisbet for allowing me to inlude his multiple scattering siumulation
    April 2020  Thanks to ChunHai Wang for helpful suggestions in readcif!
    May 2020    Thanks to AndreEbel for helpful suggestions on citations
    Dec 2020    Thanks to Chris Drozdowski for suggestions about reflection families
    Jan 2021    Thanks to aslarsen for suggestions about outputting the structure factor
    April 2021  Thanks to Trygve Ræder for suggestions about x-ray scattering factors
    Feb 2022    Thanks to Mirko for pointing out the error in two-theta values in Scatter.powder
    March 2022  Thanks to yevgenyr for suggesting new peak profiles in Scatter.powder
    Jan 2023    Thanks to Anuradha Vibhakar for pointing out the error in f0 + if'-if''
    Jan 2023    Thanks to Andreas Rosnes for testing the installation in jupyterlab
    May 2023    Thanks to Carmelo Prestipino for adding electron scattering factors
    June 2023   Thanks to Sergio I. Rincon for pointing out the rounding error in Scatter.powder
    July 2023   Thanks to asteppke for suggested update to Arrow3D for matplotlib V>3.4
    July 2023   Thanks to Yves Joly for helpful suggestions on FDMNES wrapper

-----------------------------------------------------------------------------
   Copyright 2023 Diamond Light Source Ltd.

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
from .classes_fdmnes import fdmnes_checker, Fdmnes, FdmnesAnalysis
from .functions_crystallography import readcif


__version__ = '3.1.1'
__date__ = '26/09/23'


# Build
structure_list = Structures()


def version_info():
    return 'Dans_Diffraction version %s (%s)' % (__version__, __date__)


def module_info():
    import sys
    out = 'Python version %s' % sys.version
    out += '\n%s' % version_info()
    # Modules
    out += '\n     numpy version: %s' % fg.np.__version__
    try:
        import matplotlib
        out += '\nmatplotlib version: %s' % matplotlib.__version__
    except ImportError:
        out += '\nmatplotlib version: None'
    try:
        import tkinter
        out += '\n   tkinter version: %s' % tkinter.TkVersion
    except ImportError:
        out += '\n   tkinter version: None'
    try:
        import scipy
        out += '\n     scipy version: %s' % scipy.__version__
    except ImportError:
        out += '\n     scipy version: None'
    return out


def doc_str():
    return __doc__


# tkGUI Activation
def start_gui(xtl=None):
    """Start GUI window (requires tkinter)"""
    try:
        from .tkgui import CrystalGui
        CrystalGui(xtl)
    except ImportError:
        print('GUI functionality not available, you need to install tkinter.')


# FDMNES Activation
def activate_fdmnes(initial_dir=None, fdmnes_filename='fdmnes_win64.exe'):
    """
    Call to activate FDMNES functionality
    :param fdmnes_filename: name of the executable to search for
    :param initial_dir: None or str, if directory, look here for file
    :return: None
    """
    fdmnes_checker(activate=True, fdmnes_filename=fdmnes_filename, initial_dir=initial_dir)
