"""
Dans_Diffraction tkGUI.py
  Creates a graphical user interface for Dans_Diffraction software.
  Based on tkinter.

  Requires:
      numpy, matplotlib, tkinter

By Dan Porter, PhD
Diamond
2019

Version 2.4.0
Last updated: 16/05/23

Version History:
10/11/17 0.1    Program created
23/02/19 1.0    Finished the basic program and gave it colours
09/03/19 1.1    Added properties, superstructure, other improvements
13/07/19 1.2    Added FDMNES windows
13/12/19 2.0    Changed to internal tkgui package
19/04/20 2.1    Added multi-crystal, write-cif buttons, changed layout of scattering gui
27/04/20 2.2    Added SelectionBox to basic_widgets and spacegroup entry to SymmetryGui
15/10/20 2.2.1  Slight correction to SymmetryGui - no longer adds 'm' to labels
21/01/21 2.2.2  Added 'xray dispersion' scattering option
26/01/21 2.3.0  Refactored properites into new file, added x-ray interactions GUI
16/05/23 2.4.0  Added periodic_table.py and added menu items to CrystalGUI. Changed default fonts and colours

@author: DGPorter
"""

from .crystal import CrystalGui

__version__ = '2.4.0'
__date__ = '16/05/23'
