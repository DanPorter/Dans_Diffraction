"""
CIF Directory
Various standard structures to load into Dans_Diffraction.

Usage:
    from classes_structures import Structures
    structure_list = Structures() # builds database of all cif files in Structures Directory
    xtl = structure_list.Silicon.build() # builds Crystal class of selected structure

By Dan Porter, PhD
Diamond
2018

Version 1.0
Last updated: 02/03/18

Version History:
02/03/18 1.0    Program created

@author: DGPorter
"""

import sys,os,glob
import numpy as np
from . import Crystal

__version__ = '1.0'


def cif_list():
    "Returns a list of cif files in the current directory"
    current_dir = os.path.dirname(__file__)
    structure_dir = os.path.join(current_dir,'Structures','*cif')
    cif_files = glob.glob(structure_dir)
    cif_files = np.sort(cif_files)
    return cif_files


class Structures:
    """
    Provides a database of cif files
        S = Structures() # builds database
        xtl = S.Diamond.build()
        
        Use S.list to see the available structures
    """
    def __init__(self):
        # Read cif files in current directory
        cif_files = cif_list()
        
        self.list=[]
        for filename in cif_files:
            # Generate structure name
            (dirName,filetitle)=os.path.split(filename)
            (fname, Ext)=os.path.splitext(filetitle)
            
            # Replace illegal characters
            for chars in '\'!$,.()[]{}':
                fname = fname.replace(chars,'')
            self.list += [fname]
            
            setattr(self,fname,Build_Crystal(filename))
    
    def info(self):
        "Print Available Structures"
        for s in self.list:
            print(s)


class Build_Crystal:
    """
    Storage Class for filename and build command
    Builds a Crystal class.
    """
    def __init__(self,filename):
        self.filename = filename
    
    def build(self):
        return Crystal(self.filename)

