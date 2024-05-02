# -*- coding: utf-8 -*-
"""
FDMNES class "classes_fdmnes.py"
 functions for generating fdmnes files

By Dan Porter, PhD
Diamond
2018

Version 2.0
Last updated: 20/07/23

Version History:
17/04/18 0.9    Program created
04/06/18 1.0    Program completed and tested
11/06/18 1.1    Analysis updated to find density with _sd1 and _sd0
13/07/19 1.2    Small updates for gui functionality
07/08/19 1.3    Changed reflist in FdmnesAnalysis to list of hkl
16/08/19 1.4    Added BavFile to read parts of .bav file
18/03/20 1.5    Corrected density file for new headers
22/04/20 1.6    Added FdmnesCompare and __add__ method for FdmnesAnalysis
20/07/23 2.0    Refactored program and updated, added new methods and put indata writer in classes_properties

@author: DGPorter
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

from . import functions_general as fg
from . import functions_crystallography as fc
from . import Crystal


__version__ = '2.0'


class Fdmnes:
    """
    FDMNES class: Create input files for the FDMNES software and run the program inside a python console

    FDMNES (Finite Difference Method for Near Edge Structure) is an abinitio DFT program for
    simulating x-ray absorption spectra and resonant diffraction.
    FDMNES is developed by Yves Joly at CNRS and can be freely downloaded from here:
      www.fdmnes.neel.cnrs.fr

    In case of publication, please include the following citation:
      O. Bunau and Y. Joly "Self-consistent aspects of x-ray absorption calculations"
      J. Phys.: Condens. Matter 21, 345501 (2009)

    This class creates a simple wrapper to automatically generate input files from Dans_Diffraction
    Crystal structures. The output files from the calculation can also be automatically plotted.

    E.G.
    xtl = Crystal('Co.cif')
    fdm = Fdmnes(xtl)
    fdm.setup(folder_name='New_Calculation',
              comment='Test',
              absorber='Co',
              edge='K'
              azi_ref=[0,0,1],
              hkl_reflections=[[1,0,0],[0,1,0],[1,1,0]]
    fdm.create_files()
    fdm.write_fdmfile()
    fdm.run_fdmnes()
    ###Wait for program completion###
    analysis = fdm.analyse()
    print(analysis)
    analysis.xanes.plot()
    analysis.density.plot()
    for ref in analysis:
        ref.plot3D()
    """

    def __init__(self, xtl=Crystal()):
        self.xtl = xtl

        # Options
        if fdmnes_checker():
            self.exe_path = find_fdmnes()
        else:
            self.exe_path = ''
        self.output_name = 'out'
        self.folder_name = fg.saveable(self.xtl.name)
        self.input_name = 'FDMNES_%s.txt' % fg.saveable(self.xtl.name)
        self.output_path = self.generate_output_path()
        self.comment = ''
        self.range = '-19. 1. -5 0.1 -2. 0.05 5 0.1 10. 0.5 25 1 31. '
        self.radius = 4.0
        self.edge = 'K'
        self.absorber = self.xtl.Properties.resonant_element()
        self.green = True
        self.scf = False
        self.quadrupole = False
        self.magnetism = True if self.xtl.Structure.ismagnetic() else False
        self.spinorbit = False
        self.azi_ref = [1, 0, 0]
        self.correct_azi = True
        self.hkl_reflections = [[1, 0, 0]]

    def __repr__(self):
        return "Fdmnes('%s')" % self.xtl.name

    def __str__(self):
        return self.info()

    def setup(self, exe_path=None, output_path=None, output_name=None, folder_name=None, input_name=None,
              comment=None, energy_range=None, radius=None, edge=None, absorber=None, green=None, scf=None,
              quadrupole=None, azi_ref=None, correct_azi=None, hkl_reflections=None):
        r"""
        Set FDMNES Parameters
        :param exe_path: Location of FDMNES executable, e.g. 'c:\FDMNES\fdmnes_win64.exe'
        :param output_path: Specify the output path
        :param folder_name: Specify output folder name (replaces output_path)
        :param output_name: Name of FDMNES output files
        :param input_name: Name of FDMNES input file
        :param comment: A comment written in the input file
        :param energy_range: str energy range in eV relative to Fermi energy
        :param radius: calculation radius
        :param edge: absorptin edge, 'K', 'L3', 'L2'
        :param absorber: absorbing element, 'Co'
        :param green: Green's function (muffin-tin potential)
        :param scf: True/False, Self consistent solution
        :param quadrupole: False/True, E1E2 terms allowed
        :param azi_ref: azimuthal reference, [1,0,0]
        :param correct_azi: if True, correct azimuthal reference for real cell (use in hexagonal systems)
        :param hkl_reflections: list of hkl reflections [[1,0,0],[0,1,0]]
        :return: None
        """
        if exe_path is not None:
            self.exe_path = exe_path
            self.output_path = self.generate_output_path()

        if output_path is not None:
            self.output_path = output_path

        if output_name is not None:
            self.output_name = output_name

        if folder_name is not None:
            self.folder_name = folder_name
            self.output_path = self.generate_output_path()

        if input_name is not None:
            self.input_name = input_name

        if comment is not None:
            self.comment = comment

        if energy_range is not None:
            self.range = energy_range

        if radius is not None:
            self.radius = radius

        if edge is not None:
            self.edge = edge

        if absorber is not None:
            self.absorber = absorber

        if green is not None:
            self.green = green

        if scf is not None:
            self.scf = scf

        if quadrupole is not None:
            self.quadrupole = quadrupole

        if azi_ref is not None:
            self.azi_ref = azi_ref

        if correct_azi is not None:
            self.correct_azi = correct_azi

        if hkl_reflections is not None:
            self.hkl_reflections = np.asarray(hkl_reflections).reshape(-1, 3)

    def info(self):
        """
        Print setup info
        :return: str
        """
        s = 'FDMNES Options\n'
        s += 'Crystal: %s\n' % self.xtl.name
        s += 'exe_path : %s\n' % self.exe_path
        s += 'output_path : %s\n' % self.output_path
        s += 'output_name : %s\n' % self.output_name
        s += 'input_name : %s\n' % self.input_name
        s += 'comment : %s\n' % self.comment
        s += 'range : %s\n' % self.range
        s += 'radius : %s\n' % self.radius
        s += 'absorber : %s\n' % self.absorber
        s += 'edge : %s\n' % self.edge
        s += 'green : %s\n' % self.green
        s += 'scf : %s\n' % self.scf
        s += 'quadrupole : %s\n' % self.quadrupole
        if self.correct_azi:
            s += 'azi_ref : %s\n' % self.azimuthal_reference(self.azi_ref)
        else:
            s += 'azi_ref : %s\n' % self.azi_ref
        s += 'hkl_reflections:\n'
        for ref in self.hkl_reflections:
            s += '  (%1.0f,%1.0f,%1.0f)\n' % (ref[0], ref[1], ref[2])
        return s

    def find_fdmnes_exe(self, initial_dir=None, fdmnes_filename='fdmnes_win64.exe', reset=False):
        """Find fdmnes executable"""
        self.exe_path = find_fdmnes(reset=reset, fdmnes_filename=fdmnes_filename, initial_dir=initial_dir)

    def azimuthal_reference(self, hkl=(1, 0, 0)):
        """
        Generate the azimuthal reference
        :param hkl: (1*3) array [h,k,l]
        :return: None
        """

        UV = self.xtl.Cell.UV()
        UVs = self.xtl.Cell.UVstar()

        sl_ar = np.dot(np.dot(hkl, UVs), np.linalg.inv(UVs))  # Q*/UV*
        fdm_ar = np.dot(np.dot(hkl, UVs), np.linalg.inv(UV))  # Q*/UV
        fdm_ar = fdm_ar / np.sqrt(np.sum(fdm_ar ** 2))  # normalise length to 1
        return fdm_ar

    def generate_parameters_string(self):
        """
        Create the string of parameters and comments for the input file
        :return: str
        """
        outstr = self.xtl.Properties.fdmnes_runfile(
            output_path=os.path.join(self.output_path, self.output_name),
            comment=self.comment,
            energy_range=self.range,
            radius=self.radius,
            edge=self.edge,
            absorber=self.absorber,
            green=self.green,
            scf=self.scf,
            quadrupole=self.quadrupole,
            magnetism=self.magnetism,
            spinorbit=self.spinorbit,
            azi_ref=self.azi_ref,
            correct_azi=self.correct_azi,
            hkl_reflections=self.hkl_reflections
        )
        return outstr

    def generate_output_path(self, folder_name=None, overwrite=False):
        r"""
        Creates an automatic output path in the FDMNES/Sim directory
         If overwrite is False and the directory already exists, a number will be appended to the name
        :param folder_name: str or None, if None xtl.name will be used
        :param overwrite: True/False
        :return: str directory E.G. 'c:\FDMNES\Sim\Fe2O3'
        """

        if folder_name is None:
            folder_name = self.folder_name

        newpath = os.path.join(os.path.dirname(self.exe_path), 'Sim', folder_name)

        if overwrite:
            return newpath

        n = 0
        while os.path.isdir(newpath):
            n += 1
            newpath = os.path.join(os.path.dirname(self.exe_path), 'Sim', folder_name + '_%d' % n)
        return newpath

    def generate_input_path(self):
        r"""
        Returns the input file pathname
         E.G. 'c:\FDMNES\Sim\Fe2O3\FDMNES_Fe2O3.txt'
        :return: filepath
        """
        return os.path.join(self.output_path, self.input_name)

    def create_directory(self):
        """
        Create a directory in the FDMNES/Sim folder
        :return: None
        """
        if not self.exe_path:
            raise AttributeError('FDMNES executable path has not been specified')
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
            print("Created folder: '{}'".format(self.output_path))

    def write_runfile(self, param_string=None):
        """
        Write FDMNES input data to a file
        :param param_string: str text to add to indata file (None will generate)
        :return: None
        """

        if param_string is None:
            param_string = self.generate_parameters_string()

        # Create cell file
        fname = self.generate_input_path()
        with open(fname, 'w') as fdfile:
            fdfile.write(param_string)
        print("FDMNES file written to '{}'".format(fname))

    def create_files(self, param_string=None):
        """
        Create FDMNES calculation directory and files
         - creates new directory
         - writes the input file to that directory
        :param param_string: str text to add to indata file (None will generate)
        :return: None
        """
        self.create_directory()
        self.write_runfile(param_string)
        print('Ready for calculation!')

    def write_fdmfile(self, file_list=None):
        """
        Create fdmfile with list of calculations
        :param file_list: list of parameter file names for fdmfile, or None to only calculate current parameters
        :return: None
        """

        if file_list is None:
            file_list = [os.path.join(self.output_path, self.input_name)]

        fdmstr = ''
        fdmstr += '! FDMNES Calculations\n'
        fdmstr += '! List of calculations here\n\n'

        fdmstr += ' %1.0f\n' % len(file_list)
        fdmstr += '\n'.join(file_list)

        # Create/ overwrite file
        if not self.exe_path:
            raise AttributeError('FDMNES executable path has not been specified')
        fdmfile = os.path.join(os.path.dirname(self.exe_path), 'fdmfile.txt')
        with open(fdmfile, 'w') as out:
            out.write(fdmstr)
        print('fdmfile created.')

    def run_fdmnes(self):
        """
        Run the fdmnes code, waits until the program completes
        Remember to use self.create_files and self.write_fdmfile first!
        :return: subprocess.call output
        """
        if not self.exe_path:
            raise AttributeError('FDMNES executable path has not been specified')

        import subprocess
        wd = os.getcwd()
        path, file = os.path.split(self.exe_path)
        os.chdir(path)
        print('Starting FDMNES...')
        output = subprocess.call(file, shell=True)
        os.chdir(wd)
        print('FDMNES Finished!')
        return output

    def analyse(self, folder_name=None):
        """
        Analyse the completed calculation
        :param folder_name: str name or directory of calculation (None to use current calculation)
        :return: FdmnesAnalysis Object
        """

        analysis_path = self.output_path

        if folder_name is not None:
            if os.path.isdir(folder_name):
                analysis_path = folder_name
            else:
                analysis_path = self.generate_output_path(folder_name, overwrite=True)
        calcname = '%s %s %s' % (self.xtl.name, self.absorber, self.edge)
        return FdmnesAnalysis(analysis_path, self.output_name, calc_name=calcname)


class FdmnesAnalysis:
    """
    Create fdmnes object from *_scan_conv.txt file

    Usage:
        fdm = FdmnesAnalysis(output_path, output_name)

        fdm contains all calculated reflections and xanes spectra, as well as spherical tensors
        Automated plotting of azimuths, spectra and density of states.

        fdm.xanes.plot()
        fdm.density.plot()
        fdm.I100sp.plot3D()
        fdm.sph_I100sp.plot()
    """

    energy = np.array([0])
    angle = np.array([0])
    reflections = {}
    reflist = []
    refkeys = []
    sphkeys = []

    def __init__(self, output_path, output_name='out', calc_name=None):
        r"""
        Loads data from an FDMNES calculation, allowing plotting and data cuts
         E.G.
         fdm = FdmnesAnalysis('c:\FDMNES\Fe2O3','out')
         fdm.xanes.plot()
         plt.show()

        :param output_path: directory of the calculation to be loaded
        :param output_name: base output name for the calculation (default='out')
        """
        output_path = output_path.replace('\\', '/')  # convert windows directories
        if output_path.endswith('.txt'):
            output_path, filename = os.path.split(output_path)
            filename = filename.replace('_', '.')
            output_name = filename.split('.')[0]
        elif not os.path.isdir(output_path):
            output_path = sim_folder(output_path)
        if calc_name is None:
            calc_name = output_path.split('/')[-1]  # calculation name
        bav_name = os.path.join(output_path, output_name + '_bav.txt')
        scan_conv_name = os.path.join(output_path, output_name + '_scan_conv.txt')
        convname = os.path.join(output_path, output_name + '_conv.txt')
        densityname = os.path.join(output_path, output_name + '_sd%d.txt')
        spherename = os.path.join(output_path, output_name + '_sph_signal_rxs%d.txt')

        self.output_name = output_name
        self.output_path = output_path
        self.calc_name = calc_name

        # Read Bav file (_bav.txt)
        with open(bav_name) as bavfile:
            self.bavfile = BavFile(bavfile.read())
            #self.output_text = bavfile.read()

        # Read XANES file (_conv.txt)
        if os.path.isfile(convname):
            enxanes, Ixanes = read_conv(convname)
            self.xanes = Xanes(enxanes, Ixanes, calc_name)
            self.energy = enxanes

        # Read density of states file
        # sometimes this is _sd0.txt and sometimes _sd1.txt
        if os.path.isfile(densityname % 0):
            self.density = Density(densityname % 0)
        elif os.path.isfile(densityname % 1):
            self.density = Density(densityname % 1)
        else:
            self.density = None

        if os.path.isfile(scan_conv_name):
            # Read reflection files (_scan_conv.txt)
            energy, angle, intensity = read_scan_conv(scan_conv_name)
            self.energy = energy
            self.angle = angle
            self.reflections = intensity
            #self.reflist = intensity.keys()

            # Generate list of reflections (sigma-pi refs only)
            self.reflist = self.bavfile.reflections()
            self.refkeys = []
            self.sphkeys = []

            # Assign reflection files
            for n, ref in enumerate(intensity.keys()):
                refname = ref.replace('(', '').replace(')', '').replace('-', '_')
                refobj = Reflection(self.energy, self.angle, self.reflections[ref], ref, calc_name)
                setattr(self, refname, refobj)
                self.refkeys += [refname]

                # Read spherical contribution files
                sphrefname = 'sph_' + refname
                if os.path.isfile(spherename % (n + 1)):
                    data = read_spherical(spherename % (n + 1))
                    sphobj = Spherical(data, ref, calc_name)
                    setattr(self, sphrefname, sphobj)
                    self.sphkeys += [sphrefname]
        else:
            self.reflist = []
            self.refkeys = []
            self.sphkeys = []

    def info(self):
        """
        Returns header of calculation output fipe (*_bav.txt)
        :return: str
        """
        out = 'FDMNES Analysis: %s' % os.path.join(self.output_path, self.output_name+'.txt')
        out += self.bavfile.header()
        return out

    def __repr__(self):
        return "FdmnesAnalysis('%s', '%s', calc_name='%s')" % (self.output_path, self.output_name, self.calc_name)

    def __str__(self):
        name = '%s/%s' % (self.calc_name, self.output_name)
        filename = os.path.join(self.output_path, self.output_name+'.txt')
        cycles = self.bavfile.cycles()
        edge = self.bavfile.edge()
        radius = self.bavfile.radius()
        refs = ', '.join(self.refkeys)
        sphs = ', '.join(self.sphkeys)
        out = 'FDMNES Analysis: %30s\nEdge: %20s\nRadius: %10s\nCycles: %10s\n' % (name, edge, radius, cycles)
        out += 'RXS Refs: %s\nSPH Refs: %s\n' % (refs, sphs)
        return out

    def __add__(self, other):
        return FdmnesCompare([self, other])

    def __getitem__(self, item):
        """Return Reflection"""
        if item in self.refkeys or item in self.sphkeys:
            return getattr(self, item)
        return getattr(self, self.refkeys[item])


class FdmnesCompare:
    """
    Compare FDMNES Calculations
    Method of joining sevearl FdmnesAnalysis objects to plot spectra on the same plot for comparison.
    Usage:
        fd1 = FdmnesAnalysis('calc_dir/calc1')
        fd2 = FdmnesAnalysis('calc_dir/calc2')
        fd3 = FdmnesAnalysis('calc_dir/calc3')
        fdms = FdmnesCompare([fd1, fd2, fd3])
        *or*
        fdms = fd1 + fd2 + fd3
        *or*
        fdms = load_fdmnes_files('calc_dir') # loads all calculations ind calc_dir

        fdms.plot_xanes()
        fdms.plot_reflection_spectra('I003sp', 0) # specify azimuthal angle in deg
        fdms.plot_reflection_azimuth('I003sp', 2838) # specify energy in eV
    """

    def __init__(self, fdm_objects):
        self.fdm_objects = fdm_objects
        self._gen_refkeys()

    def load(self, bav_files=()):
        """Load a series of calculation files"""
        bav_files = np.asarray(bav_files).reshape(-1)
        for file in bav_files:
            self.fdm_objects += [FdmnesAnalysis(file)]
            print(self.fdm_objects[-1])
        self._gen_refkeys()

    def _gen_refkeys(self):
        """Generate unique reflection keys"""
        all_refs = []
        for fdm in self.fdm_objects:
            all_refs += fdm.refkeys
        # unique reflections
        self.refkeys = list(np.unique(all_refs))

    def info(self):
        out = ''
        for fdm in self.fdm_objects:
            out += '%s\n' % fdm
        return out

    def __repr__(self):
        return self.info()

    def __add__(self, other):
        if type(other) is FdmnesCompare:
            return FdmnesCompare(self.fdm_objects+other.fdm_objects)
        return FdmnesCompare(self.fdm_objects + [other])

    def plot_xanes(self, normalise=False, new_figure=True):
        """
        Plot XANES spectra for each calculation
        :param normalise: normalise each spectra by max point
        :param new_figure: create a new figure if True, otherwise plot on the current axis
        :return: None
        """

        if new_figure:
            plt.figure(figsize=[12, 10], dpi=80)
        for fdm in self.fdm_objects:
            inten = fdm.xanes.intensity
            if normalise:
                inten = inten/ inten.max()
            plt.plot(fdm.xanes.energy, inten, '-', lw=3, label=fdm.calc_name)
        plt.legend(loc=0, frameon=False, fontsize=20)
        plt.xlabel('Energy [eV]', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Intensity [arb. units]', fontsize=28, fontname='Times New Roman')
        plt.xticks(fontsize=25, fontname='Times New Roman')
        plt.yticks(fontsize=25, fontname='Times New Roman')

    def plot_reflection_spectra(self, refkey, psi=0, normalise=False, new_figure=True):
        """
        Plot all energy spectra for a reflection
        :param refkey: str, reflection key as in name_conv.txt
        :param psi: float, azimuthal angle
        :param normalise: normalise each spectra by max point
        :param new_figure: create a new figure if True, otherwise plot on the current axis
        :return: None
        """

        refkey = refkey.replace('(', '').replace(')', '').replace('-', '_')

        if new_figure:
            plt.figure(figsize=[12, 10], dpi=80)
        for fdm in self.fdm_objects:
            try:
                ref = getattr(fdm, refkey)
            except AttributeError:
                print('No %s in %s' % (refkey, fdm))
                continue
            en = ref.energy
            inten = ref.eng_cut(psi)
            if normalise:
                inten = inten/inten.max()
            plt.plot(en, inten, '-', lw=3, label=fdm.calc_name)
        plt.title(r'%s $\Psi$ = %s Deg' % (refkey, psi), fontsize=32, fontname='Times New Roman')
        plt.legend(loc=0, frameon=False, fontsize=20)
        plt.xlabel('Energy [eV]', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Intensity [arb. units]', fontsize=28, fontname='Times New Roman')
        plt.xticks(fontsize=25, fontname='Times New Roman')
        plt.yticks(fontsize=25, fontname='Times New Roman')

    def plot_reflection_azimuth(self, refkey, energy=0, normalise=False, new_figure=True):
        """
        Plot all energy spectra for a reflection
        :param refkey: str, reflection key as in name_conv.txt
        :param energy: float, energy [eV]
        :param normalise: normalise each spectra by max point
        :param new_figure: create a new figure if True, otherwise plot on the current axis
        :return: None
        """

        refkey = refkey.replace('(', '').replace(')', '').replace('-', '_')

        if new_figure:
            plt.figure(figsize=[12, 10], dpi=80)
        for fdm in self.fdm_objects:
            try:
                ref = getattr(fdm, refkey)
            except AttributeError:
                print('No %s in %s' % (refkey, fdm))
                continue
            psi = ref.angle
            inten = ref.azi_cut(energy)
            if normalise:
                inten = inten/inten.max()
            plt.plot(psi, inten, '-', lw=3, label=fdm.calc_name)
        plt.title('%s E = %s eV' % (refkey, energy), fontsize=32, fontname='Times New Roman')
        plt.legend(loc=0, frameon=False, fontsize=20)
        plt.xlabel(r'$\Psi$ [Deg]', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Intensity [arb. units]', fontsize=28, fontname='Times New Roman')
        plt.xticks(fontsize=25, fontname='Times New Roman')
        plt.yticks(fontsize=25, fontname='Times New Roman')

    def plot_all_spectra(self, psi=0, normalise=False):
        """
        Plot all available spectra across calculations
        :param psi: azimuthal angle [Deg]
        :param normalise: normalise each spectra by max point
        :return:
        """
        for refkey in self.refkeys:
            self.plot_reflection_spectra(refkey, psi, normalise)

    def plot_all_azimuth(self, energy=0, normalise=False, new_figure=True):
        """
        Plot all available azimuths across calculations
        :param energy: point at which to cut energy
        :param normalise: normalise each spectra by max point
        :return:
        """
        for refkey in self.refkeys:
            self.plot_reflection_azimuth(refkey, energy, normalise)


class BavFile:
    def __init__(self, text):
        self.text = text
        self.potrmt = potrmt(text)

    def header(self):
        """
        Returns the top part of the file
        :return:
        """
        return self.text[:self.text.find('Symsite')]

    def reflections(self):
        """
        Return list of reflections specified in the header
        :return: list
        """

        rgex = [r'-?\d+[\s-]+\d+[\s-]+\d+\s+sigma pi',
                r'-?\d+[\s-]+\d+[\s-]+\d+\s+sigma sigma',
                r'-?\d+[\s-]+\d+[\s-]+\d+\s+pi pi',
                r'-?\d+[\s-]+\d+[\s-]+\d+\s+pi sigma']
        rgex = ' | '.join(rgex)

        fnd  = re.findall(rgex, self.header())
        reflist = []
        for val in fnd:
            reflist += [[int(i) for i in re.findall(r'-?\d+', val)]]
        return reflist

    def cycles(self):
        """
        Returns the number of cycles completed
        :return:
        """
        return self.text.count('Cycle')

    def fdmnes_version(self):
        """
        Returns the version date of FDMNES used in the calculation
        :return: str
        """
        return self.text.split('\n')[0].strip()

    def time(self):
        """
        Returns the date and time the calculation was run on
        :return: str
        """
        day, month, year = re.findall('Date =.+?\n', self.text)[0][6:].split()
        time = re.findall('Time =.+?\n', self.text)[0][6:].strip()
        time = time.replace('h',':').replace('mn',':').replace(' s','')
        return '%s/%s/%s  %s' % (day, month, year, time)

    def edge(self):
        """
        Returns the resoant edge used in calculation
        :return: str
        """
        fnd = re.findall('Threshold: .+\n', self.header())
        if len(fnd) < 1:
            return ''
        return fnd[0].strip('Threshold:').strip()

    def radius(self):
        """
        Returns the radius used for the calculation
        :return: float
        """
        fnd = re.findall('Radius = .+\n', self.header())
        if len(fnd) < 1:
            return ''
        return float(fnd[0].strip('Radius =').strip())

    def charge_str(self):
        """
        returns final cycle ion charge
        :return: str
        """
        z = self.potrmt['Z']
        ch = self.potrmt['ch_ion']
        outstr = '  Z : ch_ion\n'
        outstr+= '\n'.join(['%3d : %s' % (z[n], ch[n]) for n in range(len(z))])
        return outstr

    def potrmt_str(self):
        """
        Return final cycle "Potrmt" string
        :return: str
        """
        keys = list(self.potrmt.keys())
        if len(keys) == 0:
            return '"Potrmt" charge string not found'
        outstr = ' '.join(['%8s' % key for key in keys]) + '\n'
        for n in range(len(self.potrmt[keys[0]])):
            outstr += ' '.join(['%8s' % self.potrmt[key][n] for key in keys]) + '\n'
        return outstr

    def cycle_energy(self):
        """Return array of total energy per atom for each cycle"""
        # Find Delta_ener nad Total energy
        fnd_delta = re.findall('Delta_energ = .+? eV', self.text)
        fnd_energ = re.findall('Total: .+?\n|Total =.+?\n', self.text)

        delta = np.array([fn.split()[-2] for fn in fnd_delta], dtype=float)
        energ = np.array([fn.replace('Total =', '').replace('Total:', '').split()[0] for fn in fnd_energ], dtype=float)
        return energ, delta

    def plot_cycle_energy(self):
        """Plot progress of energy during SCF cycles"""

        energ, delta = self.cycle_energy()

        plt.figure(figsize=[12, 10], dpi=80)
        p1, = plt.plot(energ, 'r-', lw=2, label='Total Energy/Atom')

        plt.xlabel('Cycle', fontsize=28)
        plt.ylabel('Energy [eV]', fontsize=28)

        plt.gca().twinx()
        p2, = plt.plot(range(1, len(delta)+1), delta, 'bo:', lw=2, label='Delta Energy')

        plt.legend([p1, p2], ['Total Energy/Atom', 'Delta Energy'], loc='upper center', frameon=False, fontsize=24)


class Reflection:
    def __init__(self, energy, angle, intensity, refname, calc_name):
        """
        Container for FDMNES RXS calculations for each reflection
        :param energy: array of energy values [1xm]
        :param angle: array of angle values [1xn]
        :param intensity: array of intensity values [mxn]
        :param refname: reflection name, for plot title
        :param calc_name: calculation name, for plot title
        """
        self.energy = energy
        self.angle = angle
        self.intensity = intensity
        self.refname = refname
        self.calc_name = calc_name

    def __repr__(self):
        return "Reflection('%s', '%s')" % (self.refname, self.calc_name)

    def azi_cut(self, cutenergy=None):
        """
        Returns the array of intensitiy values at a particular energy
        :param cutenergy: energy of cut (eV)
        :return: array of intensities
        """
        cutintensity = azi_cut(self.energy, self.intensity, cutenergy)
        return cutintensity

    def eng_cut(self, cutangle=None):
        """
        Returns the array of intensitiy values at a particular angle
        :param cutangle: angle of cut (deg)
        :return: array of intensities
        """
        cutintensity = eng_cut(self.angle, self.intensity, cutangle)
        return cutintensity

    def plot3D(self):
        """
        Generate a 3D figure of energy vs angle vs intensity
        :return: None
        """
        # 3D figure
        fig = plt.figure(figsize=[12, 10], dpi=80)
        ax = fig.add_subplot(111, projection='3d')

        XX, YY = np.meshgrid(self.angle, self.energy)
        cm = plt.get_cmap('coolwarm')
        ax.plot_surface(XX, YY, self.intensity, rstride=3, cstride=3, cmap=cm,
                        linewidth=0, antialiased=False)

        # Axis labels
        ax.set_xlabel('Angle (DEG)', fontsize=18)
        ax.set_ylabel('Energy (eV)', fontsize=18)
        ax.set_zlabel('Intensity', fontsize=18)
        plt.suptitle('{}\n{}'.format(self.calc_name, self.refname), fontsize=21, fontweight='bold')

    def plot_azi(self, cutenergy='max'):
        """
        Generate a plot of azimuthal dependence at a particular energy
        :param cutenergy: 'max' to use the maximum height energy, or an energy in eV
        :return: None
        """
        cutintensity = azi_cut(self.energy, self.intensity, cutenergy)

        plt.figure(figsize=[12, 10], dpi=80)
        plt.plot(self.angle, cutintensity, lw=3)
        plt.xlabel('Angle (DEG)', fontsize=18)
        plt.ylabel('Intensity', fontsize=18)
        plt.title('{}\n{} {} eV'.format(self.calc_name, self.refname, cutenergy), fontsize=21, fontweight='bold')

    def plot_eng(self, cutangle='max'):
        """
        Generate a plot of energy dependence at a particular azimuth
        :param cutangle: 'max' to use the maximum intensity height angle, or an angle in deg
        :return: None
        """
        cutintensity = eng_cut(self.angle, self.intensity, cutangle)

        plt.figure(figsize=[12, 10], dpi=80)
        plt.plot(self.energy, cutintensity, lw=3)
        plt.xlabel('Energy (eV)', fontsize=18)
        plt.ylabel('Intensity', fontsize=18)
        plt.title('{}\n{} {} Deg'.format(self.calc_name, self.refname, cutangle), fontsize=21, fontweight='bold')


class Xanes:
    def __init__(self, energy, intensity, calc_name):
        """
        Container for XANES spectra from FDMNES
        :param energy: array of energy values
        :param intensity: array of intensity values
        :param calc_name: calculation name for plot title
        """
        self.energy = energy
        self.intensity = intensity
        self.calc_name = calc_name

    def __repr__(self):
        return "Xanes('%s')" % self.calc_name

    def plot(self):
        """
        Plot the Xanes spectra
        :return: None
        """
        plt.figure(figsize=[12, 10], dpi=80)
        plt.plot(self.energy, self.intensity, lw=3)
        plt.title(self.calc_name, fontsize=26, fontweight='bold', fontname='Times New Roman')
        plt.xlabel('Energy [eV]', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Intensity [arb. units]', fontsize=28, fontname='Times New Roman')
        plt.xticks(fontsize=25, fontname='Times New Roman')
        plt.yticks(fontsize=25, fontname='Times New Roman')


class Density:
    def __init__(self, file):
        """
        Load Density of states file from FDMNES calculations '..._sd0.txt'
        :param file: filename
        """

        dirname, filetitle = os.path.split(file)  # calculation directory
        self.calc_name = dirname.split('/')[-1]  # calculation name
        self.data = read_density(file)

        self.energy = self.data['energy']
        self.s = self.data['s']
        self.s_total = self.data['s_total']
        self.px = self.data['px']
        self.py = self.data['py']
        self.pz = self.data['pz']
        self.p_total = self.data['p_total']
        self.dxy = self.data['dxy']
        self.dxz = self.data['dxz']
        self.dyz = self.data['dyz']
        self.dx2y2 = self.data['dx2y2']
        self.dz2r2 = self.data['dz2r2']
        self.d_total = self.data['d_total']

    def __repr__(self):
        return "Density('%s')" % self.calc_name

    def plot(self):
        """
        Creates a plot of the DOS spectra, with all d states in one plot
        :return: None
        """

        plt.figure(figsize=[12, 10], dpi=80)
        plt.plot(self.energy, self.d_total, 'k:', lw=3, label='Total')
        plt.plot(self.energy, self.dxy, '-', lw=2, label='d$_{xy}$')
        plt.plot(self.energy, self.dxz, '-', lw=2, label='d$_{xz}$')
        plt.plot(self.energy, self.dyz, '-', lw=2, label='d$_{yz}$')
        plt.plot(self.energy, self.dx2y2, '-', lw=2, label='d$_{x^2-y^2}$')
        plt.plot(self.energy, self.dz2r2, '-', lw=2, label='d$_{z^2-r^2}$')

        plt.legend(loc=0, frameon=False, fontsize=25)
        plt.title(self.calc_name, fontsize=22, fontname='Times New Roman')
        plt.xlabel('Energy [eV]', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Occupation', fontsize=28, fontname='Times New Roman')
        plt.xticks(fontsize=20, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')
        plt.ticklabel_format(useOffset=False)
        plt.ticklabel_format(style='sci', scilimits=(-3, 3))


class Spherical:
    def __init__(self, data, refname, calc_name):
        """
        Load contribution from spherical harmonics from out_sph_signal_rxs1.txt
        :param data: data array from read_spherical()
        :param refname: reflection name for plot title
        :param calc_name: calculation name for plot title
        """

        self.data = data
        self.calc_name = calc_name
        self.refname = refname

        self.energy = self.data['energy']
        self.d_total = self.data['D00']
        self.dxy = self.data['D_xy']
        self.dxz = self.data['D_xz']
        self.dyz = self.data['D_yz']
        self.dx2y2 = self.data['D_x2y2']
        self.dz2r2 = self.data['D_z2']

    def __repr__(self):
        return "Spherical('%s', '%s')" % (self.refname, self.calc_name)

    def plot(self):
        """
        Plot the spherical components
        :return: None
        """

        plt.figure(figsize=[12, 10], dpi=80)
        plt.plot(self.energy, self.d_total, 'k:', lw=3, label='Total')
        plt.plot(self.energy, self.dxy, '-', lw=2, label='d$_{xy}$')
        plt.plot(self.energy, self.dxz, '-', lw=2, label='d$_{xz}$')
        plt.plot(self.energy, self.dyz, '-', lw=2, label='d$_{yz}$')
        plt.plot(self.energy, self.dx2y2, '-', lw=2, label='d$_{x^2-y^2}$')
        plt.plot(self.energy, self.dz2r2, '-', lw=2, label='d$_{z^2-r^2}$')

        plt.legend(loc=0, frameon=False, fontsize=25)
        ttl = '%s %s' % (self.calc_name, self.refname)
        plt.title(ttl, fontsize=22, fontname='Times New Roman')
        plt.xlabel('Energy [eV]', fontsize=28, fontname='Times New Roman')
        plt.ylabel('|d$_{ii}$|$^2$', fontsize=28, fontname='Times New Roman')
        plt.xticks(fontsize=20, fontname='Times New Roman')
        plt.yticks(fontsize=20, fontname='Times New Roman')
        plt.ticklabel_format(useOffset=False)
        plt.ticklabel_format(style='sci', scilimits=(-3, 3))


"--------------------------------------------------------"
"------------------------ FUNCTIONS ---------------------"
"--------------------------------------------------------"


def fdmnes_checker(activate=False, fdmnes_filename='fdmnes_win64.exe', initial_dir=None):
    """Returns True if fdmnes available and activated"""

    if activate:
        find_fdmnes(fdmnes_filename=fdmnes_filename, reset=True, initial_dir=initial_dir)

    datadir = os.path.abspath(os.path.dirname(__file__))  # same directory as this file
    pointerfile = os.path.join(datadir, 'data', 'FDMNES_pointer.txt')

    if os.path.isfile(pointerfile):
        with open(pointerfile) as file:
            for location in file:
                loc = location.strip()
                if os.path.isfile(loc):
                    return True
    return False


def find_fdmnes(fdmnes_filename='fdmnes_win64.exe', reset=False, initial_dir=None):
    """
    Finds the path of the fdmnes_win64.exe file
     The name of the file is taken from the envrionment varialbe fdmnes_filename
     The first time this is run, it will take a while, searching through every folder for the file
     On completion, a file will be generated in the /data directory with the filepath
     Subsequent runs will quickly load from this file
    :param fdmnes_filename: name of the executable to search for
    :param reset: if True, pointerfile will be ignored.
    :param initial_dir: None or str, if directory, look here for file
    :return: str : location of fdmnes executable file
    """

    # Dans_Diffraction FDMNES pointer file
    datadir = os.path.abspath(os.path.dirname(__file__))  # same directory as this file
    pointerfile = os.path.join(datadir, 'data', 'FDMNES_pointer.txt')

    if not reset and os.path.isfile(pointerfile):
        with open(pointerfile) as file:
            for location in file:
                loc = location.strip()
                if os.path.isfile(loc):
                    return loc

    if os.path.isfile(fdmnes_filename):
        # Save location
        with open(pointerfile, 'w') as file:
            file.write(os.path.abspath(fdmnes_filename))
        return os.path.abspath(fdmnes_filename)

    initial_dir = os.path.abspath(os.sep) if initial_dir is None else initial_dir

    # Find FDMNES
    print('Hang on... just looking for %s...' % fdmnes_filename)
    for dirName, subdirList, fileList in os.walk(initial_dir):
        if fdmnes_filename in fileList:
            print('Found FDMNES at: %s' % dirName)
            location = os.path.join(dirName, fdmnes_filename)
            # Save location
            with open(pointerfile, 'w') as file:
                file.write(location)
            return location
    print('%s not found, please enter location manually' % fdmnes_filename)
    return os.path.abspath(os.sep)


def sim_folder(folder_name='', new_folder=False):
    r"""
    Generates a calculation path directory in the FDMNES/Sim directory
      If new_folder is True and the directory already exists, a number will be appended to the name
    :param folder_name: str folder name
    :param new_folder: True/False*
    :return: str directory E.G. 'c:\FDMNES\Sim\folder_name'
    """
    exe_path = find_fdmnes()
    newpath = os.path.join(os.path.dirname(exe_path), 'Sim', folder_name)
    if new_folder:
        n = 0
        while os.path.isdir(newpath):
            n += 1
            newpath = os.path.join(os.path.dirname(exe_path), 'Sim', folder_name + '_%d' % n)
    return newpath


def find_fdmnes_files(parent_directory=None, output_name='out'):
    """
    Finds all fdmnes calculations below a parent directory
    Assumes all files have the same output name
    :param parent_directory: top directory, searches lower directories recursivley
    :param output_name: e.g. 'out.txt'
    :return: list of filenames ['dir/out.txt']
    """

    if parent_directory is None:
        parent_directory = find_fdmnes()

    if '.txt' not in output_name:
        output_name = output_name + '_bav.txt'

    calc_dirs = []
    for dirpath, dirnames, filenames in os.walk(parent_directory):
        if output_name in filenames:
            print(dirpath)
            calc_dirs += [dirpath]
            # print(os.path.join(dirpath, filenames[0]))
        else:
            print('')
    return calc_dirs


def load_fdmnes_files(parent_directory=None, output_name='out'):
    """
    Finds all fdmnes calculations below a parent directory, loads each file
    Assumes all files have the same output name
    :param parent_directory: top directory, searches lower directories recursivley
    :param output_name: e.g. 'out.txt'
    :return: FdmnesCompare
    """
    calc_dirs = find_fdmnes_files(parent_directory, output_name)
    fdms = FdmnesCompare([])
    fdms.load(calc_dirs)
    return fdms


def read_conv(filename='out_conv.txt', plot=False):
    """
    Reads fdmnes output file out_conv.txt, that gives the XANES spectra
      energy, intensity = read_conv(filename)
    """

    filename = filename.replace('\\', '/')  # convert windows directories
    dirname, filetitle = os.path.split(filename)  # calculation directory
    calc_name = dirname.split('/')[-1]  # calculation nam

    data = np.loadtxt(filename, skiprows=1)
    energy = data[:, 0]
    xanes = data[:, 1]

    if plot:
        plt.figure(figsize=[12, 10], dpi=80)
        plt.plot(energy, xanes, lw=3)
        plt.title(calc_name, fontsize=26, fontweight='bold', fontname='Times New Roman')
        plt.xlabel('Energy [eV]', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Intensity [arb. units]', fontsize=28, fontname='Times New Roman')
        plt.xticks(fontsize=25, fontname='Times New Roman')
        plt.yticks(fontsize=25, fontname='Times New Roman')
    return energy, xanes


def read_scan_conv(filename='out_scan_conv.txt'):
    """
    Read FDMNES _scan_conv.txt files, return simulated azimuthal and energy values
       energy, angle, intensity = read_scan_conv('out_scan_conv.txt')
    You can see all the available reflections with intensity.keys()
    :param filename: str name of '*_scan_conv.txt' file from FDMNED calculation
    :return energy: [nx1] array of energy values
    :return angle: [mx1] array on angle values
    :return intensity:  {'I(100)ss'}[nxm] dict of arrays of simulated intensities for each reflection
    """

    # Open file
    with open(filename) as file:
        # Determine reflections in file
        filetext = file.read()  # generate string
        reftext = re.findall(r'I\(-?\d+-?\d+?-?\d+?\)\w+', filetext)  # find reflection strings

        refs = np.unique(reftext)  # remove duplicates
        Npeak = len(refs)
        Nenergy = len(reftext) // Npeak
        Nangle = 180

        file.seek(0)  # return to start of file

        # pre-define arrays
        storevals = {}
        for ref in refs:
            storevals[ref] = np.zeros([Nenergy, Nangle])

        storeeng = np.zeros(Nenergy)
        storeang = np.zeros(Nangle)

        # Read file, line by line
        for E in range(Nenergy):
            file.readline()  # blank line
            storeeng[E] = float(file.readline().strip())  # energy
            for P in range(Npeak):
                peak = file.readline().strip()  # current reflection
                # read angle,Intensity lines
                vals = np.zeros([Nangle, 2])
                for m in range(Nangle):
                    vals[m, :] = [float(x) for x in file.readline().split()]

                storeang = vals[:, 0]  # store angle values
                storevals[peak][E, :] = vals[:, 1]  # store intensity values
    return storeeng, storeang, storevals


def azi_cut(storeeng, intensities, cutenergy=None):
    """
    Generate azimuthal cut at a particular energy
    cutintensity = azi_cut(storeeng,intensities,cutenergy)
        storeeng = [nx1] array of energies from read_scan_conv
        intensities = [nxm] array of simulated intensities for a single reflection (e.g. storevals['I(100)sp'])
        cutenergy = energy to take the cut at, will take value closest to cutenergy. In eV.
        cutenergy = 'max' - take the cut energy at the maximum intensity.
        cutintensity = [mx1] array of simulated intensity at this energy

    e.g.
     energy,angle,intensities = read_scan_conv(filename)
     cutintensity = azi_cut(energy,intensities['I(100)sp'],cutenergy='max')
    """

    if cutenergy == 'max' or cutenergy is None:
        i, j = np.unravel_index(np.argmax(intensities), intensities.shape)
        print(' Highest value = {} at {} eV [{},{}]'.format(intensities[i, j], storeeng[i], i, j))
        cutenergy = storeeng[i]
    elif cutenergy == 'all' or cutenergy == 'sum':
        return np.sum(intensities, axis=0)

    enpos = np.argmin(abs(storeeng - cutenergy))
    if np.abs(storeeng[enpos] - cutenergy) > 5:
        print("You havent choosen the right cutenergy. enpos = {}".format(enpos, storeeng[enpos]))

    return intensities[enpos, :]


def eng_cut(storeang, intensities, cutangle=None):
    """
    Generate energy cut at a particular azimuthal angle
    cutintensity = eng_cut(storeang,intensities,cutangle)
        storeang = [mx1] array of angles from read_scan_conv
        intensities = [nxm] array of simulated intensities for a single reflection (e.g. storevals['I(100)sp'])
        cutangle = angle to take the cut at, will take value closest to cutenergy. In Deg.
        cutangle = 'max' - take the cut angle at the maximum intensity.
        cutintensity = [nx1] array of simulated intensity at this angle

    e.g.
     energies,angles,intensities = read_scan_conv(filename)
     cutintensity = eng_cut(angles,intensities['I(100)sp'],cutangle=180)
    """

    if cutangle == 'max' or cutangle is None:
        i, j = np.unravel_index(np.argmax(intensities), intensities.shape)
        print(' Highest value = {} at {} Deg [{},{}]'.format(intensities[i, j], storeang[j], i, j))
        cutangle = storeang[j]
    elif cutangle == 'all' or cutangle == 'sum':
        return np.sum(intensities, axis=1)

    angpos = np.argmin(abs(storeang - cutangle))
    if np.abs(storeang[angpos] - cutangle) > 5:
        print("You havent choosen the right cutangle. angpos = {} [{}]".format(angpos, storeang[angpos]))
    return intensities[:, angpos]


def potrmt(bav_text):
    """
    Get the final cycle charge values from the _bav.txt file

    :param bav_text: str _bav.txt file
    :return: dict with keys ['Z', 'charge', 'ch_ion', 'Vmft', 'Ionic radius']
    """

    # Header
    idx = bav_text.rfind('ch_ion')  # last occurance of ch_ion
    if idx == -1:
        return {}
    hline = bav_text[bav_text[:idx].rfind('\n'): bav_text[idx:].find('\n') + idx].strip()
    head = re.split(r'\s\s+', hline.strip())

    lines = bav_text[idx:].split('\n')
    values = []
    for line in lines[1:]:
        data = line.split()
        if len(data) != len(head): break
        data = [float(item.replace('*', '')) for item in data]
        values += [data]

    values = np.array(values)
    out = {}
    for n in range(len(head)):
        out[head[n]] = values[:, n]
    return out


def read_density(filename):
    """Load Density of states file from FDMNES calculations '..._sd0.txt'"""

    data = np.genfromtxt(filename, skip_header=0, names=True)

    # Old field names + new field names from fdmnes version 2020+
    key_names = {
        's': ['n00', 's'],
        's_total': ['n_l0', 'Ints'],
        'px': ['n11_1', 'px'],
        'py': ['n11', 'py'],
        'pz': ['n10', 'pz'],
        'p_total': ['n_l1', 'Intp'],
        'dxy': ['n22_1', 'dxy'],
        'dxz': ['n21_1', 'dxz'],
        'dyz': ['n21', 'dyz'],
        'dx2y2': ['n22', 'dx2y2'],
        'dz2r2': ['n20', 'dz2'],
        'd_total': ['n_l2', 'Intd'],
    }

    def get_data(key):
        values = np.zeros(len(data))
        for name in key_names[key]:
            if name in data.dtype.names:
                values = data[name]
        return values
    out = {
        key: get_data(key) for key in key_names
    }
    out['energy'] = data['Energy']
    return out


def read_spherical(filename):
    """Load contribution from spherical harmonics to resonant reflection, from e.g. out_sph_signal_rxs1.txt"""
    keys = ['D00', 'D_xy', 'D_xz', 'D_yz', 'D_x2y2', 'D_z2']
    data = np.genfromtxt(filename, skip_header=3, names=True)
    # Only load L3 part if "L23" is used
    if 'D(00)_r_L3' in data.dtype.names or 'D(00)_r_1_L3' in data.dtype.names:
        l3 = '_l3'
    else:
        l3 = ''

    def modulus(key):
        if key + '_r' + l3 in data.dtype.names:
            value = data[key + '_r' + l3] + 1j * data[key + '_i' + l3]
        elif key + '_r_1' + l3 in data.dtype.names:
            n = 1
            value = 0j
            while key + ('_r_%d' % n) + l3 in data.dtype.names:
                value += data[key + ('_r_%d' % n) + l3] + 1j * data[key + ('_i_%d' % n) + l3]
                n += 1
        else:
            value = 0.0
        return np.real(value * np.conj(value))
    out = {
        key: modulus(key) for key in keys
    }
    out['energy'] = data['Energy']
    return out
