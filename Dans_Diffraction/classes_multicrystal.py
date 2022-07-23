"""
classes_multicrystal.py
Class that combines multiple crystal classes with special functions

By Dan Porter, PhD
Diamond
2020

Version 1.1
Last updated: 27/09/21

Version History:
30/03/20 1.0    class MultiCrytal moved from classes_crystal
27/09/21 1.1    Added __get__, __repr__ methods

@author: DGPorter
"""

import numpy as np

from . import functions_general as fg
from . import functions_crystallography as fc
from .classes_plotting import MultiPlotting

__version__ = '1.1'


class MultiCrystal:
    """
    Multi_Crystal class for combining multiple phases
    """
    _scattering_type = 'xray'

    def __init__(self, crystal_list):
        """
        Multi-crystal class
        """
        self.crystal_list = crystal_list
        self.Plot = MultiPlotting(crystal_list)

    def set_scale(self, index, scale=1.0):
        """Set scale of crystal[index]"""
        xtl = self.crystal_list[index]
        xtl.scale = scale

    def orient_set_r(self, rotation):
        """Set rotation matrix in diffractometer frame"""
        for xtl in self.crystal_list:
            xtl.Cell.orientation.set_r(rotation)

    def orient_6circle(self, phi=0, chi=0, eta=0, mu=0):
        """Set rotation matrix using 6-circle diffractometer axes"""
        rotation = fc.diffractometer_rotation(phi, chi, eta, mu)
        self.orient_set_r(rotation)

    def set_labframe(self, lab):
        """Set transformation matrix between diffractometer and lab"""
        for xtl in self.crystal_list:
            xtl.Cell.orientation.set_lab(lab)

    def set_labframe_i16(self):
        """Set lab transformation matrix for beamline I16 at Diamond Light Source"""
        self.set_labframe([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    def setup_scatter(self, **kwargs):
        """
        Simple way to set scattering parameters, each parameter is internal to xtl (self)

        scattering_type : self._scattering type               :  'xray','neutron','xray magnetic','neutron magnetic','xray resonant'
        energy_kev  : self._energy_kev                    :  radiation energy in keV
        wavelength_a: self._wavelength_a                  :  radiation wavelength in Angstrom
        powder_units: self._powder_units                  :  units to use when displaying/ plotting ['twotheta', 'd',' 'q']
        min_twotheta: self._scattering_min_two_theta      :  minimum detector (two-theta) angle
        max_twotheta: self._scattering_max_two_theta      :  maximum detector (two-theta) angle
        min_theta   : self._scattering_min_theta          :  minimum sample angle = -opening angle
        max_theta   : self._scattering_max_theta          :  maximum sample angle = opening angle
        theta_offset: self._scattering_theta_offset       :  sample offset angle
        specular    : self._scattering_specular_direction : [h,k,l] : reflections normal to sample surface
        parallel    : self._scattering_parallel_direction : [h,k,l] : reflections normal to sample surface
        """

        if 'type' in kwargs:
            self._scattering_type = kwargs['type']
        elif 'scattering_type' in kwargs:
            self._scattering_type = kwargs['scattering_type']

        for xtl in self.crystal_list:
            xtl.Scatter.setup_scatter(**kwargs)

    def print_all_reflections(self, energy_kev=None, print_symmetric=False,
                              min_intensity=None, max_intensity=None, units=None):
        """
        Prints a list of all allowed reflections at this energy
            energy_kev = energy in keV
            print_symmetric = False*/True : omits reflections with the same intensity at the same angle
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
            units = None/ 'twotheta'/ 'q'/ 'dspace' : choose scattering angle units to display
        """

        if min_intensity is None: min_intensity = -1
        if max_intensity is None: max_intensity = np.inf

        if energy_kev is None:
            energy_kev = self.crystal_list[0].Scatter._energy_kev
        if units is None:
            units = self.crystal_list[0].Scatter._powder_units

        hkl = np.empty(shape=(0,3))
        tth = np.empty(shape=0)
        inten = np.empty(shape=0)
        name = np.empty(shape=0)
        for xtl in self.crystal_list:
            hkl_xtl = xtl.Cell.all_hkl(energy_kev, xtl.Scatter._scattering_max_twotheta)
            if not print_symmetric:
                hkl_xtl = xtl.Symmetry.remove_symmetric_reflections(hkl_xtl)
            hkl_xtl = xtl.Cell.sort_hkl(hkl_xtl)
            # remove [0,0,0]
            hkl_xtl = hkl_xtl[1:, :]

            tth_xtl = xtl.Cell.tth(hkl_xtl, energy_kev)
            inrange = np.all([tth_xtl < xtl.Scatter._scattering_max_twotheta, tth_xtl > xtl.Scatter._scattering_min_twotheta], axis=0)
            hkl_xtl = hkl_xtl[inrange, :]
            tth_xtl = tth_xtl[inrange]
            inten_xtl = xtl.Scatter.intensity(hkl_xtl)

            name = np.append(name, [xtl.name]*len(tth_xtl))
            hkl = np.vstack((hkl, hkl_xtl))
            tth = np.append(tth, tth_xtl)
            inten = np.append(inten, inten_xtl)

        # Sort reflections from all reflections
        idx = np.argsort(tth)
        name = name[idx]
        hkl = hkl[idx, :]
        tth = tth[idx]
        inten = inten[idx]

        units = units.lower()
        if units in ['d', 'dspc', 'dspace', 'd space', 'd-space', 'dspacing', 'd spacing', 'd-spacing']:
            unit_str = 'd-spacing'
            unit = fc.caldspace(tth, energy_kev)
        elif units in ['q', 'wavevector']:
            unit_str = 'Q'
            unit = fc.calqmag(tth, energy_kev)
        else:
            unit_str = 'TwoTheta'
            unit = tth

        fmt = '(%3.0f,%3.0f,%3.0f) %10.2f  %9.2f   %s\n'
        outstr = ''

        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Radiation: %s\n' % self._scattering_type
        outstr += '( h, k, l)    %10s  Intensity  Crystal\n' % unit_str
        # outstr+= fmt % (hkl[0,0],hkl[0,1],hkl[0,2],tth[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(0, len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            count += 1
            outstr += fmt % (hkl[n, 0], hkl[n, 1], hkl[n, 2], unit[n], inten[n], name[n])
        outstr += 'Reflections: %1.0f\n' % count
        return outstr

    def find_close_reflections(self, HKL, energy_kev=None, max_twotheta=2, max_angle=10):
        """
        Find reflections close to the given one  and return list
        """

        if energy_kev is None:
            energy_kev = self.crystal_list[0].Scatter._energy_kev

        HKL_tth = self.crystal_list[0].Cell.tth(HKL, energy_kev)
        HKL_Q = self.crystal_list[0].Cell.calculateQ(HKL)

        HKL_list = np.empty([0, 3])
        TTH_list = np.empty([0])
        ANGLE_list = np.empty([0])
        I_list = np.empty([0])
        NAMES_list = np.empty([0])
        for xtl in self.crystal_list:
            # xtl._scattering_type = self._scattering_type
            all_HKL = xtl.Cell.all_hkl(energy_kev, xtl._scattering_max_twotheta)
            all_TTH = xtl.Cell.tth(all_HKL, energy_kev)
            dif_TTH = np.abs(all_TTH - HKL_tth)
            all_Q = xtl.Cell.calculateQ(all_HKL)
            all_ANG = np.abs([fg.ang(HKL_Q, Q, 'deg') for Q in all_Q])
            selected = (dif_TTH < max_twotheta) * (all_ANG < max_angle)
            sel_HKL = all_HKL[selected, :]
            sel_TTH = all_TTH[selected]
            sel_ANG = all_ANG[selected]
            sel_INT = xtl.Scatter.intensity(sel_HKL)
            NAMES = np.asarray(xtl.name).repeat(len(sel_INT))

            HKL_list = np.append(HKL_list, sel_HKL, axis=0)
            TTH_list = np.append(TTH_list, sel_TTH)
            ANGLE_list = np.append(ANGLE_list, sel_ANG)
            I_list = np.append(I_list, sel_INT)
            NAMES_list = np.append(NAMES_list, NAMES)

        # Sort by TTH
        index = np.argsort(TTH_list)
        HKL_list = HKL_list[index, :]
        TTH_list = TTH_list[index]
        ANGLE_list = ANGLE_list[index]
        I_list = I_list[index]
        NAMES_list = NAMES_list[index]

        fmt = '(%3.0f,%3.0f,%3.0f) %8.2f %8.2f %9.2f %s\n'

        out = 'Energy = %6.3f keV\n' % energy_kev()
        out += '%s Reflection: (%3.0f,%3.0f,%3.0f)\n' % (self.crystal_list[0].name, HKL[0], HKL[1], HKL[2])
        out += '( h, k, l) TwoTheta Angle    Intensity Crystal\n'
        for n in range(0, len(TTH_list)):
            print(fmt % (
                HKL_list[n, 0], HKL_list[n, 1], HKL_list[n, 2], TTH_list[n], ANGLE_list[n], I_list[n], NAMES_list[n]))
        return out

    def remove(self, index=-1):
        """Remove crystal [index] from crystal list"""
        del(self.crystal_list[index])

    def __add__(self, other):
        if type(other) is MultiCrystal:
            return MultiCrystal(self.crystal_list+other.crystal_list)
        return MultiCrystal(self.crystal_list + [other])

    def __get__(self, item):
        return self.crystal_list[item]

    def info(self):
        """Display information about the contained crystals"""

        out = "Crystals: %d\n" % len(self.crystal_list)
        for n, xtl in enumerate(self.crystal_list):
            out += "%1.0f %r\n" % (n, xtl)
        return out

    def __repr__(self):
        s = ', '.join([xtl.name for xtl in self.crystal_list])
        return "MultiCrystal([%s])" % s

    def __str__(self):
        return self.info()

