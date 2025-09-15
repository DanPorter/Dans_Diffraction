# -*- coding: utf-8 -*-
"""
Scattering Class "classes_scattering.py"
    Scattering functions for Crystal class.

By Dan Porter, PhD
Diamond
2017

Version 2.3.5
Last updated: 23/12/24

Version History:
10/09/17 0.1    Program created
30/10/17 1.0    Main functions finshed, some testing done
06/01/18 1.1    Renamed classes_scattering.py
31/10/18 1.2    Added print_symmetric_contributions
21/01/19 1.3    Added non-resonant diffraction, corrected resonant diffraction
16/12/19 1.4    Added multiple_scattering code, print_all_reflections updated with units
18/02/20 1.5    Added tensor_scattering code
20/03/20 1.6    Increased powder gauss width from 2fwhm to 6fwhm, added powder averaging
14/04/20 1.6    Added powder_correction
26/05/20 1.7    Removed tensor_scattering
16/06/20 1.7.1  Added output option of setup_scatter
04/01/21 1.7.2  Added structure_factor function
21/01/21 1.8    Added xray_dispersion scattering function
10/06/21 1.9    Added x_ray calculation using Waasmaier and Kirfel scattering factors.
09/07/21 1.9    Added new scattering factors as option on normal scattering functions
20/08/21 2.0    Switched over to new scattering module, added self.powder()
28/09/21 2.0.1  Added __repr__
08/02/22 2.0.3  Corrected error in powder of wrong tth values. Thanks Mirko!
14/03/22 2.1.0   powder() updated for new inputs and outputs for pVoight and custom peak shapes. Thanks yevgenyr!
14/01/23 2.1.1  Corrected background error in xtl.Scatter.powder
06/05/23 2.0.0  Merged pull request for non-integer hkl option on SF and electron form factors. Thanks Prestipino!
02/07/23 2.3.0  Fixed rounding error in Scatter.powder, thanks Sergio I. Rincon!
26/09/23 2.3.1  Added Scattering.orientation_reflections for automatic orientation help
19/20/23 2.3.2  Fixed scatteringbasis so xray_resonant() now works with non-cubic systems
28/03/24 2.3.3  Fixed scattering type comparison to compare .lower() scattering types
02/05/24 2.3.4  added min_twotheta to get_hkl, added generate_envelope_cut, fixed low tth error in powder()
15/05/24 2.3.4  Added "save" and "load" methods to structure factor calculation, improved powder for large calculations
16/05/24 2.3.4  Added printed progress bar to generate_intensity_cut during convolusion
17/05/24 2.3.4  Changed generate_intensity_cut to make it much faster
23/12/24 2.3.5  Added polarised neutron options
15/09/25 2.4.0  Added custom scattering factors

@author: DGPorter
"""

import numpy as np
import datetime

from . import functions_general as fg
from . import functions_crystallography as fc
from . import functions_scattering as fs
from . import multiple_scattering as ms
# from . import tensor_scattering as ts  # Removed V1.7

__version__ = '2.4.0'


class Scattering:
    """
    Simulate diffraction from Crystal
    Useage:
        xtl = Crystal()
        xtl.Scatter.setup_scatter(type='x-ray',energy_keV=8.0)
        xtl.Scatter.intensity([h,k,l]) # Returns intensity
        print(xtl.Scatter.print_all_refelctions()) # Returns formated string of all allowed reflections
        
        Allowed radiation types:
            'xray','neutron','xray magnetic','neutron magnetic','xray resonant', 'custom'
    """
    
    #------Options-------
    # Standard Options
    _hkl = None  # added so recalculation not required
    _scattering_type = 'xray'  # 'xray','neutron','xray magnetic','neutron magnetic','xray resonant', 'custom'
    _scattering_specular_direction = [0, 0, 1]  # reflection
    _scattering_parallel_direction = [0, 0, 1]  # transmission
    _scattering_theta_offset = 0.0
    _scattering_min_theta = -180.0
    _scattering_max_theta = 180.0
    _scattering_min_twotheta = -180.0
    _scattering_max_twotheta = 180.0
    _integer_hkl = True
    _time_report = True
    _debug_mode = False

    # powder options
    _powder_units = 'tth'  # tth (two theta), Q, d
    _powder_background = 0.0
    _powder_peak_width = 0.01  # in Deg
    _powder_lorentz_fraction = 0.5
    _powder_average = True
    _powder_pixels = 2000  # no. pixels per inverse angstrom (in q)
    _powder_min_overlap = 0.02
    
    # Complex Structure factor
    _return_structure_factor = False

    # Uses the coefficients for analytical approximation to the scattering factors from:
    #        "Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431"
    _use_waaskirf_scattering_factor = False

    # Use the neutron scattering factors from the Internation tables (Sears 1995)
    _use_sears_scattering_lengths = False
    
    # Thermal Factors
    _use_isotropic_thermal_factor = True
    _use_anisotropic_thermal_factor = False
    
    # Magnetic Options
    _calclate_magnetic_component = True
    _use_magnetic_form_factor = True
    
    # Polarisation Options
    _polarised = False
    _polarisation = 'sp'
    _polarisation_vector_incident = (0, 1, 0)
    
    # Radiation energy
    _energy_kev = fg.Cu
    
    # Resonant X-ray Options
    _azimuthal_angle = 0
    _azimuthal_reference = [1, 0, 0]
    _resonant_flm = (0, 1, 0)
    _resonant_approximation_e1e1 = True
    _resonant_approximation_e2e2 = False
    _resonant_approximation_e1e2 = False
    _resonant_approximation_m1m1 = False
    
    def __init__(self, xtl):
        self.xtl = xtl

        # Initialise the scattering type container
        self.Type = ScatteringTypes(self, fs.SCATTERING_TYPES)

    def __repr__(self):
        return 'Scattering(%s, %s)' % (self.xtl.name, self._scattering_type)

    def __str__(self):
        out = 'Scatter(%r)\n' % self.xtl
        out += '       Type: %s\n' % self._scattering_type
        out += '     Energy: %s keV\n' % self._energy_kev
        out += ' Wavelength: %s A\n' % fc.energy2wave(self._energy_kev)
        out += ' ---Settings---\n'
        out += '      Powder units: %s\n' % self._powder_units
        out += '    Isotropic ADPs: %s\n' % self._use_isotropic_thermal_factor
        #out += '  Anisotropic ADPs: %s\n' % self._use_anisotropic_thermal_factor
        out += '  Specular Direction (reflection): (%2.0f,%2.0f,%2.0f)\n' % (
            self._scattering_specular_direction[0], self._scattering_specular_direction[1],
            self._scattering_specular_direction[2])
        out += 'Parallel Direction (transmission): (%2.0f,%2.0f,%2.0f)\n' % (
            self._scattering_parallel_direction[0], self._scattering_parallel_direction[1],
            self._scattering_parallel_direction[2])
        out += '      theta offset: %s\n' % self._scattering_theta_offset
        out += '         min theta: %s\n' % self._scattering_min_theta
        out += '         max theta: %s\n' % self._scattering_max_theta
        out += '      min twotheta: %s\n' % self._scattering_min_twotheta
        out += '      max twotheta: %s\n' % self._scattering_max_twotheta
        out += ' ---X-Ray Settings---\n'
        out += ' Waasmaier scattering factor: %s\n' % self._use_waaskirf_scattering_factor
        out += ' ---Neutron Settings---\n'
        out += ' Sears (ITC) scattering lengths: %s\n' % self._use_sears_scattering_lengths
        out += ' ---Magnetic Settings---\n'
        out += '   Mag. scattering: %s\n' % self._calclate_magnetic_component
        out += '  Mag. form factor: %s\n' % self._use_magnetic_form_factor
        out += '         Polarised: %s\n' % self._polarised
        out += '      Polarisation: %s\n' % self._polarisation
        out += '       Pol. vector: (%s,%s,%s)\n' % (self._polarisation_vector_incident[0],
                                                     self._polarisation_vector_incident[1],
                                                     self._polarisation_vector_incident[2])
        out += ' ---Resonant Settings---\n'
        out += '   Azimuthal angle: %s\n' % self._azimuthal_angle
        out += '    Azimuthal ref.: (%s,%s,%s)\n' % (self._azimuthal_reference[0], self._azimuthal_reference[1],
                                                     self._azimuthal_reference[2])
        out += '               flm: (%s,%s,%s)\n' % (self._resonant_flm[0], self._resonant_flm[1],
                                                     self._resonant_flm[2])
        out += '  use e1e1 approx.: %s\n' % self._resonant_approximation_e1e1
        #out += '  use e2e2 approx.: %s\n' % self._resonant_approximation_e2e2
        #out += '  use e1e2 approx.: %s\n' % self._resonant_approximation_e1e2
        #out += '  use m1m1 approx.: %s\n' % self._resonant_approximation_m1m1
        return out

    def setup_scatter(self, scattering_type=None, energy_kev=None, energy_mev=None, wavelength_a=None,
                      use_sears=None, use_waaskirf=None,
                      powder_units=None, powder_pixels=None, powder_lorentz=None, powder_overlap=None,
                      int_hkl=None, specular=None, parallel=None, theta_offset=None,
                      min_theta=None, max_theta=None, min_twotheta=None, max_twotheta=None,
                      output=True, scattering_factors=None, scattering_lengths=None, magnetic_formfactor=None,
                      polarisation=None, polarisation_vector=None, azimuthal_reference=None, azimuth=None, flm=None):
        """
        Simple way to set scattering parameters, each parameter is internal to xtl (self)

        scattering_type: self._scattering type            :  'xray','neutron','xray magnetic','neutron magnetic','xray resonant', 'xray dispersion', 'neutron polarised', 'xray polarised'
        energy_kev  : self._energy_kev                    :  radiation energy in keV
        wavelength_a: self._wavelength_a                  :  radiation wavelength in Angstrom
        powder_units: self._powder_units                  :  units to use when displaying/ plotting ['twotheta', 'd',' 'q']
        powder_pixels: self._powder_pixels                :  number of bins per inverse-angstrom in the powder spectrum
        powder_lorentz: self._powder_lorentz_fraction     :  the fraction of Lorentzian in the peak function psuedo-Voight
        powder_overlap: self._powder_min_overlap          :  minimum overlap of grouped reflections in powder
        int_hkl: self._integer_hkl                        :  round hkl values to integers
        min_twotheta: self._scattering_min_two_theta      :  minimum detector (two-theta) angle
        max_twotheta: self._scattering_max_two_theta      :  maximum detector (two-theta) angle
        min_theta   : self._scattering_min_theta          :  minimum sample angle = -opening angle
        max_theta   : self._scattering_max_theta          :  maximum sample angle = opening angle
        theta_offset: self._scattering_theta_offset       :  sample offset angle
        specular    : self._scattering_specular_direction : [h,k,l] : reflections normal to sample surface
        parallel    : self._scattering_parallel_direction : [h,k,l] : reflections normal to sample surface
        scattering_factors: self._use_waaskirf_scattering_factor : xray scattering factor ['waaskirf', 'itc']
        scattering_lengths: self._use_sears_scattering_lengths : neutron scattering lengths ['sears', 'default']
        magnetic_formfactor: self._use_magnetic_form_factor: True/False magnetic form factor for magnetic SF
        polarisation: self._polarisation                  : beam polarisation setting ['ss', 'sp'*, 'sp', 'pp']
        polarisation_vector: _polarisation_vector_incident: [x,y,z] incident polarisation vector
        azimuthal_reference: self._azimuthal_reference    : [h,k,l] direction of azimuthal zero angle
        azimuth    : self._azimuthal_angle                : azimuthal angle in deg
        flm        : self._resonant_flm                   : Resonant settings (flm1, flm2, flm3)
        """

        if scattering_type is not None:
            self._scattering_type = fs.get_scattering_type(scattering_type)

        if energy_kev is not None:
            self._energy_kev = energy_kev

        if energy_mev is not None:
            self._energy_kev = energy_mev * 1e-6

        if wavelength_a is not None:
            if 'neutron' in self._scattering_type:
                self._energy_kev = fc.neutron_energy(wavelength_a) * 1e-6  # meV
            elif 'electron' in self._scattering_type:
                self._energy_kev = fc.electron_energy(wavelength_a) * 1e-3  # eV
            else:
                self._energy_kev = fc.wave2energy(wavelength_a)

        if use_sears is not None:
            self._use_sears_scattering_lengths = use_sears

        if use_waaskirf is not None:
            self._use_waaskirf_scattering_factor = use_waaskirf

        if powder_units is not None:
            self._powder_units = powder_units

        if powder_pixels is not None:
            self._powder_pixels = powder_pixels

        if powder_lorentz is not None:
            self._powder_lorentz_fraction = powder_lorentz

        if powder_overlap is not None:
            self._powder_min_overlap = powder_overlap

        if int_hkl is not None:
            self._integer_hkl = int_hkl

        if specular is not None:
            self._scattering_specular_direction = specular

        if parallel is not None:
            self._scattering_parallel_direction = parallel

        if theta_offset is not None:
            self._scattering_theta_offset = theta_offset

        if min_theta is not None:
            self._scattering_min_theta = min_theta

        if max_theta is not None:
            self._scattering_max_theta = max_theta

        if min_twotheta is not None:
            self._scattering_min_twotheta = min_twotheta

        if max_twotheta is not None:
            self._scattering_max_twotheta = max_twotheta

        if scattering_factors is not None:
            if scattering_factors.lower() in ['ws', 'waaskirf', 'alternate', 'alt']:
                print('Using scattering factors from: "Waasmaier and Kirfel, Acta Cryst. (1995) A51, 416-431"')
                self._use_waaskirf_scattering_factor = True
            else:
                print('Using scattering factors from: International Tables of Crystallography Vol. C, Table 6.1.1.4')
                self._use_waaskirf_scattering_factor = False

        if scattering_lengths is not None:
            if scattering_lengths.lower() in ['sears', 'itc', 'alternate', 'alt']:
                print('Using scattering lengths from International Tables of Crystallography Vol. C, Table 4.4.4.1')
                self._use_sears_scattering_lengths = True
            else:
                print('Using scattering lengths from Neutron Data Booklet')
                self._use_sears_scattering_lengths = False

        if magnetic_formfactor is not None:
            self._use_magnetic_form_factor = magnetic_formfactor

        if polarisation_vector is not None:
            self._polarisation_vector_incident = np.array(polarisation_vector, dtype=float).reshape(3)

        if polarisation is not None:
            self._polarisation = polarisation

        if azimuthal_reference is not None:
            self._azimuthal_reference = np.array(azimuthal_reference, dtype=float).reshape(3)

        if azimuth is not None:
            self._azimuthal_angle = azimuth

        if flm is not None:
            self._resonant_flm = np.array(flm).reshape(3)

        if output:
            print(self)

    def get_energy(self, **kwargs):
        """
        Return energy
        :param kwargs: energy_kev, wavelength_a
        :return: energy_kev
        """
        if 'energy_kev' in kwargs:
            return kwargs['energy_kev']
        if 'wavelength_a' in kwargs:
            return fc.wave2energy(kwargs['wavelength_a'])
        return self._energy_kev

    def get_hkl(self, regenerate=True, remove_symmetric=False, reflection=False, transmission=False,  **kwargs):
        """
        Return stored hkl or generate
        :param regenerate: if True, hkl list will be regenerated, if False - previous list will be returned
        :param remove_symmetric: generate only non-symmetric hkl values
        :param reflection: generate only reflections possible in reflection geometry
        :param transmission: generate only reflections possible in transmission geometry
        :param kwargs: additional options to pass to setup_scatter()
        :return: array
        """
        if not regenerate and self._hkl is not None:
            return self._hkl
        self.setup_scatter(output=False, **kwargs)

        en = self._energy_kev
        max_tth = self._scattering_max_twotheta

        hkl = self.xtl.Cell.all_hkl(en, max_tth)
        if remove_symmetric:
            hkl = self.xtl.Symmetry.remove_symmetric_reflections(hkl)
        hkl = self.xtl.Cell.sort_hkl(hkl)[1:]  # remove (0, 0, 0)

        tth = self.xtl.Cell.tth(hkl, en)
        hkl = hkl[tth > self._scattering_min_twotheta, :]
        tth = tth[tth > self._scattering_min_twotheta]

        if reflection:
            # tth = self.xtl.Cell.tth(hkl, en)
            # hkl = hkl[tth > self._scattering_min_twotheta, :]
            # tth = tth[tth > self._scattering_min_twotheta]
            theta = self.xtl.Cell.theta_reflection(hkl, en, self._scattering_specular_direction,
                                                   self._scattering_theta_offset)
            p1 = (theta > self._scattering_min_theta) * (theta < self._scattering_max_theta)
            p2 = (tth > (theta + self._scattering_min_theta)) * (tth < (theta + self._scattering_max_theta))
            hkl = hkl[p1 * p2]
        elif transmission:
            # tth = self.xtl.Cell.tth(hkl, en)
            # hkl = hkl[tth > self._scattering_min_twotheta, :]
            # tth = tth[tth > self._scattering_min_twotheta]
            theta = self.xtl.Cell.theta_transmission(hkl, en, self._scattering_parallel_direction)

            p1 = (theta > self._scattering_min_theta) * (theta < self._scattering_max_theta)
            p2 = (tth > (theta + self._scattering_min_theta)) * (tth < (theta + self._scattering_max_theta))
            hkl = hkl[p1 * p2]
        self._hkl = hkl
        return self._hkl

    def _debug(self):
        """Toggle on debug mode"""
        if self._debug_mode:
            print('Debug mode: off')
            self._debug_mode = False
            fs.DEBUG_MODE = False
        else:
            print('Debug mode: on')
            self._debug_mode = True
            fs.DEBUG_MODE = True

    def structure_factor(self, hkl=None, scattering_type=None, int_hkl=None, **kwargs):
        """
        Calculate the structure factor at reflection indexes (h,k,l)
                sf = sum( f_i * occ_i * dw_i * exp( -i * 2 * pi * hkl.uvw )
        Where f_i is the elemental scattering factor, occ_i is the site occupancy, dw_i
        is the Debye-Waller thermal factor, hkl is the reflection and uvw is the site position.

        The following options for scattering_type are  supported:
          'xray'  - uses x-ray form factors
          'neutron' - uses neutron scattering lengths
          'xray magnetic' - calculates the magnetic (non-resonant) component of the x-ray scattering
          'neutron magnetic' - calculates the magnetic component of neutron scattering with average polarisation
          'xray resonant' - calculates magnetic resonant scattering
          'xray dispersion' - uses x-ray form factors including f'-if'' components
          'neutron polarised' - calcualtes magnetic component with incident polarised neutrons
          'xray polarised' - calcualtes magnetic component with incident polarised x-rays

        Notes:
        - Uses x-ray atomic form factors, calculated from approximated tables in the ITC
        - Debye-Waller factor (atomic displacement) is applied for isotropic ADPs
        - Crystal.scale is used to scale the complex structure factor, so the intensity is
         reduced by (Crystal.scale)^2
        - Testing against structure factors calculated by Vesta.exe is exactly the same when using Waasmaier structure factors.

        Save/Load behaviour
            structure_factor(..., save='sf.npy')  # saves structure factors after calculation in compressed file
            structure_factor(..., load='sf.npy')  # loads structure factors if sf.npy, rather than calculating
              - if the number of values in load doesn't match the number of hkl indices, an Exception is raised

        :param hkl: array[n,3] : reflection indexes (h, k, l)
        :param scattering_type: str : one of ['xray','neutron','xray magnetic','neutron magnetic','xray resonant']
        :param int_hkl: Bool : when True, hkl values are converted to integer.
        :param kwargs: additional options to pass to scattering function
        :return: complex array[n] : structure factors
        """

        if hkl is None:
            hkl = self.get_hkl()
        if scattering_type is None:
            scattering_type = self._scattering_type
        # scattering_type = scattering_type.lower()
        if int_hkl is None:
            int_hkl = self._integer_hkl
        if int_hkl:
            hkl = np.asarray(np.rint(hkl), dtype=float).reshape([-1, 3])
        else:
            hkl = np.asarray(hkl, dtype=float).reshape([-1, 3])
        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()

        q = self.xtl.Cell.calculateQ(hkl)
        r = self.xtl.Cell.calculateR(uvw)

        moment = fc.euler_moment(mxmymz, self.xtl.Cell.UV())
        azi_ref_q = self.xtl.Cell.calculateQ(self._azimuthal_reference)

        if 'energy_kev' in kwargs:
            energy_kev = kwargs.pop('energy_kev')
        elif 'wavelength_a' in kwargs:
            energy_kev = fc.wave2energy(kwargs.pop('wavelength_a'))
        else:
            energy_kev = self._energy_kev
        energy_kev = np.asarray(energy_kev, dtype=float).reshape(-1)
        nenergy = len(energy_kev)

        if 'psi' in kwargs:
            psi = kwargs.pop('psi')
        else:
            psi = self._azimuthal_angle
        psi = np.asarray(psi, dtype=float).reshape(-1)
        npsi = len(psi)

        options = fs.options(
            occ=occ,
            moment=moment,
            incident_polarisation_vector=self._polarisation_vector_incident,
            polarisation=self._polarisation,
            azi_ref_q=azi_ref_q,
            f0=self._resonant_flm[0],
            f1=self._resonant_flm[1],
            f2=self._resonant_flm[2],
        )
        options.update(kwargs)
        scattering_fun = fs.get_scattering_function(scattering_type)
        # print('Scattering function: %s' % scattering_fun.__name__)

        if 'load' in kwargs:
            # Load SF from file
            sf = np.load(kwargs['load'])
            if sf.shape[0] != len(hkl):
                raise Exception(
                    "File '%s' with %d structure factors doesn't match %d reflections" % (
                        kwargs['load'], sf.shape[0], len(hkl))
                )
            return sf

        # Break up long lists of HKLs
        nref, natom = len(q), len(r)
        if nref == 0:
            if nenergy == 1 and npsi == 1:
                return np.empty([0], dtype=complex)  # shape(nref)
            if nenergy == 1:
                return np.empty([0, nenergy], dtype=complex)  # shape(nref)  # shape(nref, nenergy)
            if npsi == 1:
                return np.empty([0, npsi], dtype=complex)  # shape(nref)  # shape(nref, npsi)
            return np.empty([0, nenergy, npsi], dtype=complex)
        n_arrays = np.ceil(nref * natom / fs.MAX_QR_ARRAY)
        if n_arrays > 1:
            print('Splitting %d reflections (%d atoms) into %1.0f parts' % (nref, natom, n_arrays))
        q_array = np.array_split(q, n_arrays)
        sf = np.zeros([nref, nenergy, npsi], dtype=complex)
        start_time = datetime.datetime.now()
        for e, enval in enumerate(energy_kev):  # for all the energy values
            for p, psival in enumerate(psi):   # for all azimutal angle
                ls = 0
                for n, _q in enumerate(q_array):  # for all the reflections parts
                    if n_arrays > 1:
                        print(' Starting %2.0f/%2.0f: %d:%d' % (n + 1, n_arrays, ls, ls + len(_q)))
                    qmag = fg.mag(_q)      # q magnitude
                    # Scattering factors
                    ff = fs.scattering_factors(
                        scattering_type=scattering_type,
                        atom_type=atom_type,
                        qmag=qmag,
                        enval=enval,
                        use_sears=self._use_sears_scattering_lengths,
                        use_wasskirf=self._use_waaskirf_scattering_factor
                    )

                    # Get Debye-Waller factor
                    if self._use_isotropic_thermal_factor:
                        dw = fc.debyewaller(uiso, qmag)
                    elif self._use_anisotropic_thermal_factor:
                        raise Exception('anisotropic thermal factor calcualtion not implemented yet')
                    else:
                        dw = None

                    # Get magnetic form factors
                    if self._use_magnetic_form_factor:
                        mf = fc.magnetic_form_factor(*atom_type, qmag=qmag)
                    else:
                        mf = None

                    options['scattering_factor'] = ff
                    options['debyewaller'] = dw
                    options['magnetic_formfactor'] = mf
                    options['energy_kev'] = enval
                    options['psi'] = psival

                    sf[ls: ls + len(_q), e, p] = scattering_fun(_q, r, **options)
                    ls = ls + len(_q)

        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        if self._time_report and time_difference.total_seconds() > 10:
            print('Calculated %d structure factors in %s' % (nref, time_difference))
        sf = sf / self.xtl.scale
        if nenergy == 1 and npsi == 1:
            sf = sf[:, 0, 0]  # shape(nref)
        elif nenergy == 1:
            sf = sf[:, 0, :]  # shape(nref, npsi)
        elif npsi == 1:
            sf = sf[:, :, 0]  # shape(nref, nenergy)
        if 'save' in kwargs:
            np.save(kwargs['save'], sf, allow_pickle=True)
            print("Saved %d structure factors to '%s'" % (sf.size, kwargs['save']))
        return sf
    new_structure_factor = structure_factor

    def intensity(self, hkl=None, scattering_type=None, int_hkl=None, **options):
        """
        Return the structure factor squared
                I = |sum( f_i * occ_i * dw_i * exp( -i * 2 * pi * hkl.uvw ) |^2
        Where f_i is the elemental scattering factor, occ_i is the site occupancy, dw_i
        is the Debye-Waller thermal factor, hkl is the reflection and uvw is the site position.

        The following options for scattering_type are  supported:
          'xray'  - uses x-ray form factors
          'neutron' - uses neutron scattering lengths
          'xray magnetic' - calculates the magnetic (non-resonant) component of the x-ray scattering
          'neutron magnetic' - calculates the magnetic component of neutron scattering
          'xray resonant' - calculates magnetic resonant scattering
          'xray dispersion' - uses x-ray form factors including f'-if'' components

        :param hkl: array[n,3] : reflection indexes (h, k, l)
        :param scattering_type: str : one of ['xray','neutron', 'electron', 'xray magnetic','neutron magnetic','xray resonant']
        :param int_hkl: Bool : when True, hkl values are converted to integer.
        :param kwargs: additional options to pass to scattering function
        :return: float array[n] : array of |SF|^2
        """
        return fs.intensity(self.new_structure_factor(hkl, scattering_type, int_hkl, **options))
    new_intensity = intensity

    def powder(self, scattering_type=None, units=None, peak_width=None, background=None, pixels=None,
               powder_average=None, lorentz_fraction=None, custom_peak=None, min_overlap=None, **options):
        """
        Generates array of intensities along a spaced grid, equivalent to a powder pattern.
          tth, inten, reflections = Scatter.powder('xray', units='tth', energy_kev=8)

        Note: This function is the new replacement for generate_power and uses both _scattering_min_twotheta
        and _scattering_max_twotheta.

        :param scattering_type: str : one of ['xray','neutron','xray magnetic','neutron magnetic','xray resonant']
        :param units: str : one of ['tth', 'dspace', 'q']
        :param peak_width: float : Peak with in units of inverse wavevector (Q)
        :param background: float : if >0, a normal background around this value will be added
        :param pixels: int : number of pixels per inverse-anstrom to add to the resulting mesh
        :param powder_average: Bool : if True, intensities will be reduced for the powder average
        :param lorentz_fraction: float 0-1: sets the Lorentzian fraction of the psuedo-Voight peak functions
        :param custom_peak: array: if not None, the array will be convolved with delta-functions at each reflection.
        :param min_overlap: minimum overlap of neighboring reflections.
        :param options: additional arguments to pass to intensity calculation
        :return xval: arrray : x-axis of powder scan (units)
        :return inten: array :  intensity values at each point in x-axis
        :return reflections: (h, k, l, xval, intensity) array of reflection positions, grouped by min_overlap
        """
        if scattering_type is None:
            scattering_type = self._scattering_type
        if units is None:
            units = self._powder_units
        if peak_width is None:
            peak_width = self._powder_peak_width
        if background is None:
            background = self._powder_background
        if pixels is None:
            pixels = self._powder_pixels
        if powder_average is None:
            powder_average = self._powder_average
        if lorentz_fraction is None:
            lorentz_fraction = self._powder_lorentz_fraction
        if min_overlap is None:
            min_overlap = self._powder_min_overlap
        energy_kev = self.get_energy(**options)

        # Extend range to account for peak widths beyond range
        ext = 3  # int, multiples of the peak_width used

        # Units
        min_twotheta = self._scattering_min_twotheta
        if min_twotheta <= 1.0: min_twotheta = 1.0
        max_twotheta = self._scattering_max_twotheta
        if max_twotheta >= 179: max_twotheta = 179
        q_min = fc.calqmag(min_twotheta, energy_kev) - ext * peak_width
        q_max = fc.calqmag(max_twotheta, energy_kev) + ext * peak_width
        if q_min <= 0: q_min = 0.0
        if q_max >= fc.calqmag(179, energy_kev): q_max = fc.calqmag(179, energy_kev)
        q_range = q_max - q_min

        # create plotting mesh
        tot_pixels = int(pixels * q_range)  # reduce this to make convolution faster
        pixel_size = q_range / float(tot_pixels)  # units of A-1
        peak_width_pixels = int(np.round(peak_width / pixel_size))  # peak_width is in A-1
        mesh = np.zeros([tot_pixels])
        mesh_q = np.linspace(q_min, q_max, tot_pixels)

        # Get reflections
        HKL = self.xtl.Cell.all_hkl(maxq=q_max + pixel_size)
        HKL = self.xtl.Cell.sort_hkl(HKL)  # required for labels
        Qmag = self.xtl.Cell.Qmag(HKL)

        # remove reflections not within the calculation
        pixel_coord = np.round(tot_pixels * (Qmag - q_min) / q_range).astype(int)
        select = (pixel_coord < tot_pixels) * (pixel_coord >= 0)
        HKL = HKL[select, :]
        Qmag = Qmag[select]
        pixel_coord = pixel_coord[select]

        # Calculate intensities
        I = self.intensity(HKL, scattering_type, **options)

        if powder_average:
            # Apply powder averging correction, I0/|Q|**2
            I = I/(Qmag+0.001)**2

        for n in range(len(I)):
            mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + I[n]

        # Convolve with a function (if > 0)
        if custom_peak is not None:
            mesh = np.convolve(mesh, custom_peak, mode='same')
        elif peak_width > 0:
            # peak_x = np.arange(-3*peak_width_pixels, 3*peak_width_pixels + 1)  # gaussian width = 2*FWHM
            peak_x = np.linspace(-tot_pixels/2, tot_pixels/2, tot_pixels)
            peak_shape = fg.pvoight(peak_x, height=1, centre=0, fwhm=peak_width_pixels, l_fraction=lorentz_fraction)
            mesh = np.convolve(mesh, peak_shape, mode='same')

        # Add background (if >0 or not None)
        if background:
            bkg = np.random.normal(background, np.sqrt(background), [tot_pixels])
            mesh = mesh + bkg

        # Change output units
        xval = fc.q2units(mesh_q, units, energy_kev)

        # Determine non-overlapping hkl coordinates
        xvalues = fc.q2units(Qmag, units, energy_kev)
        ref_n = fc.group_intensities(xvalues, I, min_overlap)

        grp_hkl = HKL[ref_n, :]
        grp_xval = xvalues[ref_n]
        grp_inten = mesh[pixel_coord[ref_n]]
        reflections = np.transpose([grp_hkl[:, 0], grp_hkl[:, 1], grp_hkl[:, 2], grp_xval, grp_inten])

        # Remove extended part
        ixmin = np.nanargmin(np.abs(xval - fc.q2units(fc.calqmag(min_twotheta, energy_kev), units, energy_kev)))
        ixmax = np.nanargmin(np.abs(xval - fc.q2units(fc.calqmag(max_twotheta, energy_kev), units, energy_kev)))
        # mesh = mesh[ext * peak_width_pixels:-ext * peak_width_pixels - 1]
        # xval = xval[ext * peak_width_pixels:-ext * peak_width_pixels - 1]
        mesh = mesh[ixmin:ixmax]
        xval = xval[ixmin:ixmax]
        return xval, mesh, reflections

    def generate_intensity_cut(self, x_axis=(1, 0, 0), y_axis=(0, 1, 0), centre=(0, 0, 0),
                               q_max=4.0, cut_width=0.05, background=0.0, peak_width=0.05, pixels=1001):
        """
        Generate a cut through reciprocal space, returns an array with centred reflections
        Inputs:
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
          background = average background value
          peak_width = reflection width in A-1
          pixels = size of the plotting mesh
        Returns:
          Qx/Qy = [pixels x pixels] array of coordinates
          plane = [pixels x pixels] array of plane in reciprocal space

        E.G. hk plane at L=3 for hexagonal system:
            Qx,Qy,plane = xtl.Scatter.generate_intensity_cut([1,0,0],[0,1,0],[0,0,3])
            plt.figure()
            plt.pcolormesh(Qx,Qy,plane)
            plt.axis('image')
        """
        qx, qy, hkl = self.xtl.Cell.reciprocal_space_plane(x_axis, y_axis, centre, q_max, cut_width)

        # Calculate intensities
        inten = self.xtl.Scatter.intensity(hkl)

        # create plotting mesh
        pixel_size = (2.0 * q_max) / pixels
        mesh = np.zeros([pixels, pixels])
        mesh_x = np.linspace(-q_max, q_max, pixels)
        xx, yy = np.meshgrid(mesh_x, mesh_x)

        if peak_width is None or peak_width < pixel_size:
            peak_width = pixel_size / 2

        # Add Gaussian profile to each peak
        KS = 3  # kernel size in units of peak width
        kernel_size = int(2 * KS * peak_width * pixels / (2 * q_max))
        kernel_size = kernel_size + 1 if kernel_size % 2 == 1 else kernel_size  # kernel_size must be even
        hks = kernel_size // 2
        kernel_x = np.linspace(-KS * peak_width, KS * peak_width, kernel_size)
        kxx, kyy = np.meshgrid(kernel_x, kernel_x)
        kernel = np.exp(-np.log(2) * ((kxx ** 2 + kyy ** 2) / (peak_width / 2) ** 2))
        for n in range(len(inten)):
            ix = np.nanargmin(np.abs(mesh_x - qy[n]))  # I need to switch qx,qy here for some reason
            iy = np.nanargmin(np.abs(mesh_x - qx[n]))  # must be a flip somewhere
            ix_min = 0 if ix < hks else ix - hks
            ix_max = pixels if ix > (pixels - hks) else ix + hks
            iy_min = 0 if iy < hks else iy - hks
            iy_max = pixels if iy > (pixels - hks) else iy + hks
            ikx_min = -(ix - hks) if ix < hks else 0
            ikx_max = hks + pixels - ix if ix > (pixels - hks) else kernel_size
            iky_min = -(iy - hks) if iy < hks else 0
            iky_max = hks + pixels - iy if iy > (pixels - hks) else kernel_size
            mesh[ix_min:ix_max, iy_min:iy_max] += inten[n] * kernel[ikx_min:ikx_max, iky_min:iky_max]

        # Add background (if not None or 0)
        if background:
            bkg = np.random.normal(background, np.sqrt(background), [pixels, pixels])
            mesh = mesh + bkg
        return xx, yy, mesh

    def generate_intensity_cut_old(self, x_axis=(1, 0, 0), y_axis=(0, 1, 0), centre=(0, 0, 0),
                                   q_max=4.0, cut_width=0.05, background=0.0, peak_width=0.05, pixels=1001):
        """
        Generate a cut through reciprocal space, returns an array with centred reflections
          **Old version - creates full Gaussian for each peak, very slow**
        Inputs:
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          cut_width = width in height that will be included, in A-1
          background = average background value
          peak_width = reflection width in A-1
        Returns:
          Qx/Qy = [1000x1000] array of coordinates
          plane = [1000x1000] array of plane in reciprocal space

        E.G. hk plane at L=3 for hexagonal system:
            Qx,Qy,plane = xtl.generate_intensity_cut([1,0,0],[0,1,0],[0,0,3])
            plt.figure()
            plt.pcolormesh(Qx,Qy,plane)
            plt.axis('image')
        """
        qx, qy, hkl = self.xtl.Cell.reciprocal_space_plane(x_axis, y_axis, centre, q_max, cut_width)

        # Calculate intensities
        inten = self.xtl.Scatter.intensity(hkl)

        # create plotting mesh
        pixel_size = (2.0 * q_max) / pixels
        mesh = np.zeros([pixels, pixels])
        mesh_x = np.linspace(-q_max, q_max, pixels)
        xx, yy = np.meshgrid(mesh_x, mesh_x)

        if peak_width is None or peak_width < pixel_size:
            peak_width = pixel_size / 2

        start_time = datetime.datetime.now()
        print('Convolving with 2D Gaussian: |' + 50 * '-' + '|', end='\r', flush=True)
        for n in range(len(inten)):
            # Add each reflection as a gaussian - this is very slow!
            mesh += inten[n] * np.exp(-np.log(2) * (((xx - qx[n]) ** 2 + (yy - qy[n]) ** 2) / (peak_width / 2) ** 2))
            if n % (len(inten)/50) < 1:
                _done = int(n // (len(inten)/50))
                print('Convolving with 2D Gaussian: |' + (_done * '█') + (50-_done) * '-' + '|', end='\r', flush=True)
        print('Convolving with 2D Gaussian: |' + 50 * '█' + '|', end='\n', flush=True)
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time
        print('Time taken for %d reflections: %s' % (len(inten), time_difference))

        # Add background (if not None or 0)
        if background:
            bkg = np.random.normal(background, np.sqrt(background), [pixels, pixels])
            mesh = mesh + bkg
        return xx, yy, mesh

    def generate_envelope_cut(self, x_axis=(1, 0, 0), y_axis=(0, 1, 0), centre=(0, 0, 0),
                              q_max=4.0, background=0.0, pixels=301):
        """
        *In Development*
        Generate the envelope function, calculating the structure factor at discrete points
        Inputs:
          x_axis = direction along x, in units of the reciprocal lattice (hkl)
          y_axis = direction along y, in units of the reciprocal lattice (hkl)
          centre = centre of the plot, in units of the reciprocal lattice (hkl)
          q_max = maximum distance to plot to - in A-1
          background = average background value
          pixels = size of mesh, calculates structure factor at each pixel
        Returns:
          Qx/Qy = [pixels x pixels] array of coordinates
          plane = [pixels x pixels] array of plane in reciprocal space

        E.G. hk plane at L=3 for hexagonal system:
            Qx, Qy, plane = xtl.generate_envelope_cut([1,0,0],[0,1,0],[0,0,3])
            plt.figure()
            plt.pcolormesh(Qx, Qy, plane)
            plt.axis('image')
        """
        x_cart = self.xtl.Cell.calculateQ(x_axis)
        y_cart = self.xtl.Cell.calculateQ(y_axis)
        x_cart, y_cart, z_cart = fc.orthogonal_axes(x_cart, y_cart)
        c_cart = self.xtl.Cell.calculateQ(centre)

        xx, yy = np.meshgrid(range(pixels), range(pixels))
        q_range = np.linspace(-q_max, q_max, pixels)
        qx, qy = q_range[xx], q_range[yy]  # [pixels, pixels]
        qxqy = np.transpose([qx.reshape(-1), qy.reshape(-1)])  # returns [qx, qy]
        qxqyz = np.dot(qxqy, [x_cart, y_cart])  # returns [qx,qy,qz]
        hkl = self.xtl.Cell.indexQ(qxqyz + c_cart)

        # Calculate intensities
        inten = self.intensity(hkl, int_hkl=False)
        mesh = inten.reshape([pixels, pixels])

        # Add background (if not None or 0)
        if background:
            bkg = np.random.normal(background, np.sqrt(background), [pixels, pixels])
            mesh = mesh + bkg
        return qx, qy, mesh

    def x_ray(self, HKL):
        """
        Calculate the squared structure factor for the given HKL, using x-ray scattering factors
          Scattering.x_ray([1,0,0])
          Scattering.x_ray([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        Nref = len(HKL)
        
        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get atomic form factors
        if self._use_waaskirf_scattering_factor:
            ff = fc.xray_scattering_factor_WaasKirf(atom_type, Qmag)
        else:
            ff = fc.xray_scattering_factor(atom_type, Qmag)
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso, Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calcualtion not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)
        
        # Calculate structure factor
        # Broadcasting used on 2D ff
        SF = np.sum(ff*dw*occ*np.exp(1j*2*np.pi*dot_KR), axis=1)
        #SF = np.zeros(Nref,dtype=complex)
        #for ref in range(Nref):
        #    for at in range(Nat): 
        #        SF[ref] += ff[ref,at]*dw[ref,at]*occ[at]*np.exp(1j*2*np.pi*dot_KR[ref,at])
        
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def x_ray_fast(self, HKL):
        """
        Calculate the squared structure factor for the given HKL, using atomic number as scattering length
          Scattering.x_ray_fast([1,0,0])
          Scattering.x_ray_fast([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        
        uvw, atomtype, label, occ, uiso, mxmymz = self.xtl.Structure.get()
        
        # Get atomic form factors
        ff = fc.atom_properties(atomtype, 'Z')
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            Qmag = self.xtl.Cell.Qmag(HKL)
            dw = fc.debyewaller(uiso,Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        # Calculate structure factor
        SF = np.sum(ff * dw * occ * np.exp(1j * 2 * np.pi * dot_KR), axis=1)
        
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)

    def xray_dispersion(self, HKL, energy_kev):
        """
        Calculate the squared structure factor for the given HKL,
        using x-ray scattering factors with dispersion corrections
          Scattering.xray_dispersion([1,0,0], 9.71)
          Scattering.xray_dispersion([[1,0,0],[2,0,0],[3,0,0]], 2.838)
          Scattering.xray_dispersion([[1,0,0],[2,0,0],[3,0,0]], np.arange(2.83, 2.86, 0.001))
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        if energy_kev is an array, the returned array will have shape [len(HKL), len(energy_kev)]
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()

        Qmag = self.xtl.Cell.Qmag(HKL)

        # Get atomic form factors
        ff = fc.xray_scattering_factor_resonant(atom_type, Qmag, energy_kev)  # shape (len(HKL), len(type), len(en))

        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso, Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calcualtion not implemented yet')
        else:
            dw = 1

        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)

        energy_kev = np.asarray(energy_kev).reshape(-1)
        sf = np.zeros([len(HKL), len(energy_kev)], dtype=complex)
        for n, en in enumerate(energy_kev):
            # Broadcasting used on 2D ff
            sf[:, n] = np.sum(ff[:, :, n] * dw * occ * np.exp(1j * 2 * np.pi * dot_KR), axis=1)
        if len(energy_kev) == 1:
            sf = sf[:, 0]
        elif len(HKL) == 1:
            sf = sf[0, :]

        sf = sf / self.xtl.scale

        if self._return_structure_factor: return sf

        # Calculate intensity
        I = sf * np.conj(sf)
        return np.real(I)

    def electron(self, HKL):
        """
        Calculate the squared structure factor for the given HKL, using electron form factors
          Scattering.electron([1,0,0])
          Scattering.electron([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """
        return self.intensity(HKL, scattering_type='electron')

    def neutron(self, HKL):
        """
        Calculate the squared structure factor for the given HKL, using neutron scattering length
          Scattering.neutron([1,0,0])
          Scattering.neutron([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()
        
        # Get atomic form factors
        ff = fc.atom_properties(atom_type, 'Coh_b')
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            Qmag = self.xtl.Cell.Qmag(HKL)
            dw = fc.debyewaller(uiso,Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)
        
        # Calculate structure factor
        SF = np.sum(ff * dw * occ * np.exp(1j * 2 * np.pi * dot_KR), axis=1)
        
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def magnetic_neutron(self, HKL):
        """
        Calculate the magnetic component of the structure factor for the given HKL, using neutron rules and form factor
          Scattering.magnetic_neutron([1,0,0])
          Scattering.magnetic_neutron([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        Nref = len(HKL)

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)
        
        Q = self.xtl.Cell.calculateQ(HKL)
        Qhat = fg.norm(Q).reshape([-1,3])
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get magnetic form factors
        if self._use_magnetic_form_factor:
            ff = fc.magnetic_form_factor(*atom_type, qmag=Qmag)
        else:
            ff = np.ones([len(HKL), Nat])
        
        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1,1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)
        moment = momentmag*fg.norm(momentxyz) # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)

        # Calculate structure factor
        SF = np.zeros(Nref,dtype=complex)
        for n,Qh in enumerate(Qhat):
            SFm = [0.,0.,0.]
            for m,mom in enumerate(moment):
                # Calculate Magnetic part
                QM = mom - np.dot(Qh,mom)*Qh

                # Calculate structure factor
                SFm = SFm + ff[n, m] * np.exp(1j * 2 * np.pi * dot_KR[n, m]) * QM
            
            # Calculate polarisation with incident neutron
            if self._polarised:
                SF[n] = np.dot(SFm, self._polarisation_vector_incident)
            else:
                #SF[n] = np.dot(SFm,SFm) # maximum possible
                SF[n] = (np.dot(SFm,[1,0,0]) + np.dot(SFm,[0,1,0]) + np.dot(SFm,[0,0,1]))/3 # average polarisation

        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)

    def xray_magnetic(self, HKL):
        """
        Calculate the non-resonant magnetic component of the structure factor 
        for the given HKL, using x-ray rules and form factor
          Scattering.xray_magnetic([1,0,0])
          Scattering.xray_magnetic([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        
        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
        Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
        No orbital component assumed
        magnetic moments assumed to be in the same reference frame as the polarisation
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        Nref = len(HKL)

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()
        Nat = len(uvw)

        Qmag = self.xtl.Cell.Qmag(HKL)

        # Get magnetic form factors
        if self._use_magnetic_form_factor:
            ff = fc.magnetic_form_factor(*atom_type, qmag=Qmag)
        else:
            ff = np.ones([len(HKL), Nat])

        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1, 1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)  # moment direction in cartesian reference frame
        moment = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.

        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)

        # Calculate structure factor
        SF = np.zeros(Nref, dtype=complex)
        for n in range(Nref):
            # Calculate vector structure factor
            SFm = [0., 0., 0.]
            for m, mom in enumerate(moment):
                SFm = SFm + ff[n, m] * np.exp(1j * 2 * np.pi * dot_KR[n, m]) * mom

            # Calculate polarisation with incident x-ray
            # The reference frame of the x-ray and the crystal are assumed to be the same
            # i.e. pol=[1,0,0] || mom=[1,0,0] || (1,0,0)
            if self._polarised:
                SF[n] = np.dot(SFm, self._polarisation_vector_incident)
            else:
                # SF[n] = np.dot(SFm,SFm) # maximum possible
                SF[n] = (
                                np.dot(SFm, [1, 0, 0]) + np.dot(SFm, [0, 1, 0]) + np.dot(SFm, [0, 0, 1])
                        ) / 3  # average polarisation

        SF = SF / self.xtl.scale

        if self._return_structure_factor: return SF

        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def xray_resonant(self, HKL, energy_kev=None, polarisation='sp', F0=1, F1=1, F2=1,
                      azim_zero=(1, 0, 0), PSI=[0], disp=False):
        """
        Calculate structure factors using resonant scattering factors in the dipolar approximation
          I = Scattering.xray_resonant(HKL,energy_kev,polarisation,F0,F1,F2)
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
            energy_kev = x-ray energy in keV
            polarisation = x-ray polarisation: 'ss',{'sp'},'ps','pp'
            F0/1/2 = Resonance factor Flm
            azim_zero = [h,k,l] vector parallel to the origin of the azimuth
            psi = azimuthal angle defining the scattering plane
        
        Uses the E1E1 resonant x-ray scattering amplitude:
            fxr_n = (ef.ei)*F0 -i(ef X ei).z_n*F1 + (ef.z_n)(ei.z_n)F2
        
        Where ei and ef are the initial and final polarisation states, respectively,
        and z_n is a unit vector in the direction of the magnetic moment of the nth ion.
        The polarisation states are determined to be one of the natural synchrotron 
        states, where sigma (s) is perpendicular to the scattering plane and pi (p) is
        parallel to it.
                ( s-s  s-p )
                ( p-s  p-p )
        
        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (15)
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        Nref = len(HKL)
        
        PSI = np.asarray(PSI,dtype=float).reshape([-1])
        Npsi = len(PSI)

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()
        
        Qmag = self.xtl.Cell.Qmag(HKL)
        
        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso,Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL,uvw.T)
        
        SF = np.zeros([Nref,Npsi],dtype=complex)
        for psival in range(Npsi):
            # Get resonant form factor
            fxres = self.xray_resonant_scattering_factor(HKL,energy_kev,polarisation,F0,F1,F2,azim_zero,PSI[psival],disp=disp)
            
            # Calculate structure factor
            # Broadcasting used on 2D fxres
            SF[:, psival] = np.sum(fxres*dw*occ*np.exp(1j*2*np.pi*dot_KR), axis=1)
            
        SF = SF/self.xtl.scale
        
        if self._return_structure_factor: return SF
        
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    def xray_resonant_scattering_factor(self, HKL, energy_kev=None, polarisation='sp', F0=1, F1=1, F2=1,
                                        azim_zero=(1, 0, 0), psi=0, disp=False):
        """
        Calcualte fxres, the resonant x-ray scattering factor
          fxres = Scattering.xray_resonant_scattering_factor(HKL,energy_kev,polarisation,F0,F1,F2,azim_zero,psi)
        energy_kev = x-ray energy in keV
            polarisation = x-ray polarisation: 'ss',{'sp'},'ps','pp'
            F0/1/2 = Resonance factor Flm
            azim_zero = [h,k,l] vector parallel to the origin of the azimuth {[1,0,0]}
            psi = azimuthal angle defining the scattering plane {0}
        
        Uses the E1E1 resonant x-ray scattering amplitude:
            fxr_n = (ef.ei)*F0 -i(ef X ei).z_n*F1 + (ef.z_n)(ei.z_n)F2
        
        Where ei and ef are the initial and final polarisation states, respectively,
        and z_n is a unit vector in the direction of the magnetic moment of the nth ion.
        The polarisation states are determined to be one of the natural synchrotron 
        states, where sigma (s) is perpendicular to the scattering plane and pi (p) is
        parallel to it.
                ( s-s  s-p )
                ( p-s  p-p )
        
        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (15)
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        Nref = len(HKL)
        Qmag = self.xtl.Cell.Qmag(HKL)
        tth = fc.cal2theta(Qmag, energy_kev)
        
        mxmymz = self.xtl.Structure.mxmymz()
        Nat = len(mxmymz)
        
        fxres = np.zeros([Nref,Nat],dtype=complex)
        for ref in range(Nref):
            # Resonant scattering factor
            # Electric Dipole transition at 3d L edge
            #F0,F1,F2 = 1,1,1 # Flm (form factor?)
            z1, z2, z3 = self.scatteringcomponents(mxmymz, HKL[ref], azim_zero, psi).T
            tthr = np.deg2rad(tth[ref])/2.0

            polarisation = polarisation.replace('-', '').replace(' ', '')
            if polarisation in ['sigmasigma', 'sigsig', 'ss']:    # Sigma-Sigma
                f0 = 1*np.ones(Nat)
                f1 = 0*np.ones(Nat)
                f2 = z2**2
            elif polarisation in ['sigmapi', 'sigpi', 'sp']:  # Sigma-Pi
                f0 = 0 * np.ones(Nat)
                f1 = z3 * np.sin(tthr) + z1 * np.cos(tthr)
                f2 = z2 * (z1 * np.sin(tthr) + z3 * np.cos(tthr))
            elif polarisation in ['pisigma', 'pisig', 'ps']:  # Pi-Sigma
                f0 = 0*np.ones(Nat)
                f1 = z1*np.cos(tthr) - z3*np.sin(tthr)
                f2 = -z2*(z1*np.sin(tthr)-z3*np.cos(tthr))
            elif polarisation in ['pipi', 'pp']:  # Pi-Pi
                f0 = np.cos(2*tthr)*np.ones(Nat)
                f1 = -z2*np.sin(2*tthr)
                f2 = -(np.cos(tthr)**2)*(z1**2*np.tan(tthr)**2 + z3**2)
            else:
                raise ValueError('Incorrect polarisation. pol should be e.g. ''ss'' or ''sp''')
            fxres[ref,:] = F0*f0 -1j*F1*f1 + F2*f2
            if disp:
                print('( h, k, l)   TTH  (    mx,    my,    mz)  (    z1,    z2,    z3)')
                fmt = '(%2.0f,%2.0f,%2.0f) %6.2f  (%6.3f,%6.3f,%6.3f)  (%6.3f,%6.3f,%6.3f)  f0=%8.4f  f1=%8.4f  f2=%8.4f fxres= (%8.4f + %8.4fi)'
                for at in range(Nat):
                    vals = (HKL[ref,0],HKL[ref,1],HKL[ref,2],tth[ref],
                            mxmymz[at,0],mxmymz[at,1],mxmymz[at,2],
                            z1[at],z2[at],z3[at],
                            f0[at],f1[at],f2[at],
                            fxres[ref,at].real,fxres[ref,at].imag)
                    print(fmt%vals)
        return fxres

    def xray_nonresonant_magnetic(self, HKL, energy_kev=None, azim_zero=(1, 0, 0), psi=0, polarisation='s-p', disp=False):
        """
        Calculate the non-resonant magnetic component of the structure factor
        for the given HKL, using x-ray rules and form factor
          Scattering.xray_magnetic([1,0,0])
          Scattering.xray_magnetic([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.

        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
        Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
        No orbital component assumed
        magnetic moments assumed to be in the same reference frame as the polarisation
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()

        kin, kout, ein, eout = self.scatteringvectors(HKL, energy_kev, azim_zero, psi, polarisation)

        Qmag = self.xtl.Cell.Qmag(HKL)

        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso, Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1

        # Get magnetic form factors
        if self._use_magnetic_form_factor:
            ff = fc.magnetic_form_factor(*atom_type, qmag=Qmag)
        else:
            ff = np.ones([len(HKL), len(uvw)])

        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1, 1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)  # moment direction in cartesian reference frame
        moment = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.

        # Magnetic form factor
        # f_non-res_mag = i.r0.(hw/mc^2).fD.[.5.L.A + S.B] #equ 2 Hill+McMorrow 1996
        # ignore orbital moment L
        B = np.zeros([len(HKL), 3])
        for n in range(len(HKL)):
            #print(n,HKL[n],kin[n],kout[n],ein[n],eout[n])
            B[n, :] = np.cross(eout[n], ein[n]) + \
                np.cross(kout[n], eout[n]) * np.dot(kout[n], ein[n]) - \
                np.cross(kin[n], ein[n]) * np.dot(kin[n], eout[n]) - \
                np.cross(np.cross(kout[n], eout[n]), np.cross(kin[n], ein[n]))
        fspin = 1j * ff * np.dot(moment, B.T).T

        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)

        # Calculate structure factor
        SF = np.sum(fspin * dw * occ * np.exp(1j * 2 * np.pi * dot_KR), axis=1)

        if disp:
            for n in range(len(HKL)):
                pr_h = '(%2.0f,%2.0f,%2.0f)'%tuple(HKL[n,:])
                pr_b = '(%6.2g,%6.2g,%6.2g)'%tuple(B[n,:])
                print('psi=%3.0f  %3d hkl=%10s B=%22s'%(psi, n, pr_h, pr_b))
                ctot = 0j
                for m in range(len(moment)):
                    if np.sum(moment[m,:]**2) < 0.01: continue
                    pr_m = '(%6.2g,%6.2g,%6.2g)'%tuple(moment[m,:])
                    dot = np.dot(moment[m],B[n,:])
                    pdot = '%6.3f'%dot
                    phase = np.exp(1j * 2 * np.pi * dot_KR[n, m])
                    prph = '%5.2f+i%5.2f'%(np.real(phase),np.imag(phase))
                    prod = dot*phase
                    pprd = '%5.2f+i%5.2f'%(np.real(prod),np.imag(prod))
                    ctot += prod
                    ptot = '%5.2f+i%5.2f'%(np.real(ctot),np.imag(ctot))
                    print('\t%3d mom=%22s dot(mom,B)=%6s   exp(ik.r)=%12s   sum=%12s   tot=%12s'%(m,pr_m,pdot,prph,pprd,ptot))

        SF = SF / self.xtl.scale

        if self._return_structure_factor: return SF

        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)

    def xray_resonant_magnetic(self, HKL, energy_kev=None, azim_zero=(1, 0, 0), psi=0, polarisation='s-p', F0=0, F1=1, F2=0, disp=True):
        """
        Calculate the resonant magnetic component of the structure factor
        for the given HKL, using x-ray rules and form factor
          Scattering.xray_magnetic([1,0,0])
          Scattering.xray_magnetic([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.

        From Hill+McMorrow Acta Cryst. 1996 A52, 236-244 Equ. (2)
        Book: "X-ray Scattering and Absorption by Magnetic Materials" by Loevesy and Collins. Ch 2. Eqn.2.21+1
        No orbital component assumed
        magnetic moments assumed to be in the same reference frame as the polarisation
        """

        if self._integer_hkl:
            HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        else:
            HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])

        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()

        kin, kout, ein, eout = self.scatteringvectors(HKL, energy_kev, azim_zero, psi, polarisation)

        Qmag = self.xtl.Cell.Qmag(HKL)

        # Get Debye-Waller factor
        if self._use_isotropic_thermal_factor:
            dw = fc.debyewaller(uiso, Qmag)
        elif self._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calculation not implemented yet')
        else:
            dw = 1

        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1, 1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)  # moment direction in cartesian reference frame
        moment = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.
        fe1e1 = np.zeros([len(HKL), len(uvw)], dtype=complex)
        for ref in range(len(HKL)):
            # Magnetic form factor
            # f_res_mag = [(e'.e)F0 - i(e'xe).Z*F1 + (e'.Z)*(e.Z)*F2] #equ 7 Hill+McMorrow 1996
            f0 = np.dot(eout[ref], ein[ref])
            f1 = np.dot(np.cross(eout[ref], ein[ref]), moment.T)
            f2 = np.dot(eout[ref], moment.T) * np.dot(ein[ref], moment.T)
            fe1e1[ref, :] = f0*F0 - 1j*f1*F1 + f2*F2

        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)

        if disp:
            for ref in range(len(HKL)):
                pr_h = '(%2.0f,%2.0f,%2.0f)'%tuple(HKL[ref,:])
                print('psi=%3.0f  %3d hkl=%10s'%(psi, ref, pr_h))
                ctot = 0j
                for m in range(len(moment)):
                    if np.sum(moment[m,:]**2) < 0.01: continue
                    pr_m = '(%6.2g,%6.2g,%6.2g)'%tuple(moment[m,:])
                    f0 = np.dot(eout[ref], ein[ref])
                    f1 = np.dot(np.cross(eout[ref], ein[ref]), moment[m,:])
                    f2 = np.dot(eout[ref], moment[m,:]) * np.dot(ein[ref], moment[m,:])
                    dot = f0*F0 - 1j*f1*F1 + f2*F2
                    phase = np.exp(1j * 2 * np.pi * dot_KR[ref, m])
                    prph = '%5.2f+i%5.2f'%(np.real(phase),np.imag(phase))
                    prod = dot*phase
                    pprd = '%5.2f+i%5.2f'%(np.real(prod),np.imag(prod))
                    ctot += prod
                    ptot = '%5.2f+i%5.2f'%(np.real(ctot),np.imag(ctot))
                    print('\t%3d mom=%22s f0=%5.2f f1=%5.2f f2=%5.2f   exp(ik.r)=%12s   sum=%12s   tot=%12s'%(m,pr_m,f0,f1,f2,prph,pprd,ptot))

        # Calculate structure factor
        SF = np.sum(fe1e1 * dw * occ * np.exp(1j * 2 * np.pi * dot_KR), axis=1)

        SF = SF / self.xtl.scale

        if self._return_structure_factor: return SF

        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)

    def scatteringvectors(self, hkl, energy_kev=None, azim_zero=(1, 0, 0), psi=0, polarisation='s-p'):
        """
        Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
        :param xtl: Crystal Class
        :param hkl: [n,3] array of reflections
        :param energy_kev: x-ray scattering energy in keV
        :param azim_zero: [1,3] direction along which the azimuthal zero angle is determind
        :param psi: float angle in degrees about the azimuth
        :param polarisation: polarisation with respect to the scattering plane, options:
                    'ss' : sigma-sigma polarisation
                    'sp' : sigma-pi polarisation
                    'ps' : pi-sigma polarisation
                    'pp' : pi-pi polarisation
                or: polarisation: float polarisation angle of scattered vector in degrees
        :return: kin, kout, ein, eout
        Returned values are [n,3] arrays
            kin : [n,3] array of incident wavevectors
            kout: [n,3] array of scattered wavevectors
            ein : [n,3] array of incident polarisation
            eout: [n,3] array of scattered polarisation

        The basis is chosen such that Q defines the scattering plane, sigma and pi directions are normal to this plane.
        Q is defined as Q = kout - kin, with kout +ve along the projection of azim_zero
        """

        hkl = np.asarray(hkl).reshape([-1, 3])

        if energy_kev is None:
            energy_kev = self._energy_kev

        out_kin = np.zeros([len(hkl), 3])
        out_kout = np.zeros([len(hkl), 3])
        out_ein = np.zeros([len(hkl), 3])
        out_eout = np.zeros([len(hkl), 3])
        for n in range(len(hkl)):
            # Define coordinate system I,J,Q (U1,U2,U3)
            Ihat, Jhat, Qhat = self.scatteringbasis(hkl[n, :], azim_zero, psi)
            Ihat = Ihat.reshape([-1, 3])
            Jhat = Jhat.reshape([-1, 3])
            Qhat = Qhat.reshape([-1, 3])

            # Determine wavevectors
            bragg = self.xtl.Cell.tth(hkl[n, :], energy_kev) / 2.
            rb = np.deg2rad(bragg).reshape([-1, 1])
            kin = np.cos(rb) * Ihat - np.sin(rb) * Qhat
            kout = np.cos(rb) * Ihat + np.sin(rb) * Qhat
            esig = Jhat  # sigma polarisation (in or out)
            piin = np.cross(kin, esig)  # pi polarisation in
            piout = np.cross(kout, esig)  # pi polarisation out

            # Polarisation
            try:
                # polarisation = 'ss' or 's-s'
                polarisation = polarisation.replace('-', '').replace(' ', '')
            except AttributeError:
                # polarisation = angle in deg from sigma' to pi'
                ein = 1.0 * esig
                pol = np.deg2rad(polarisation)
                eout = np.cos(pol)*esig + np.sin(pol)*piout
            if polarisation in ['sigmasigma', 'sigsig', 'ss']:
                ein = 1.0 * esig
                eout = 1.0 * esig
            elif polarisation in ['sigmapi', 'sigpi', 'sp']:
                ein = 1.0 * esig
                eout = 1.0 * piout
            elif polarisation in ['pisigma', 'pisig', 'ps']:
                ein = 1.0 * piin
                eout = 1.0 * esig
            elif polarisation in ['pipi', 'pp']:
                ein = 1.0 * piin
                eout = 1.0 * piout
            out_kin[n, :] = kin
            out_kout[n, :] = kout
            out_ein[n, :] = ein
            out_eout[n, :] = eout

        return out_kin, out_kout, out_ein, out_eout

    def scatteringcomponents(self, mxmymz, hkl, azim_zero=(1, 0, 0), psi=0):
        """
        Transform magnetic vector into components within the scattering plane
        First transforms the moments into a cartesian reference frame.
           U1 = direction || ki - kf = Q
           U2 = direction || kf + ki (within scattering plane, pi)
           U3 = direction perp. to U1, U2 (normal to scattering plane, sigma)
        """

        # Define coordinate system I,J,Q (U1,U2,U3)
        U = self.scatteringbasis(hkl, azim_zero, psi)

        # Calculate moment
        momentmag = fg.mag(mxmymz).reshape([-1, 1])
        momentxyz = self.xtl.Cell.calculateR(mxmymz)  # moment direction in cartesian reference frame
        moment = momentmag * fg.norm(momentxyz)  # broadcast n*1 x n*3 = n*3
        moment[np.isnan(moment)] = 0.
        
        # Determine components of the magnetic vector
        z1z2z3 = np.dot(moment, U.T)  # [mxmymz.I, mxmymz.J, mxmymz.Q]
        return fg.norm(z1z2z3)

    def scatteringbasis(self, hkl, azim_zero=(1, 0, 0), psi=0):
        """
        Determine the scattering and polarisation vectors of a reflection based on energy, azimuth and polarisation.
        :param hkl: [n,3] array of reflections
        :param azim_zero: [1,3] direction along which the azimuthal zero angle is determind
        :param psi: float azimuthal angle about U3 in degrees
        :return: U1, U2, U3
        The basis is chosen such that Q defines the scattering plane, the sigma direction is normal to this plane,
        the pi direction is always within this plane.
        The azimuthal angle defines a rotation about the Q axis in a clockwise mannor, matching I16.
        At an azimuth of 0degrees, U1 is perpendicular to Q, along the direction of azim_zero.
        """

        # Define coordinate system I,J,Q (U1,U2,U3)
        # See FDMNES User's Guide p20 'II-11) Anomalous or resonant diffraction'
        # U1 || projection of azim_zero
        # U2 _|_ U1,U3
        # U3 || Q = kf-ki
        azim_zero = fg.norm(self.xtl.Cell.calculateQ(azim_zero)) # put in orthogonal basis
        Qhat = fg.norm(self.xtl.Cell.calculateQ(hkl)).reshape([-1,3])  # || Q
        AxQ = fg.norm(np.cross(azim_zero, Qhat))
        Ihat = fg.norm(np.cross(Qhat, AxQ)).reshape([-1,3])  # || to projection of azim_zero
        Jhat = fg.norm(np.cross(Qhat, Ihat)).reshape([-1,3])  # _|_ to I and Q

        # Rotate psi about Qhat
        rpsi = np.deg2rad(psi)
        # -ve sin makes clockwise rotation
        # This was checked on 21/1/19 vs CRO paper + sergio's calculations and seems to agree with experiment,
        # however we never did an azimuthal scan of the (103) which would have distinguished this completely.
        Ihat_psi = fg.norm(np.cos(rpsi) * Ihat - np.sin(rpsi) * Jhat)
        Jhat_psi = fg.norm(np.cross(Qhat, Ihat_psi))
        return np.vstack([Ihat_psi, Jhat_psi, Qhat])

    def scattering_factors(self, hkl=(0, 0, 0), energy_kev=None, qmag=None):
        """Return the scattering factors[n, m] for each reflection, hkl[n,3] and each atom [m]"""

        if energy_kev is None:
            energy_kev = self._energy_kev

        if qmag is None:
            qmag = self.xtl.Cell.Qmag(hkl)
        # Scattering factors
        ff = fs.scattering_factors(
            scattering_type=self._scattering_type,
            atom_type=self.xtl.Structure.type,
            qmag=qmag,
            enval=energy_kev,
            use_sears=self._use_sears_scattering_lengths,
            use_wasskirf=self._use_waaskirf_scattering_factor,
        )
        return ff

    def print_scattering_coordinates(self, hkl, azim_zero=(1, 0, 0), psi=0):
        """
        Transform magnetic vector into components within the scattering plane
            ***warning - may not be correct for non-cubic systems***
        """
        
        # Define coordinate system I,J,Q (U1,U2,U3)
        Qhat = fg.norm(self.xtl.Cell.calculateQ(hkl)) # || Q
        AxQ = fg.norm(np.cross(azim_zero,Qhat))
        Ihat = fg.norm(np.cross(Qhat,AxQ)) # || to azim_zero
        Jhat = fg.norm(np.cross(Qhat,Ihat)) # -| to I and Q
        
        # Rotate coordinate system by azimuth
        Ihat_psi = fg.norm(np.cos(np.deg2rad(psi))*Ihat + np.sin(np.deg2rad(psi))*Jhat)
        Jhat_psi = fg.norm(np.cross(Qhat,Ihat_psi))
        
        # Determine components of the magnetic vector
        U=np.vstack([Ihat_psi,Jhat_psi,Qhat])
        print('U1 = (%5.2f,%5.2f,%5.2f)'%(U[0,0],U[0,1],U[0,2]))
        print('U2 = (%5.2f,%5.2f,%5.2f)'%(U[1,0],U[1,1],U[1,2]))
        print('U3 = (%5.2f,%5.2f,%5.2f)'%(U[2,0],U[2,1],U[2,2]))
    
    def print_intensity(self, HKL):
        """
        Print intensities calcualted in different ways
        """
        
        HKL = np.asarray(np.rint(HKL),dtype=float).reshape([-1,3])
        Qmag =  self.xtl.Cell.Qmag(HKL)
        srt = np.argsort(Qmag)
        HKL = HKL[srt,:]
        
        IN=self.neutron(HKL)
        IX=self.x_ray(HKL)
        INM=self.magnetic_neutron(HKL)*1e4
        IXM=self.xray_magnetic(HKL)*1e4
        IXRss=self.xray_resonant(HKL, None, 'ss')
        IXRsp=self.xray_resonant(HKL, None, 'sp')
        IXRps=self.xray_resonant(HKL, None, 'ps')
        IXRpp=self.xray_resonant(HKL, None, 'pp')
        
        fmt = '(%2.0f,%2.0f,%2.0f)  %8.1f  %8.1f  %8.2f  %8.2f  ss=%8.2f  sp=%8.2f  ps=%8.2f  pp=%8.2f\n'
        outstr = '( h, k, l)   Neutron      xray   Magn. N  Magn. XR   sig-sig    sig-pi    pi-sig     pi-pi\n'
        for n in range(len(HKL)):
            vals=(HKL[n][0],HKL[n][1],HKL[n][2],IN[n],IX[n],INM[n],IXM[n],IXRss[n],IXRsp[n],IXRps[n],IXRpp[n])
            outstr += fmt % vals
        return outstr

    def print_atom_scattering_factors(self, hkl, energy_kev=None):
        """show scattering factors for each atom for each reflection"""
        qmag = self.xtl.Cell.Qmag(hkl)
        ff = self.scattering_factors(hkl, energy_kev)
        hkl = np.asarray(hkl, dtype=float).reshape([-1, 3])

        out = f"{self._scattering_type}\n"
        for n, (h, k, l) in enumerate(hkl):
            out += f"({h:.3g}, {k:.3g}, {l:.3g})  |Q| = {qmag[n]:.4g} A-1\n"
            for m, ele in enumerate(self.xtl.Structure.type):
                out += f"  {ele:5}: {ff[n, m]: 12.5f}\n"
        return out

    def print_scattering_factor_coefficients(self):
        """generate string of scattering factor coefficients for each atom"""
        scattering_type = fs.get_scattering_type(self._scattering_type)

        if 'neutron' in scattering_type:
            if self._use_sears_scattering_lengths:
                table = 'sears'
            else:
                table = 'ndb'
        elif 'electron' in scattering_type:
            table = 'peng'
        elif 'xray' in scattering_type:
            if self._use_waaskirf_scattering_factor:
                table = 'waaskirf'
            else:
                table = 'itc'
        else:
            raise Exception(f"Unknown scattering type: {scattering_type}")

        element_coefs = fc.scattering_factor_coefficients_custom(*self.xtl.Structure.type, default_table=table)


        structrue = self.xtl.Structure
        structure_list = zip(structrue.label, structrue.type, structrue.u, structrue.v, structrue.w)
        out = f"{scattering_type}\n"
        for n, (label, el, u, v, w) in enumerate(structure_list):
            out += f"{n:3}  {label:6} {el:5}: {u:8.3g}, {v:8.3g}, {w:8.3g}  : "

            coefs = element_coefs[el]
            if abs(coefs[-1]) < 1e-6:
                coefs = coefs[:-1]  # trim final column if all zeros
            n_doubles = len(coefs) // 2
            n_singles = len(coefs) % 2
            out += ', '.join(f"a{d} = {coefs[2*d]:.3f}, A{d} = {coefs[2*d+1]:.3f}" for d in range(n_doubles))
            out += (', ' * int(n_doubles > 0)) + (f"b = {coefs[-1]:.3f}" * n_singles)
            out += '\n'
        return out

    def old_intensity(self, HKL, scattering_type=None):
        """
        Calculate the squared structure factor for the given HKL
          Crystal.intensity([1,0,0])
          Crystal.intensity([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the real intensity at each reflection.
        
        Notes:
        - Uses x-ray atomic form factors, calculated from approximated tables in the ITC
        - This may be a little slow for large numbers of reflections, as it is not currently
         possible to use accelerated calculation methods in Jython.
        - Debye-Waller factor (atomic displacement) is applied for isotropic ADPs
        - Crystal.scale is used to scale the complex structure factor, so the intensity is  
         reduced by (Crystal.scale)^2
        - Testing against structure factors calculated by Vesta.exe is very close, though there
          are some discrepancies, probably due to the method of calculation of the form factor.
        """

        if scattering_type is None:
            scattering_type = self._scattering_type
        scattering_type = scattering_type.lower()

        # Break up long lists of HKLs
        n_arrays = np.ceil(len(HKL)*len(self.xtl.Structure.u)/10000.)
        hkl_array = np.array_split(HKL, n_arrays)

        intensity = []
        for _hkl in hkl_array:
            if scattering_type in ['xray','x','x-ray','thomson','charge']:
                intensity += self.x_ray(_hkl).tolist()
            elif scattering_type in ['neutron','n','nuclear']:
                intensity += self.neutron(_hkl).tolist()
            elif scattering_type in ['xray magnetic','magnetic xray','spin xray','xray spin']:
                intensity += list(self.xray_magnetic(_hkl)*1e4)
            elif scattering_type in ['neutron magnetic','magnetic neutron','magnetic']:
                intensity += list(self.magnetic_neutron(_hkl)*1e4)
            elif scattering_type in ['xray dispersion']:
                intensity += self.xray_dispersion(_hkl, self._energy_kev).tolist()
            elif scattering_type in ['xray resonant','resonant','resonant xray','rxs']:
                intensity += self.xray_resonant(_hkl).tolist()
            elif scattering_type in ['xray resonant magnetic', 'xray magnetic resonant',
                                     'resonant magnetic', 'magnetic resonant']:
                intensity += self.xray_resonant_magnetic(
                    _hkl,
                    self._energy_kev,
                    self._azimuthal_reference,
                    self._azimuthal_angle,
                    self._polarisation,
                    F0=0, F1=1, F2=0).tolist()
            elif scattering_type in ['xray nonresonant magnetic', 'xray magnetic nonresonant',
                                     'nonresonant magnetic', 'magnetic nonresonant',
                                     'xray non-resonant magnetic', 'xray magnetic non-resonant',
                                     'non-resonant magnetic', 'magnetic non-resonant']:
                intensity += self.xray_resonant_magnetic(
                    _hkl,
                    self._energy_kev,
                    self._azimuthal_reference,
                    self._azimuthal_angle,
                    self._polarisation).tolist()
            else:
                print('Scattering type not defined')
        return np.array(intensity)

    def old_structure_factor(self, HKL, scattering_type=None):
        """
        Calculate the complex structure factor for the given HKL
          Crystal.structure_factor([1,0,0])
          Crystal.structure_factor([[1,0,0],[2,0,0],[3,0,0])
        Returns an array with the same length as HKL, giving the complex structure factor at each reflection.

        Notes:
        - Uses x-ray atomic form factors, calculated from approximated tables in the ITC
        - This may be a little slow for large numbers of reflections, as it is not currently
         possible to use accelerated calculation methods in Jython.
        - Debye-Waller factor (atomic displacement) is applied for isotropic ADPs
        - Crystal.scale is used to scale the complex structure factor, so the intensity is
         reduced by (Crystal.scale)^2
        - Testing against structure factors calculated by Vesta.exe is very close, though there
          are some discrepancies, probably due to the method of calculation of the form factor.
        """
        prev_sf_setting = self._return_structure_factor
        self._return_structure_factor = True
        sf = self.intensity(HKL, scattering_type)
        self._return_structure_factor = prev_sf_setting
        return sf
    
    def hkl(self, HKL, energy_kev=None):
        """ Calculate the two-theta and intensity of the given HKL, display the result"""
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=float).reshape([-1,3])
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        inten = self.intensity(HKL)
        
        print('Energy = %6.3f keV' % energy_kev)
        print('( h, k, l) TwoTheta  Intensity')
        for n in range(len(tth)):
            print('(%2.0f,%2.0f,%2.0f) %8.2f  %9.2f' % (HKL[n,0],HKL[n,1],HKL[n,2],tth[n],inten[n]))
    
    def hkl_reflection(self, HKL, energy_kev=None):
        """
        Calculate the theta, two-theta and intensity of the given HKL in reflection geometry, display the result
        Uses sample orientation set up in setup_scatter
        :param HKL: [h,k,l] or list of hkl
        :param energy_kev: None or float
        :return: str
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=float).reshape([-1,3])
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        theta = self.xtl.Cell.theta_reflection(HKL, energy_kev, self._scattering_specular_direction, self._scattering_theta_offset)
        inten = self.intensity(HKL)
        
        print('Energy = %6.3f keV' % energy_kev)
        print('Specular Direction = (%1.0g,%1.0g,%1.0g)' %
              (self._scattering_specular_direction[0],
               self._scattering_specular_direction[1],
               self._scattering_specular_direction[2]))
        print('( h, k, l)    Theta TwoTheta  Intensity')
        for n in range(len(tth)):
            print('(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f' %
                  (HKL[n, 0], HKL[n, 1], HKL[n, 2], theta[n], tth[n], inten[n]))
    
    def hkl_transmission(self,HKL,energy_kev=None):
        " Calculate the theta, two-theta and intensity of the given HKL in transmission geometry, display the result"
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        HKL = np.asarray(np.rint(HKL),dtype=float).reshape([-1,3])
        tth = self.xtl.Cell.tth(HKL,energy_kev)
        theta = self.xtl.Cell.theta_transmission(HKL, energy_kev, self._scattering_specular_direction,self._scattering_theta_offset)
        inten = self.intensity(HKL)
        
        print('Energy = %6.3f keV' % energy_kev)
        print('Direction parallel to beam  = (%1.0g,%1.0g,%1.0g)' %(self._scattering_parallel_direction[0],self._scattering_parallel_direction[1],self._scattering_parallel_direction[2]))
        print('( h, k, l)    Theta TwoTheta  Intensity')
        for n in range(len(tth)):
            print('(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f' % (HKL[n,0],HKL[n,1],HKL[n,2],theta[n],tth[n],inten[n]))

    def generate_powder(self, q_max=8, peak_width=0.01, background=0, powder_average=True):
        """
        *DEPRECIATED*
        Generates array of intensities along a spaced grid, equivalent to a powder pattern.
          Q,I = generate_powder(energy_kev=8.0,peak_width=0.05,background=0)
            q_max = maximum Q, in A-1
            peak_width = width of convolution, in A-1
            background = average of normal background
            powder_average = True*/False, apply the powder averaging correction
          Returns:
            Q = [1000x1] array of wave-vector values
            I = [1000x1] array of intensity values

        Note: This function is depreciated, use self.powder() instead.
        Note: To get two-theta values use:
            tth = fc.cal2theta(Q, energy_kev)
        Note: To get d-spacing values use:
            dspace = fc.caldspace(Q)
        """

        # Get reflections
        hmax, kmax, lmax = fc.maxHKL(q_max, self.xtl.Cell.UVstar())
        HKL = fc.genHKL([hmax, -hmax], [kmax, -kmax], [lmax, -lmax])
        HKL = self.xtl.Cell.sort_hkl(HKL)  # required for labels
        Qmag = self.xtl.Cell.Qmag(HKL)
        HKL = HKL[Qmag < q_max, :]
        Qmag = self.xtl.Cell.Qmag(HKL)
        # Qmag = Qmag[Qmag<q_max]

        # Calculate intensities
        I = self.intensity(HKL)

        if powder_average:
            # Apply powder averging correction, I0/|Q|**2
            I = I/(Qmag+0.001)**2

        # create plotting mesh
        pixels = int(2000 * q_max)  # reduce this to make convolution faster
        pixel_size = q_max / (1.0 * pixels)
        peak_width_pixels = peak_width / (1.0 * pixel_size)
        mesh = np.zeros([pixels])
        mesh_q = np.linspace(0, q_max, pixels)

        # add reflections to background
        pixel_coord = Qmag / (1.0 * q_max)
        pixel_coord = (pixel_coord * (pixels - 1)).astype(int)

        for n in range(1, len(I)):
            mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + I[n]

        # Convolve with a gaussian (if >0 or not None)
        if peak_width:
            gauss_x = np.arange(-3*peak_width_pixels, 3*peak_width_pixels + 1)  # gaussian width = 2*FWHM
            G = fg.gauss(gauss_x, None, height=1, centre=0, fwhm=peak_width_pixels, bkg=0)
            mesh = np.convolve(mesh, G, mode='same')

            # Add background (if >0 or not None)
        if background:
            bkg = np.random.normal(background, np.sqrt(background), [pixels])
            mesh = mesh + bkg
        return mesh_q, mesh

    def powder_correction(self, HKL, intensities, symmetric_multiplyer=True, powder_average=True):
        """
        Averages symmetric reflections and applies symmetry multipliers and 1/q^2 correction
        Ic = I0*C
        :param HKL: [nx3] array of [h,k,l] reflections
        :param intensities: [nx1] array of reflection intensities
        :return: [mx3], [mx1] arrays of averaged, corrected reflections + intensity
        """
        # Average symmetric reflections
        rhkl, rinten = self.xtl.Symmetry.average_symmetric_intensity(HKL, intensities)

        if symmetric_multiplyer:
            multiplyer = self.xtl.Symmetry.reflection_multiplyer(rhkl)
            rinten = rinten*multiplyer

        if powder_average:
            q = self.xtl.Cell.Qmag(rhkl)
            rinten = rinten/(q+0.001)**2

        return rhkl, rinten

    def detector_image(self, detector_distance_mm=100., delta=0, gamma=0, height_mm=100., width_mm=100.,
                       pixel_size_mm=0.1, energy_range_ev=1000., peak_width_deg=0.5, wavelength_a=None,
                       background=0, min_intensity=0.01):
        """
        Calcualte detector image
          Creates a detector rotated about the sample and generates incident reflections on the detector.
          Reflections incident on the detector are scaled by the distance of the scattered wavevector from the Ewald
          sphere and also scaled by all reflections incident on the Ewald sphere.
        Example
          tth = xtl.Cell.tth([0, 0, 2], wavelength_a=1)[0]
          xtl.Cell.orientation.rotate_6circle(mu=tth/2)  # Orient crystal to scattering position
          xx, yy, mesh, reflist = xtl.Scatter.detector_image(detector_distance_mm=100, gamma=tth, wavelength_a=1)
          # Plot detector image
          plt.figure()
          plt.pcolormesh(xx, yy, mesh, vmin=0, vmax=1, shading='auto')
          # reflection labels
          for n in range(len(reflist['hkl'])):
              plt.text(reflist['detx'][n], reflist['dety'][n], reflist['hkl_str'][n], c='w')
        :param detector_distance_mm: float, Detector distance in mm (along (0,1,0))
        :param delta: flaot, angle to rotate detector about (0,0,1) in Deg
        :param gamma: float, angle to rotate detector about (1,0,0) in Deg
        :param height_mm: float, detector size vertically (along (0,0,1))
        :param width_mm: float, detector size horizontally (along (1,0,0))
        :param pixel_size_mm: float, size of pixels, determines grid size
        :param energy_range_ev: float, determines width of energies in incident intensity in eV
        :param peak_width_deg: float, determines peak width on detector
        :param wavelength_a: float, wavelength in A
        :param background: float, detector background
        :param min_intensity: float, min intenisty to include
        :return xx: [height//pixelsize x width//pixelsize] array of x-coordinates
        :return yy: [height//pixelsize x width//pixelsize] array of y-coordinates
        :return mesh: [height//pixelsize x width//pixelsize] array of intensities
        :return reflist: dict with info about each reflection on detector
        """
        domain_size = fc.scherrer_size(peak_width_deg, 45, wavelength_a=wavelength_a)  # A
        peak_width = fc.dspace2q(domain_size)  # inverse Angstrom
        resolution = fc.wavevector(energy_kev=energy_range_ev / 1000.)  # inverse Angstrom

        # Define Detector
        D = fc.rotmatrixz(-delta)  # left handed
        G = fc.rotmatrixx(gamma)
        R = np.dot(G, D)
        lab = self.xtl.Cell.orientation.labframe
        initial_position = detector_distance_mm * np.array([0, 1., 0])
        position_mm = fc.labvector(initial_position, R=R, LAB=lab)
        normal_dir = fc.labvector([0, -1, 0], R=R, LAB=lab)
        x_axis = width_mm * fg.norm(np.cross(normal_dir, (0, 0, 1.)))
        z_axis = height_mm * fg.norm(np.cross(x_axis, normal_dir))
        pixels_width = int(width_mm / pixel_size_mm) + 1  # n pixels
        distance_mm = fg.mag(position_mm)
        fwhm_mm = distance_mm * np.tan(np.deg2rad(peak_width_deg))
        corners = np.array([
            position_mm + x_axis / 2 + z_axis / 2,
            position_mm + x_axis / 2 - z_axis / 2,
            position_mm - x_axis / 2 - z_axis / 2,
            position_mm - x_axis / 2 + z_axis / 2,
        ])

        # Define lattice
        if wavelength_a is None:
            wavelength_a = fc.energy2wave(self.get_energy())
        hkl = self.xtl.Cell.all_hkl(max_angle=180, wavelength_a=wavelength_a)
        qxqyqz = self.xtl.Cell.calculateQ(hkl)

        # Find lattice points incident on detector
        lab = self.xtl.Cell.orientation.labframe
        ki = fc.wavevector(wavelength=wavelength_a) * fc.labvector([0, 1, 0], LAB=lab)
        kf = qxqyqz + ki
        directions = fg.norm(kf)
        diff = fc.wavevector_difference(qxqyqz, ki)  # difference in A-1

        corner_angle = max(abs(fg.vectors_angle_degrees(position_mm, corners)))
        vec_angles = abs(fg.vectors_angle_degrees(position_mm, directions))
        check = vec_angles < corner_angle  # reflections in the right quadrant
        ixyz = np.nan * np.zeros([len(directions), 3])
        for n in np.flatnonzero(check):
            ixyz[n] = fg.plane_intersection((0, 0, 0), directions[n], position_mm, normal_dir)
        # incident positions on detector
        iuvw = fg.index_coordinates(np.subtract(ixyz, position_mm), [x_axis, z_axis, normal_dir])
        iuvw[np.any(abs(iuvw) > 0.5, axis=1)] = [np.nan, np.nan, np.nan]

        # Remove non-incident reflections
        idx = ~np.isnan(iuvw[:, 0])
        hkl = hkl[idx, :]
        qxqyqz = qxqyqz[idx, :]
        iuvw = iuvw[idx, :]

        # Calculate intensities
        intensity = self.xtl.Scatter.intensity(hkl)
        res = np.sqrt(resolution ** 2 + peak_width ** 2)  # combine resolutions in A-1
        scaled_inten = fc.scale_intensity(intensity, diff[idx], res)
        scale = sum(fc.scale_intensity(1, diff, res))
        scaled_inten = scaled_inten / scale  # reduce intensity by total reflected intensity
        good = scaled_inten > min_intensity  # minimise the number of gaussians to generate

        # Calculate peak widths etc.
        qmag = fg.mag(qxqyqz[good])
        tth = fc.cal2theta(qmag, wavelength_a=wavelength_a)
        tth = 1.0 * tth
        tth[tth > 175] = 175.  # stop peaks becoming too broad at high angle
        fwhm = fc.scherrer_fwhm(domain_size, tth, wavelength_a=wavelength_a)
        hkl_str = np.array(fc.hkl2str(hkl[good]).split('\n'))
        peak_x = width_mm * iuvw[good, 0]
        peak_z = height_mm * iuvw[good, 1]

        # Generate the detector plane using gaussians on a plane
        xx, yy, mesh = fc.peaks_on_plane(
            peak_x=peak_x,
            peak_y=peak_z,
            peak_height=scaled_inten[good],
            peak_width=fwhm_mm,
            max_x=width_mm / 2,
            max_y=height_mm / 2,
            pixels_width=pixels_width,
            background=background
        )
        reflist = {
            'hkl': hkl[good, :],
            'hkl_str': hkl_str,
            'qxqyqz': qxqyqz[good, :],
            'fwhm': fwhm,
            'intensity': intensity[good],
            'scaled': scaled_inten[good],
            'detx': peak_x,
            'dety': peak_z,
        }
        return xx, yy, mesh, reflist

    def print_all_reflections(self, energy_kev=None, print_symmetric=False,
                              min_intensity=0.01, max_intensity=None, units=None):
        """
        Prints a list of all allowed reflections at this energy
            energy_kev = energy in keV
            print_symmetric = False*/True : omits reflections with the same intensity at the same angle
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
            units = None/ 'twotheta'/ 'q'/ 'dspace' : choose scattering angle units to display
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        if min_intensity is None: min_intensity = -1
        if max_intensity is None: max_intensity = np.inf
        
        hkl = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        if not print_symmetric:
            hkl = self.xtl.Symmetry.remove_symmetric_reflections(hkl)
        hkl = self.xtl.Cell.sort_hkl(hkl)

        tth = self.xtl.Cell.tth(hkl, energy_kev)
        inrange = np.all([tth < self._scattering_max_twotheta, tth > self._scattering_min_twotheta], axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]
        #inten = np.sqrt(self.intensity(hkl)) # structure factor
        inten = self.intensity(hkl)

        if units is None:
            units = self._powder_units
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
        
        fmt = '(%3.0f,%3.0f,%3.0f) %10.2f  %9.2f\n'
        outstr = ''
        
        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Radiation: %s\n' % self._scattering_type
        outstr += '( h, k, l)    %10s  Intensity\n' % unit_str
        #outstr+= fmt % (hkl[0,0],hkl[0,1],hkl[0,2],tth[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(1, len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            count += 1
            outstr += fmt % (hkl[n,0], hkl[n,1], hkl[n,2],unit[n],inten[n])
        outstr += 'Reflections: %1.0f\n' % count
        return outstr
    
    def print_ref_reflections(self, energy_kev=None, min_intensity=0.01, max_intensity=None):
        """
        Prints a list of all allowed reflections at this energy in the reflection geometry
            energy = energy in keV
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
                       |
                    // |\
                    \\ |
                     \\|___/____
                      \\   \
                       \\__
        
        Note, to change min/max theta values or specular direciton, change the following attributres of 
        the crystal object:
            self._scattering_max_two_theta   :  maximum detector (two-theta) angle
            self._scattering_min_theta       :  minimum sample angle = -opening angle
            self._scattering_max_theta       :  maximum sample angle = opening angle
            self._scattering_theta_offset    :  sample offset angle
            self._scattering_specular_direction : [h,k,l] : reflections normal to sample surface
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev

        if min_intensity is None: min_intensity = -1
        if max_intensity is None: max_intensity = np.inf

        HKL = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        HKL = self.xtl.Cell.sort_hkl(HKL)
        tth = self.xtl.Cell.tth(HKL, energy_kev)
        HKL = HKL[tth > self._scattering_min_twotheta, :]
        tth = tth[tth > self._scattering_min_twotheta]
        theta = self.xtl.Cell.theta_reflection(HKL, energy_kev, self._scattering_specular_direction,
                                               self._scattering_theta_offset)
        # inten = np.sqrt(self.intensity(HKL)) # structure factor
        inten = self.intensity(HKL)

        p1 = (theta > self._scattering_min_theta) * (theta < self._scattering_max_theta)
        p2 = (tth > (theta + self._scattering_min_theta)) * (tth < (theta + self._scattering_max_theta))
        pos_theta = p1 * p2
        
        fmt = '(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f\n'
        outstr = ''

        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Radiation: %s\n' % self._scattering_type
        outstr += 'Specular Direction = (%1.0g,%1.0g,%1.0g)\n' % (
            self._scattering_specular_direction[0],
            self._scattering_specular_direction[1],
            self._scattering_specular_direction[2]
        )
        outstr += '( h, k, l) TwoTheta    Theta  Intensity\n'
        # outstr+= fmt % (HKL[0,0],HKL[0,1],HKL[0,2],tth[0],theta[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(1, len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            if not pos_theta[n]: continue
            # if not print_symmetric and np.abs(tth[n]-tth[n-1]) < 0.01: continue # only works if sorted
            count += 1
            outstr += fmt % (HKL[n, 0], HKL[n, 1], HKL[n, 2], tth[n], theta[n], inten[n])
        outstr += 'Reflections: %1.0f\n' % count
        return outstr
    
    def print_tran_reflections(self, energy_kev=None, min_intensity=0.01, max_intensity=None):
        r"""
        Prints a list of all allowed reflections at this energy in the transmission geometry
            energy = energy in keV
            min_intensity = None/ 0.01 : omits reflections less than this (remove extinctions)
            max_intensity = None/ 0.01 : omits reflections greater than this (show extinctions only)
                   \ /      
             --<-- || --<-- 
                  / \       
        
        Note, to change min/max theta values or specular direciton, change the following attributres of 
        the crystal object:
            self._scattering_max_two_theta   :  maximum detector (two-theta) angle
            self._scattering_min_theta       :  minimum sample angle = -opening angle
            self._scattering_max_theta       :  maximum sample angle = opening angle
            self._scattering_theta_offset    :  sample offset angle
            self._scattering_parallel_direction : [h,k,l] : reflections normal to sample surface, parallel to beam at theta = 0
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev

        if min_intensity is None: min_intensity = -1
        if max_intensity is None: max_intensity = np.inf

        HKL = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        HKL = self.xtl.Cell.sort_hkl(HKL)
        tth = self.xtl.Cell.tth(HKL, energy_kev)
        HKL = HKL[tth > self._scattering_min_twotheta, :]
        tth = tth[tth > self._scattering_min_twotheta]
        theta = self.xtl.Cell.theta_transmission(HKL, energy_kev, self._scattering_parallel_direction)
        # inten = np.sqrt(self.intensity(HKL)) # structure factor
        inten = self.intensity(HKL)

        p1 = (theta > self._scattering_min_theta) * (theta < self._scattering_max_theta)
        p2 = (tth > (theta + self._scattering_min_theta)) * (tth < (theta + self._scattering_max_theta))
        pos_theta = p1 * p2

        fmt = '(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f\n'
        outstr = ''

        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Radiation: %s\n' % self._scattering_type
        outstr += 'Direction parallel to beam  = (%1.0g,%1.0g,%1.0g)\n' % (
            self._scattering_parallel_direction[0],
            self._scattering_parallel_direction[1],
            self._scattering_parallel_direction[2]
        )
        outstr += '( h, k, l) TwoTheta    Theta  Intensity\n'
        # outstr+= fmt % (HKL[0,0],HKL[0,1],HKL[0,2],tth[0],theta[0],inten[0]) # hkl(0,0,0)
        count = 0
        for n in range(1, len(tth)):
            if inten[n] < min_intensity: continue
            if inten[n] > max_intensity: continue
            if not pos_theta[n]: continue
            # if not print_symmetric and np.abs(tth[n]-tth[n-1]) < 0.01: continue # only works if sorted
            count += 1
            outstr += fmt % (HKL[n, 0], HKL[n, 1], HKL[n, 2], tth[n], theta[n], inten[n])
        outstr += ('Reflections: %1.0f\n' % count)
        return outstr

    def print_xray_resonant(self, HKL, energy_kev=None, azim_zero=(1, 0, 0), psi=0, F0=0, F1=1, F2=0):
        """
        Return string with resonant magnetic contriubtions to the x-ray scattering
        :param HKL: array(nx3) of (h,k,l) reflections
        :param energy_kev: incident photon energy in keV
        :param azim_zero: (h,k,l) reference vector defining azimuthal zero angle
        :param psi: float, azimuthal angle
        :param F0, F1, F2: Resonance factor Flm
        :return: str
        """
        HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        inp = {
            'HKL': HKL,
            'energy_kev': energy_kev,
            'azim_zero': azim_zero,
            'psi': psi,
            'F0': F0, 'F1': F1, 'F2': F2,
            'disp': False
        }
        xr = self.x_ray(HKL)
        ss = self.xray_resonant_magnetic(polarisation='ss', **inp)
        sp = self.xray_resonant_magnetic(polarisation='sp', **inp)
        ps = self.xray_resonant_magnetic(polarisation='ps', **inp)
        pp = self.xray_resonant_magnetic(polarisation='pp', **inp)

        tth = self.xtl.Cell.tth(HKL, energy_kev)
        hkl_str = ['({:3.0f},{:3.0f},{:3.0f})'.format(*h) for h in HKL]

        outstr = 'Resonant Magnetic X-Ray scattering: %s\n' % self.xtl.name
        outstr += u'Energy = %6.3f keV (%.3f \u212B)\n' % (energy_kev, fc.energy2wave(energy_kev))
        outstr += u'azimuth = %.2f Deg, azim_zero = %s || incident beam\n' % (psi, fc.hkl2str(azim_zero))
        outstr += '(  h,  k,  l)  Two-Theta      Charge      '\
                  '\u03c3-\u03c3       \u03c3-\u03c0       \u03c0-\u03c3       \u03c0-\u03c0\n'
        fmt = '%s %10.2f  %9.2f  %8.4f  %8.4f  %8.4f  %8.4f'
        outstr += '\n'.join([fmt % (hkl_str[n], tth[n], xr[n], ss[n], sp[n], ps[n], pp[n]) for n in range(len(HKL))])
        return outstr

    def print_xray_nonresonant(self, HKL, energy_kev=None, azim_zero=(1, 0, 0), psi=0):
        """
        Return string with non-resonant magnetic contriubtions to the x-ray scattering
        :param HKL: array(nx3) of (h,k,l) reflections
        :param energy_kev: incident photon energy in keV
        :param azim_zero: (h,k,l) reference vector defining azimuthal zero angle
        :param psi: float, azimuthal angle
        :return: str
        """
        HKL = np.asarray(HKL, dtype=float).reshape([-1, 3])
        inp = {
            'HKL': HKL,
            'energy_kev': energy_kev,
            'azim_zero': azim_zero,
            'psi': psi,
            'disp': False
        }
        xr = self.x_ray(HKL)
        ss = self.xray_nonresonant_magnetic(polarisation='ss', **inp)
        sp = self.xray_nonresonant_magnetic(polarisation='sp', **inp)
        ps = self.xray_nonresonant_magnetic(polarisation='ps', **inp)
        pp = self.xray_nonresonant_magnetic(polarisation='pp', **inp)

        tth = self.xtl.Cell.tth(HKL, energy_kev)
        hkl_str = ['({:3.0f},{:3.0f},{:3.0f})'.format(*h) for h in HKL]

        outstr = 'NonResonant Magnetic X-Ray scattering: %s\n' % self.xtl.name
        outstr += u'Energy = %6.3f keV (%.3f \u212B)\n' % (energy_kev, fc.energy2wave(energy_kev))
        outstr += u'azimuth = %.2f Deg, azim_zero = %s || incident beam\n' % (psi, fc.hkl2str(azim_zero))
        outstr += '(  h,  k,  l)  Two-Theta      Charge      '\
                  '\u03c3-\u03c3       \u03c3-\u03c0       \u03c0-\u03c3       \u03c0-\u03c0\n'
        fmt = '%s %10.2f  %9.2f  %8.4f  %8.4f  %8.4f  %8.4f'
        outstr += '\n'.join([fmt % (hkl_str[n], tth[n], xr[n], ss[n], sp[n], ps[n], pp[n]) for n in range(len(HKL))])
        return outstr

    def print_symmetric_reflections(self, HKL):
        """Prints equivalent reflections"""
        
        symHKL = self.xtl.Symmetry.symmetric_reflections_unique(HKL)
        Nsyms = len(symHKL)
        outstr = ''
        outstr+= 'Equivalent reflections: %d\n' % Nsyms
        for n in range(Nsyms):
            outstr+= '(%5.3g,%5.3g,%5.3g)\n' % (symHKL[n,0],symHKL[n,1],symHKL[n,2])
        return outstr
    
    def print_atomic_contributions(self, HKL):
        """
        Prints the atomic contributions to the structure factor
        """

        HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        Nref = len(HKL)

        # Calculate the full intensity
        I = self.intensity(HKL)

        # Calculate the structure factors of the symmetric atomic sites
        base_label = self.xtl.Atoms.label
        uvw, atom_type, label, occ, uiso, mxmymz = self.xtl.Structure.get()

        Qmag = self.xtl.Cell.Qmag(HKL)

        # Get atomic form factors
        # ff = fc.xray_scattering_factor(atom_type, Qmag)
        ff = self.scattering_factors(HKL)

        # Get Debye-Waller factor
        dw = fc.debyewaller(uiso, Qmag)

        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)

        # Calculate structure factor
        SF = ff * dw * occ * np.exp(1j * 2 * np.pi * dot_KR)

        # Sum structure factors of each base label in atoms
        SFbase = np.zeros([len(HKL), len(base_label) + 1], dtype=complex)
        for n in range(len(base_label)):
            label_idx = label == base_label[n]
            SFbase[:, n] = np.sum(SF[:, label_idx], axis=1)
        SFbase[:, -1] = SFbase.sum(axis=1)  # add the sum

        # Get the real part of the structure factor
        # SFtot = np.sqrt(np.real(SF * np.conj(SF)))
        SFrel = np.real(SFbase)
        SFimg = np.imag(SFbase)

        # Generate the results
        outstr = ''
        outstr += '( h, k, l) Intensity ' + ' '.join(['%12s    ' % x for x in base_label]) + '      Total SF\n'
        for n in range(Nref):
            ss = ' '.join(['%6.1f + i%-6.1f' % (x, y) for x, y in zip(SFrel[n], SFimg[n])])
            outstr += '(%2.0f,%2.0f,%2.0f) %9.2f   %s\n' % (HKL[n, 0], HKL[n, 1], HKL[n, 2], I[n], ss)
        return outstr

    def print_symmetry_contributions(self, HKL):
        """
        Prints the symmetry contributions to the structure factor for each atomic site
        """

        HKL = np.asarray(np.rint(HKL), dtype=float).reshape([-1, 3])
        Nref = len(HKL)

        # Calculate the full intensity
        I = self.intensity(HKL)

        # Calculate the structure factors of the symmetric atomic sites
        buvw, btype, base_label, bocc, buiso, bmxmymz = self.xtl.Atoms.get()
        operations = np.hstack([self.xtl.Symmetry.symmetric_coordinate_operations(buvw[n]) for n in range(len(buvw))])
        rotations = np.hstack(
            [self.xtl.Symmetry.symmetric_coordinate_operations(buvw[n], True)[1] for n in range(len(buvw))])

        # Calculate the structure factors
        uvw, type, label, occ, uiso, mxmymz = self.xtl.Structure.get()
        Qmag = self.xtl.Cell.Qmag(HKL)
        # ff = fc.xray_scattering_factor(type, Qmag)
        ff = self.scattering_factors(HKL)
        dw = fc.debyewaller(uiso, Qmag)
        dot_KR = np.dot(HKL, uvw.T)
        phase = np.exp(1j * 2 * np.pi * dot_KR)
        sf = ff * dw * occ * phase

        # Generate the results
        outstr = ''
        for n in range(Nref):
            ss = '\n'
            all_phase = 0j
            all_sf = 0j
            for lab in base_label:
                label_idx = np.argwhere(label == lab)
                ss += '  %s\n' % lab
                tot_phase = 0j
                tot_sf = 0j
                for a in label_idx:
                    uvwstr = '(%-7.3g,%-7.3g,%-7.3g)' % (uvw[a, 0], uvw[a, 1], uvw[a, 2])
                    phstr = fg.complex2str(phase[n, a])
                    sfstr = fg.complex2str(sf[n, a])
                    val = (a, uvwstr, operations[a[0]], rotations[a[0]], phstr, sfstr)
                    ss += '    %3d %s %25s %20s  %s  %s\n' % val
                    tot_phase += phase[n, a]
                    tot_sf += sf[n, a]
                ss += '%74sTotal:  %s  %s\n' % (' ', fg.complex2str(tot_phase), fg.complex2str(tot_sf))
                all_phase += tot_phase
                all_sf += tot_sf
            ss += '%62s Reflection Total:  %s  %s\n' % (' ', fg.complex2str(all_phase), fg.complex2str(all_sf))
            outstr += '(%2.0f,%2.0f,%2.0f) I = %9.2f    %s\n' % (HKL[n, 0], HKL[n, 1], HKL[n, 2], I[n], ss)
        return outstr

    def orientation_reflections(self, energy_kev, hkl_1=None):
        """
        Return 2 reflections to use to orient a crystal
         1. a strong reflection that is easy to discriminate in 2-theta
         2. another strong reflection non-parallel and non-normal to (1)

        hkl_1, hkl_2, alternatives = xtl.Scatter.orientation_reflections(8)

        :param energy_kev: photon energy in keV
        :param hkl_1: None or [h,k,l], allows to specify the first reflection
        :returns hkl_1: [h,k,l] indices of reflection 1
        :returns hkl_2: [h,k,l] indices of reflection 2
        :returns hkl_2_alternatives: [[h,k,l],...] list of alternative reflections with same angles
        """
        # Reflection 1 seletor
        if hkl_1 is None:
            refs = self.get_hkl(energy_kev=energy_kev, remove_symmetric=True)
            ref_tth = self.xtl.Cell.tth(refs, energy_kev=energy_kev)
            ref_multiplicity = self.xtl.Symmetry.reflection_multiplyer(refs)
            ref_intensity = self.intensity(refs)
            ref_cluster = np.array([np.sum(1 / (np.abs(th - ref_tth) + 1)) - 1 for th in ref_tth])
            ref_select = ref_intensity * ref_multiplicity / ref_cluster  # ** 2
            hkl_1 = refs[np.argmax(ref_select), :]

        # Reflection 2 selector
        next_refs = self.get_hkl(energy_kev=energy_kev, remove_symmetric=False)
        # tth_1 = self.xtl.Cell.tth(hkl_1, energy_kev=energy_kev)
        # tth_2 = self.xtl.Cell.tth(refs, energy_kev=energy_kev)
        q_1 = self.xtl.Cell.calculateQ(hkl_1)
        q_2 = self.xtl.Cell.calculateQ(next_refs)
        angles = abs(fg.vectors_angle_degrees(q_1, q_2))
        ref_select = (angles > 1.) * (angles < 40.)
        if sum(ref_select) < 1:
            ref_select = angles > 1.
        next_refs = next_refs[ref_select]
        angles = angles[ref_select]
        tth_1 = self.xtl.Cell.tth(hkl_1, energy_kev=energy_kev)
        tth_2 = self.xtl.Cell.tth(next_refs, energy_kev=energy_kev)
        next_select = self.intensity(next_refs) / angles
        idx = np.argmax(next_select)
        hkl_2 = next_refs[idx]
        hkl_2_options = (abs(angles - angles[idx]) < 1.) * (abs(tth_2 - tth_2[idx]) < 1.)
        return hkl_1, hkl_2, next_refs[hkl_2_options]

    def find_close_reflections(self, HKL, energy_kev=None, max_twotheta=2, max_angle=10):
        """
        Find and print list of reflections close to the given one
        """
        
        if energy_kev is None:
            energy_kev = self._energy_kev
        
        all_HKL = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        all_HKL = self.xtl.Cell.sort_hkl(all_HKL)
        
        all_tth = self.xtl.Cell.tth(all_HKL,energy_kev)
        tth1 = self.xtl.Cell.tth(HKL, energy_kev)
        tth_dif = np.abs(all_tth-tth1)
        
        all_Q = self.xtl.Cell.calculateQ(all_HKL)
        Q1 = self.xtl.Cell.calculateQ(HKL)
        all_angles = np.abs([fg.ang(Q1,Q2,'deg') for Q2 in all_Q])
        
        selected = (tth_dif < max_twotheta)*(all_angles < max_angle)
        sel_HKL = all_HKL[selected,:]
        sel_tth = all_tth[selected]
        sel_angles = all_angles[selected]
        sel_intensity = self.intensity(sel_HKL)
        
        # Generate Results
        fmt = '(%2.0f,%2.0f,%2.0f) %8.2f %8.2f  %9.2f\n'
        outstr = '' 
        
        outstr+= 'Energy = %6.3f keV\n' % energy_kev
        outstr+= 'Close to Reflection (%2.0f,%2.0f,%2.0f)\n' %(HKL[0],HKL[1],HKL[2])
        outstr+= '( h, k, l) TwoTheta    Angle  Intensity\n'
        count = 0
        for n in range(0,len(sel_HKL)):
            count += 1
            outstr+= fmt % (sel_HKL[n,0],sel_HKL[n,1],sel_HKL[n,2],sel_tth[n],sel_angles[n],sel_intensity[n])
        outstr+= 'Reflections: %1.0f\n'%count
        return outstr

    def diff6circle_intensity(self, phi=0, chi=0, eta=0, mu=0, delta=0, gamma=0,
                              energy_kev=None, wavelength=1.0, fwhm=0.5):
        """
        Return the closest hkl and intensity factor
        :param phi: float sample angle in degrees
        :param chi: float sample angle in degrees
        :param eta: float sample angle in degrees
        :param mu: float sample angle in degrees
        :param delta: float detector angle in degrees
        :param gamma: float detector angle in degrees
        :param energy_kev: float energy in KeV
        :param wavelength: float wavelength in A
        :param fwhm: float peak width in A-1
        :return: [h,k,l], If
        """
        hkl, factor = self.xtl.Cell.diff6circle_match(phi, chi, eta, mu, delta, gamma, energy_kev, wavelength, fwhm)
        #self.setup_scatter(energy_kev=energy_kev, wavelength_a=wavelength)
        inten = self.intensity(hkl, energy_kev=energy_kev)
        return inten * factor

    def multiple_scattering(self, hkl, azir=(0, 0, 1), pv=(1, 0), energy_range=(7.8, 8.2), numsteps=60,
                            full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False):
        """
        Run multiple scattering code, plot result.

        mslist = xtl.Scatter.multiple_scattering([h,k,l], energy_range=[7.8, 8.2])

        See multiple_scattering.py for more details. Code created by Dr Gareth Nisbet, DLS
        :param hkl: [h,k,l] principle reflection
        :param azir: [h,k,l] reference of azimuthal 0 angle
        :param pv: [s,p] polarisation vector
        :param energy_range: [min, max] energy range in keV
        :param numsteps: int: number of calculation steps from energy min to max
        :param full: True/False: calculation type: full
        :param pv1: True/False: calculation type: pv1
        :param pv2: True/False: calculation type: pv2
        :param sfonly: True/False: calculation type: sfonly *default
        :param pv1xsf1: True/False: calculation type: pv1xsf1?
        :return: array
        """
        return ms.run_calcms(self.xtl, hkl, azir, pv, energy_range, numsteps,
                             full=full, pv1=pv1, pv2=pv2, sfonly=sfonly, pv1xsf1=pv1xsf1)

    def ms_azimuth(self, hkl, energy_kev, azir=(0, 0, 1), pv=(1, 0), numsteps=3, peak_width=0.1,
                   full=False, pv1=False, pv2=False, sfonly=True, pv1xsf1=False, energy_sum_range=0.002):
        """
        Returns an azimuthal dependence at a particular energy

        :param xtl: Crystal structure from Dans_Diffraction
        :param hkl: [h,k,l] principle reflection
        :param energy_kev: calculation energy
        :param azir: [h,k,l] reference of azimuthal 0 angle
        :param pv: [s,p] polarisation vector
        :param numsteps: int: number of calculation steps from energy min to max
        :param peak_width: convolution width
        :param full: True/False: calculation type: full
        :param pv1: True/False: calculation type: pv1
        :param pv2: True/False: calculation type: pv2
        :param sfonly: True/False: calculation type: sfonly *default
        :param pv1xsf1: True/False: calculation type: pv1xsf1?
        :param energy_sum_range: energy in keV to sum the calculation over (from energy_kev-range/2 to energy_kev+range/2)
        :return: None
        """

        energy_range = [energy_kev-energy_sum_range/2, energy_kev+energy_sum_range/2]
        mslist = self.xtl.Scatter.multiple_scattering(hkl, azir, pv, energy_range, numsteps,
                                                      full=full, pv1=pv1, pv2=pv2, sfonly=sfonly, pv1xsf1=pv1xsf1)

        if pv1 + pv2 + sfonly + full + pv1xsf1 != 0:
            azimuth = np.concatenate(mslist[:, [3, 4]])
            intensity = np.concatenate(mslist[:, [-1, -1]])
        else:
            azimuth = np.concatenate(mslist[:, [3, 4]])
            intensity = np.ones(azimuth)

        # create plotting mesh
        peak_width_pixels = 10
        pixels = int(360*(peak_width_pixels/ peak_width))
        mesh_azi = np.linspace(-180, 180, pixels)
        mesh = np.zeros(mesh_azi.shape)

        # add reflections to background
        pixel_coord = (azimuth-180) / 360.
        pixel_coord = (pixel_coord * pixels).astype(int)

        for n in range(len(intensity)):
            mesh[pixel_coord[n]] = mesh[pixel_coord[n]] + intensity[n]

        # Convolve with a gaussian (if >0 or not None)
        if peak_width:
            gauss_x = np.arange(-3 * peak_width_pixels, 3 * peak_width_pixels + 1)  # gaussian width = 2*FWHM
            G = fg.gauss(gauss_x, None, height=1, centre=0, fwhm=peak_width_pixels, bkg=0)
            mesh = np.convolve(mesh, G, mode='same')
        return mesh_azi, mesh

    '''  Removed tensor scattering 26/05/20 V1.7
    def tensor_scattering(self, atom_label, hkl, energy_kev=None, azir=[0, 0, 1], psideg=0, process='E1E1',
                          rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None):
        """
        Return tensor scattering intensities
          ss, sp, ps, pp = tensor_scattering('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param hkl: list/array, [h,k,l] reflection to calculate
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float/array, azimuthal angle/ range
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :return: ss, sp, ps, pp intensity values
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        # Calculate tensor scattering
        ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
            process, B, sitevec, sglist, lam, hkl, azir, psideg,
            K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
        )
        return ss, sp, ps, pp

    def tensor_scattering_stokes(self, atom_label, hkl, energy_kev=None, azir=[0, 0, 1], psideg=0, stokes=0,
                                 pol_theta=45, process='E1E1', rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None):
        """
        Return tensor scattering intensities for non-standard polarisation
          pol = tensor_scattering_stokes('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90, stokes=45)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param hkl: list/array, [h,k,l] reflection to calculate
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float, azimuthal angle
        :param stokes: float/array, rotation of polarisation analyser (0=sigma), degrees
        :param pol_theta: float, scattering angle of polarisation analyser, degrees
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :return: array of intensity values
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        # Calculate tensor scattering
        stokesvec_swl = [0, 0, 1]
        pol = ts.CalculateIntensityFromPolarizationAnalyser(
            process, B, sitevec, sglist, lam, hkl, azir, psideg, stokes, pol_theta,
            stokesvec_swl, rank, time, parity, mk, lk, sk
        )
        return pol

    def print_tensor_scattering(self, atom_label, hkl, energy_kev=None, azir=[0, 0, 1], psideg=0, process='E1E1',
                                rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None):
        """
        Return tensor scattering intensities
          ss, sp, ps, pp = tensor_scattering('Ru1', [0,0,3], 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param hkl: list/array, [h,k,l] reflection to calculate
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float, azimuthal angle
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :return: str
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        # Calculate tensor scattering
        ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
            process, B, sitevec, sglist, lam, hkl, azir, psideg,
            K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
        )
        outstr1 = ts.tensorproperties(sitevec, sglist, hkl, Parity=parity, Time=time)
        outstr2 = ts.print_tensors(B, sitevec, sglist, hkl, K=rank, Parity=parity, Time=time)
        outstr3 = "\nScattering Tensor:\n\n    [ss, sp] = [%5.2f, %5.2f]\n    [ps, pp]   [%5.2f, %5.2f]"
        outstr3 = outstr3 % (ss, sp, ps, pp)
        return outstr1 + outstr2 + outstr3

    def print_tensor_scattering_refs(self, atom_label, energy_kev=None, azir=[0, 0, 1], psideg=0, process='E1E1',
                                     rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None,
                                     print_symmetric=False, units=None):
        """
        Return tensor scattering intensities for all reflections at given azimuth and energy
          ss, sp, ps, pp = tensor_scattering('Ru1', 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param psideg: float, azimuthal angle
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :param print_symmetric: False*/True : omits reflections with the same intensity at the same angle
        :param units: None/ 'twotheta'/ 'q'/ 'dspace' : choose scattering angle units to display
        :return: str
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        hkl = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        if not print_symmetric:
            hkl = self.xtl.Symmetry.remove_symmetric_reflections(hkl)
        hkl = self.xtl.Cell.sort_hkl(hkl)

        tth = self.xtl.Cell.tth(hkl, energy_kev)
        inrange = np.all([tth < self._scattering_max_twotheta, tth > self._scattering_min_twotheta], axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]

        if units is None:
            units = self._powder_units
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

        fmt = '(%3.0f,%3.0f,%3.0f) %10.2f  %11.2f %11.2f %11.2f %11.2f\n'
        outstr = 'Tensor Scattering %s\n' % self.xtl.name
        outstr += 'Process: %s, site: %s\n' % (process, atom_label)
        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Psi_0 = (%.2g,%.2g,%.2g)  Psi = %3.3g\n' % (azir[0], azir[1], azir[2], psideg)
        outstr += '( h, k, l)    %10s  Sigma-Sigma    Sigma-Pi    Pi-Sigma       Pi-Pi\n' % unit_str

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        for n in range(1, len(tth)):
            ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
                process, B, sitevec, sglist, lam, hkl[n, :], azir, psideg,
                K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
            )
            outstr += fmt % (hkl[n, 0], hkl[n, 1], hkl[n, 2], unit[n], ss, sp, ps, pp)
        outstr += 'Reflections: %1.0f\n' % len(tth)
        return outstr

    def print_tensor_scattering_refs_max(self, atom_label, energy_kev=None, azir=[0, 0, 1], process='E1E1',
                                     rank=2, time=+1, parity=+1, mk=None, lk=None, sk=None,
                                     print_symmetric=False, units=None):
        """
        Return tensor scattering intensities for all reflections at given energy at maximum intensity psi
          ss, sp, ps, pp = tensor_scattering('Ru1', 2.838, [0,1,0], psideg=90)
        Uses TensorScatteringClass.py by Prof. Steve Collins, Dimaond Light Source Ltd.
        :param atom_label: str atom site label, e.g. Ru1
        :param energy_kev: float
        :param azir: list/array, [h,k,l] azimuthal reference
        :param process: str: 'Scalar', 'E1E1', 'E1E2', 'E2E2', 'E1E1mag', 'NonResMag'
        :param rank: int, 1,2,3: tensor rank. Only required
        :param time: +/-1 time symmetry
        :param parity: +/-1 parity
        :param mk:
        :param lk:
        :param sk:
        :param print_symmetric: False*/True : omits reflections with the same intensity at the same angle
        :param units: None/ 'twotheta'/ 'q'/ 'dspace' : choose scattering angle units to display
        :return: str
        """

        if energy_kev is None:
            energy_kev = self._energy_kev

        hkl = self.xtl.Cell.all_hkl(energy_kev, self._scattering_max_twotheta)
        if not print_symmetric:
            hkl = self.xtl.Symmetry.remove_symmetric_reflections(hkl)
        hkl = self.xtl.Cell.sort_hkl(hkl)

        tth = self.xtl.Cell.tth(hkl, energy_kev)
        inrange = np.all([tth < self._scattering_max_twotheta, tth > self._scattering_min_twotheta], axis=0)
        hkl = hkl[inrange, :]
        tth = tth[inrange]
        # Caluclate structure factor **2
        inten = self.intensity(hkl)

        if units is None:
            units = self._powder_units
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

        fmt = '(%3.0f,%3.0f,%3.0f) %10.2f %10.2f  %5.2f (%3.0f) %5.2f (%3.0f) %5.2f (%3.0f) %5.2f (%3.0f)\n'
        outstr = 'Tensor Scattering %s\n' % self.xtl.name
        outstr += 'Process: %s, site: %s\n' % (process, atom_label)
        outstr += 'Energy = %6.3f keV\n' % energy_kev
        outstr += 'Psi_0 = (%.2g,%.2g,%.2g)\n' % (azir[0], azir[1], azir[2])
        outstr += '( h, k, l)    %10s         I0  Sigma-Sigma    Sigma-Pi    Pi-Sigma       Pi-Pi\n' % unit_str

        # Structure data for input
        B = ts.latt2b(self.xtl.Cell.lp())
        sitevec = self.xtl.Atoms.uvw()[self.xtl.Atoms.label.index(atom_label)]
        sglist = ts.spacegroup_list_from_genpos_list(self.xtl.Symmetry.symmetry_operations)
        lam = fc.energy2wave(energy_kev)

        psideg = np.arange(0, 361, 5)
        for n in range(1, len(tth)):
            ss, sp, ps, pp = ts.CalculateIntensityInPolarizationChannels(
                process, B, sitevec, sglist, lam, hkl[n, :], azir, psideg,
                K=rank, Time=time, Parity=parity, mk=mk, lk=lk, sk=sk
            )
            iss = np.argmax(ss)
            isp = np.argmax(sp)
            ips = np.argmax(ps)
            ipp = np.argmax(pp)
            ss = ss[iss]
            sp = sp[isp]
            ps = ps[ips]
            pp = pp[ipp]
            psi_ss = psideg[iss]
            psi_sp = psideg[isp]
            psi_ps = psideg[ips]
            psi_pp = psideg[ipp]
            outstr += fmt % (hkl[n, 0], hkl[n, 1], hkl[n, 2], unit[n], inten[n],
                             ss, psi_ss, sp, psi_sp, ps, psi_ps, pp, psi_pp)
        outstr += 'Reflections: %1.0f\n' % len(tth)
        return outstr
    '''


class ScatteringTypes:
    """
    Container for available scattering types
    """
    def __init__(self, parent, typedict):
        self.parent = parent
        self.typedict = typedict

        typenames = typedict.keys()

        for name in typenames:
            attrname = name.replace(' ','_')
            setattr(self, attrname, self.ScatteringType(parent, name))

    def __call__(self, type):
        """
        Calling this container with a scattering type will set the scattering type in the parent class
        :param type: str scattering type
        :return: None
        """
        for typename in self.typedict:
            if type.lower() in self.typedict[typename]:
                self.parent._scattering_type = typename
                print('Changed scattering type to: %s'%typename)

    def i16(self):
        """
        Set max/min angles consistent with reflection geometry on beamline I16
        :return: None
        """
        self.parent.setup_scatter(type='xray', energy_kev=8.0, min_theta=-20, max_theta=140, min_twotheta=0, max_twotheta=140)

    def wish(self):
        """
        Set max/min angles consistent with reflection geometry on beamline I16
        :return: None
        """
        self.parent.setup_scatter(type='neutron', wavelength_a=1.5, min_theta=-180, max_theta=180, min_twotheta=-180, max_twotheta=180)

    class ScatteringType:
        """
        Container for scattering type switcher
        """
        def __init__(self, parent, typename):
            self.parent = parent
            self.typename = typename

        def __call__(self):
            self.parent._scattering_type = self.typename


class Reflections:
    """
    Contains h,k,l indices, intensity

    masking doesn't work currently
    """
    _energy_kev = 8.0
    _hkl_format = '(%3.0f,%3.0f,%3.0f)'

    def __init__(self, hkl, q_mag, intensity, energy_kev=None, wavelength_a=None):
        self.hkl = np.asarray(hkl).reshape([-1, 3])
        self.q_mag = np.asarray(q_mag).reshape([-1])
        self.intensity = np.asarray(intensity).reshape([-1])
        self.sort_qmag()
        self._mask = np.zeros_like(self.q_mag, dtype=int)

        if energy_kev is not None:
            self.set_energy(energy_kev)

        if wavelength_a is not None:
            self.set_wavelength(wavelength_a)

    def __repr__(self):
        return "Reflections( %d reflections, energy_kev=%1.5g )" % (len(self.hkl), self._energy_kev)

    def __str__(self):
        out = '  hkl           Q [A^-1]    d [A] 2theta [Deg]  Intensity\n'
        dspace = self.dspacing()
        tth = self.two_theta()
        labels = self.labels()
        for n in range(len(self.hkl)):
            if self._mask[n] == 0:
                out += '%14s %8.3f %8.3f %12.2f %10.2f\n' % (labels[n], self.q_mag[n], dspace[n], tth[n], self.intensity[n])
        return out

    def __getitem__(self, item):
        return self.hkl[item], self.q_mag[item], self.intensity[item]

    def set_energy(self, energy_kev):
        self._energy_kev = energy_kev

    def set_wavelength(self, wavelength_a):
        self.set_energy(fc.wave2energy(wavelength_a))

    def limits(self, min_val=None, max_val=None, limit_type='q'):
        """
        Adds limits
        :param min_val: float, minimum value in units determined by overlap_type
        :param max_val: float, maximum value
        :param limit_type: str 'q', 'tth' or 'd'
        :return: None
        """

        limit_type = limit_type.lower().replace(' ', '').replace(' ', '').replace('-', '').replace('_', '')
        if limit_type in ['tth', 'twotheta', '2theta']:
            min_val = 0 if min_val is None else fc.calqmag(min_val, self._energy_kev)
            max_val = np.max(self.q_mag) if max_val is None else fc.calqmag(max_val, self._energy_kev)
        elif limit_type in ['d', 'dspacing', 'dspace']:
            min_val = 0 if min_val is None else fc.dspace2q(min_val)
            max_val = np.max(self.q_mag) if max_val is None else fc.dspace2q(max_val)
        else:
            min_val = 0 if min_val is None else min_val
            max_val = np.max(self.q_mag) if max_val is None else max_val

        self._mask = np.zeros_like(self.q_mag, dtype=int)
        self._mask[self.q_mag < min_val] = 1
        self._mask[self.q_mag > max_val] = 1

    def mask(self, center=None, width=None, mask_type='q'):
        """
        Adds mask
        :param center: float, centre of mask in units determined by overlap_type
        :param width: float, width of mask
        :param limit_type: str 'q', 'tth' or 'd'
        :return: None
        """

        mask_type = mask_type.lower().replace(' ', '').replace(' ', '').replace('-', '').replace('_', '')
        if mask_type in ['tth', 'twotheta', '2theta']:
            center = -1 if center is None else fc.calqmag(center, self._energy_kev)
            width = 0 if width is None else fc.calqmag(width, self._energy_kev)
        elif mask_type in ['d', 'dspacing', 'dspace']:
            center = -1 if center is None else fc.dspace2q(center)
            width = 0 if width is None else fc.dspace2q(width)
        else:
            center = -1 if center is None else center
            width = 0 if width is None else width

        idx = np.abs(self.q_mag - center) <= width
        self._mask[idx] = 1

    def sort_qmag(self):
        """Sort arrays by qmag"""
        #idx = np.argsort(self.q_mag)
        qmag = np.round(self.q_mag, 4)
        inten = np.round(self.intensity, 4)
        idx = np.lexsort((self.hkl[:, 1], self.hkl[:, 0], self.hkl[:, 2], inten, qmag))
        self.hkl = self.hkl[idx, :]
        self.q_mag = self.q_mag[idx]
        self.intensity = self.intensity[idx]

    def sort_intensity(self):
        idx = np.argsort(self.intensity)
        self.hkl = self.hkl[idx, :]
        self.q_mag = self.q_mag[idx]
        self.intensity = self.intensity[idx]

    def sort_hkl(self):
        idx = np.lexsort((self.hkl[:, 1], self.hkl[:, 0], self.hkl[:, 2]))
        self.hkl = self.hkl[idx, :]
        self.q_mag = self.q_mag[idx]
        self.intensity = self.intensity[idx]

    def two_theta(self):
        return fc.cal2theta(self.q_mag, self._energy_kev)

    def dspacing(self):
        return fc.q2dspace(self.q_mag)

    def labels(self):
        """Return list of str labels "(h,k,l)" """
        return [self._hkl_format % (h[0], h[1], h[2]) for h in self.hkl]

    def non_overlapping_refs(self, min_overlap=0.05, overlap_type='q'):
        """
        Return list of non-overlapping label
        :param min_overlap: float, minimum overlap in units determined by overlap_type
        :param overlap_type: str 'q', 'tth' or 'd'
        :return:
        """

        self.sort_qmag()
        # Group the qmag array
        overlap_type = overlap_type.lower().replace(' ', '').replace(' ', '').replace('-', '').replace('_', '')
        if overlap_type in ['tth', 'twotheta', '2theta']:
            groups, array_index, group_index, counts = fg.group(self.two_theta(), min_overlap)
        elif overlap_type in ['d', 'dspacing', 'dspace']:
            groups, array_index, group_index, counts = fg.group(self.dspacing(), min_overlap)
        else:
            groups, array_index, group_index, counts = fg.group(self.q_mag, min_overlap)

        # loop over groups and select reflection with largest intensity
        ref_n = np.zeros(len(groups), dtype=int)
        for n in range(len(groups)):
            args = np.where(array_index == n)[0]
            # find max intensity
            ref_n[n] = args[np.argmax(self.intensity[args])]
        return ref_n

