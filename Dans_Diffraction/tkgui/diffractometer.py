"""
Dans-Diffraction
Diffractometer Simulator GUI

Simulate a generic detector situated around the crystal sample.
The sample and detector orientation are specified like a 6-axis diffractometer.

The angle conventions are those according to H. You, J. Appl. Cryst. (1999). 32, 614

On the top left, the reciprocal lattice is shown and any reflections incident on the detector are highlighted.

On the top right, the detector image is shown and incident reflections are highlighted.

On the bottom left, the lattice and detector orientation angles are controlled.

On the bottom right, the incident beam, intensity calculation options and detector size and shaper are specified.

Additional options can be found in the top menu. Any axis can be scanned to plot
the sum of the detector at each step

 Dr Daniel G Porter, dan.porter@diamond.ac.uk
 www.diamond.ac.uk
 Diamond Light Source, Chilton, Didcot, Oxon, OX11 0DE, U.K.
"""

import numpy as np
from itertools import cycle
from matplotlib.figure import Figure
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg

from .. import functions_general as fg
from .. import functions_crystallography as fc
from .. import functions_plotting as fp
from .. import Crystal
from .basic_widgets import tk, StringViewer, messagebox, topmenu, popup_about
from .basic_widgets import (TF, BF, SF, LF, HF, TTF,
                            bkg, ety, btn, opt, btn2,
                            btn_active, opt_active, txtcol,
                            btn_txt, ety_txt, opt_txt)

# Figure DPI
FIGURE_DPI = 80
FIGURE_3D_SIZE = [8, 4]
FIGURE_DET_SIZE = [7, 4]

# Help messages
MSG_PHI = "1st rotation, left handed roation about z'''-axis (crystal axis)"
MSG_CHI = "2nd rotation, right handed rotation about y''-axis."
MSG_ETA = "3rd rotation, left handed rotation about z'-axis"
MSG_MU = "4th rotation, right handed rotation about diffractometer x-axis"
MSG_DEL = "Detector rotation, left handed rotation about diffractometer z-axis"
MSG_GAM = "Detector rotation, right handed rotation about diffractometer x-axis"

# Defaults
DEFAULT_ENERGY = 8  # keV
DEFAULT_RESOLUTION = 200  # eV
DEFAULT_PEAKWIDTH = 0.2  # deg
DEFAULT_MAXQ = 4  # A-1
DEFAULT_DET_DISTANCE = 1000  # mm
DEFAULT_DET_WIDTH = 200  # mm
DEFAULT_DET_HEIGHT = 200  # mm
DEFAULT_DET_PIXEL = 0.2  # mm
# Plot options
OPTIONS = {
    '3d_lattice': True,
    '3d_basis': False,
    '3d_detector_corners': False,
    '3d_ref_arrows': False,
    'det_labels': True,
    'det_log': False,
    'det_clim': [0.0, 1.0],
    'det_cmap': 'viridis',
    'fig_dpi': FIGURE_DPI,
    'maxq': 4.0,
    'min_intensity': 0.01
}
# Detector Reflection list
REFLIST = {
    'hkl': np.zeros([0, 3], dtype=int),
    'qxqyqz': np.zeros([0, 3], dtype=float),
    'intensity': np.zeros([0], dtype=float),
    'scaled': np.zeros([0], dtype=float),
    'detx': np.zeros([0], dtype=float),
    'dety': np.zeros([0], dtype=float),
    'hkl_str': [],
    'fwhm': np.zeros([0], dtype=float),
    'color': [],
}
# Colours
col_cycle = cycle([TABLEAU_COLORS[c] for c in TABLEAU_COLORS])


def default_fun(*args, **kwargs):
    return None


def reflist_str():
    """Generate string from REFLIST"""
    out = 'Refs: %d\n' % (len(REFLIST['hkl']))
    out += 'X [mm] Y [mm]  Intensity     Scaled    FWHM    (h,k,l)\n'
    out += '\n'.join([
        '%6.2f %6.2f %10.2f %10.2f  %6.2f    %s' % (
            REFLIST['detx'][n], REFLIST['dety'][n], REFLIST['intensity'][n],
            REFLIST['scaled'][n], REFLIST['fwhm'][n], REFLIST['hkl_str'][n])
        for n in range(len(REFLIST['hkl']))
    ])
    return out


class Lattice:
    """
    Reciprocal Lattice
    """
    _current_phi = 0
    _current_chi = 0
    _current_eta = 0
    _current_mu = 0
    update_plots = default_fun
    generate_refs = default_fun
    update_widgets = default_fun
    get_wavelength = default_fun

    def __init__(self, crystal: Crystal):
        self.xtl = crystal
        self._uvstar = crystal.Cell.UVstar()
        self._u = crystal.Cell.orientation.umatrix
        self._lab = crystal.Cell.orientation.labframe

        self.wavelength_a = 1  # A
        self.resolution = 0.5  # A-1
        self.domain_size = 1000  # A
        self.hkl = np.zeros([0, 3], dtype=float)
        self.initial_qxqyqz = np.zeros([0, 3], dtype=float)
        self.rotated_qxqyqz = np.zeros([0, 3], dtype=float)
        self.initial_vectors = np.eye(3, dtype=float)
        self.vectors = np.eye(3, dtype=float)
        self.qmag = np.zeros([0], dtype=float)
        self.tth = np.zeros([0], dtype=float)
        self.fwhm = np.zeros([0], dtype=float)
        self.hkl_str = []
        self.intensity = np.zeros([0], dtype=float)
        self._calculated = np.zeros([0], dtype=bool)
        self.rotate_to(0, 0, 0, 0)

    def set_callbacks(self, generate_refs=None, update_widgets=None, update_plots=None, get_wavelength=None):
        if generate_refs:
            self.generate_refs = generate_refs
        if update_widgets:
            self.update_widgets = update_widgets
        if update_plots:
            self.update_plots = update_plots
        if get_wavelength:
            self.get_wavelength = get_wavelength

    def set_orientation(self, umatrix=None, labmatrix=None):
        if umatrix is not None:
            self._u = umatrix
        if labmatrix is not None:
            self._lab = labmatrix
        self.rotate_to(
            phi=self._current_phi,
            chi=self._current_chi,
            eta=self._current_eta,
            mu=self._current_mu,
        )

    def rotate_to(self, phi=0., chi=0., eta=0., mu=0.):
        self._current_phi = phi
        self._current_chi = chi
        self._current_eta = eta
        self._current_mu = mu
        R = fc.diffractometer_rotation(phi, chi, eta, mu)
        self.rotated_qxqyqz = fc.labvector(self.initial_qxqyqz, self._u, R, self._lab)
        self.vectors = fc.labvector(self.initial_vectors, self._u, R, self._lab)

    def generate_hkl(self, wavelength_a, max_q=4):
        """Generate reflection list"""
        self.wavelength_a = wavelength_a
        self.hkl = self.xtl.Cell.all_hkl(maxq=max_q)
        self.initial_qxqyqz = self.xtl.Cell.calculateQ(self.hkl)
        self.initial_vectors = self.xtl.Cell.calculateQ([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.qmag = fg.mag(self.initial_qxqyqz)
        self.tth = fc.cal2theta(self.qmag, wavelength_a=wavelength_a)
        tth = 1.0 * self.tth
        tth[tth > 175] = 175.  # stop peaks becoming too broad at high angle
        self.fwhm = fc.scherrer_fwhm(self.domain_size, tth, wavelength_a=wavelength_a, shape_factor=1.0)
        self.hkl_str = np.array(fc.hkl2str(self.hkl).split('\n'))
        self.intensity = np.zeros_like(self.qmag)
        self._calculated = np.zeros_like(self.qmag, dtype=bool)
        self.rotate_to(self._current_phi, self._current_chi, self._current_eta, self._current_mu)

    def generate_intensities(self, calculate_index=()):
        """Generate list of structure factors"""
        this_calculation = np.zeros_like(self.qmag, dtype=bool)
        this_calculation[calculate_index] = True
        this_calculation[self._calculated] = False  # remove already calculated
        print('Calculating %d structure factors' % sum(this_calculation))
        if sum(this_calculation) > 0:
            self.intensity[this_calculation] = self.xtl.Scatter.intensity(self.hkl[this_calculation])
            self._calculated[this_calculation] = True  # Don't calcualte these again

    def latt_str(self):
        """Generate list of reflections"""
        out = '%s\n' % self.xtl.name
        out += u'Wavelength: %.4f \u212B\n' % self.wavelength_a
        out += 'NRefs: %d\n' % len(self.hkl)
        out += u'hkl               2\u03B8      Intensity\n'
        fmt = '%12s %10.2f   %.2f'
        idx = np.argsort(self.tth)
        out += '\n'.join([fmt % (self.hkl[n], self.tth[n], self.intensity[n]) for n in idx])
        return out


class Detector:
    """
    Detector class

     distance: distance to detector in meters
     normal_dir: (dx, dy, dz) detector normal
     x_size: detector width along x-axis in meters
     z_size: detector height along z-axis in meters

    assumes incident vector (0,1,0)
    rotate delta (about z), gamma (about x)
    """
    position = np.array([0, 0, 0], dtype=float)
    normal_dir = np.array([0, -1, 0], dtype=float)
    x_axis = np.array([1, 0, 0], dtype=float)
    z_axis = np.array([0, 0, 1], dtype=float)
    _current_delta = 0
    _current_gamma = 0
    update_plots = default_fun
    update_widgets = default_fun
    generate_detector = default_fun

    def __init__(self, distance=1., normal_dir=(0, -1., 0), x_size=1., z_size=1.,
                 pixel_size=0.01, labmatrix=np.eye(3)):
        self._distance = distance
        self._initial_position = distance * np.array([0, 1., 0])
        self._initial_normal_dir = fg.norm(normal_dir)
        self._x_size = x_size
        self._z_size = z_size
        self._initial_x_axis = x_size * fg.norm(np.cross(self.normal_dir, (0, 0, 1.)))
        self._initial_z_axis = z_size * fg.norm(np.cross(self.x_axis, self.normal_dir))
        self._pixel_size = pixel_size
        self._lab = labmatrix
        self.rotate_to(0, 0)

    def __repr__(self):
        s = "Detector(delta=%.5g, gamma=%.5g)"
        return s % (self._current_delta, self._current_gamma)

    def set_detector(self, distance=1., normal_dir=(0, -1., 0), x_size=1., z_size=1.,
                     pixel_size=0.01):
        self._distance = distance
        self._initial_position = distance * np.array([0, 1., 0])
        self._initial_normal_dir = fg.norm(normal_dir)
        self._x_size = x_size
        self._z_size = z_size
        self._initial_x_axis = x_size * fg.norm(np.cross(self.normal_dir, (0, 0, 1.)))
        self._initial_z_axis = z_size * fg.norm(np.cross(self.x_axis, self.normal_dir))
        self._pixel_size = pixel_size
        self.rotate_to(self._current_delta, self._current_gamma)

    def set_labframe(self, labmatrix):
        self._lab = labmatrix
        self.rotate_to(self._current_delta, self._current_gamma)

    def set_callbacks(self, generate_detector=None, update_widgets=None, update_plots=None):
        if generate_detector:
            self.generate_detector = generate_detector
        if update_widgets:
            self.update_widgets = update_widgets
        if update_plots:
            self.update_plots = update_plots

    def rotate_to(self, delta=0., gamma=0.):
        """
        Rotate detector position
        :param delta: float angle in degrees in vertical direction (about diff-z)
        :param gamma: float angle in degrees in horizontal direction (about diff-x)
        :return: None
        """
        D = fc.rotmatrixz(-delta)  # left handed
        G = fc.rotmatrixx(gamma)
        R = np.dot(G, D)
        self.position = fc.labvector(self._initial_position, R=R, LAB=self._lab)
        self.normal_dir = fc.labvector(self._initial_normal_dir, R=R, LAB=self._lab)
        self.x_axis = fc.labvector(self._initial_x_axis, R=R, LAB=self._lab)
        self.z_axis = fc.labvector(self._initial_z_axis, R=R, LAB=self._lab)
        self._current_delta = delta
        self._current_gamma = gamma

    def corners(self):
        """Returns coordinates of 4 corners"""
        c = [
            self.position + self.x_axis / 2 + self.z_axis / 2,
            self.position + self.x_axis / 2 - self.z_axis / 2,
            self.position - self.x_axis / 2 - self.z_axis / 2,
            self.position - self.x_axis / 2 + self.z_axis / 2,
            self.position + self.x_axis / 2 + self.z_axis / 2,
        ]
        return np.array(c)

    def q_shape(self, wavelength_a):
        """Return line in reciprocal space tracing the borders of the detector"""
        c = [self.position + self.x_axis / 2 + self.z_axis / 2]
        c += [self.position + self.x_axis / 2 + self.z_axis / 2 - n * 0.01 * self.z_axis for n in range(100)]
        c += [self.position + self.x_axis / 2 - n * 0.01 * self.x_axis - self.z_axis / 2 for n in range(100)]
        c += [self.position - self.x_axis / 2 - self.z_axis / 2 + n * 0.01 * self.z_axis for n in range(100)]
        c += [self.position - self.x_axis / 2 + n * 0.01 * self.x_axis + self.z_axis / 2 for n in range(100)]
        c += [self.position + self.x_axis / 2 + self.z_axis / 2]
        wv = fc.wavevector(wavelength=wavelength_a)
        kf = wv * fg.norm(c)
        return kf - wv * fc.labvector([0, 1, 0], LAB=self._lab)

    def ki(self, wavelength_a):
        wv = fc.wavevector(wavelength=wavelength_a)
        return wv * fc.labvector([0, 1, 0], LAB=self._lab)

    def kfkiq(self, wavelength_a):
        """Return lines traving kf, ki and Q"""
        wv = fc.wavevector(wavelength=wavelength_a)
        ki = wv * fc.labvector([0, 1, 0], LAB=self._lab)
        kf = wv * fg.norm(self.position)
        kikf = np.array([-ki, [0, 0, 0], kf])
        qq = np.array([[0, 0, 0], kf - ki])
        return kikf, qq

    def kfkiq_corners(self, wavelength_a):
        """Return lines to corners of kif, ki and Q"""
        wv = fc.wavevector(wavelength=wavelength_a)
        ki = wv * fc.labvector([0, 1, 0], LAB=self._lab)
        kf = wv * fg.norm(self.corners())
        z = [0, 0, 0]
        kikf = np.vstack([-ki, z, kf[0], kf[1], z, kf[1],  kf[2], z, kf[2], kf[3], z, kf[3], kf[0]])
        qq = np.array([
            z, kf[0] - ki,
            z, kf[1] - ki,
            z, kf[2] - ki,
            z, kf[3] - ki,
        ])
        return kikf, qq

    def twotheta(self):
        """Return detector angle in degrees"""
        incident = fc.labvector(self._initial_position, LAB=self._lab)
        return fg.vectors_angle_degrees(incident, self.position)

    def width_lim(self):
        return [-1000 * self._x_size / 2, 1000 * self._x_size / 2]

    def height_lim(self):
        return [-1000 * self._z_size / 2, 1000 * self._z_size / 2]

    def peak_width_mm(self, delta_theta):
        """
        Calculate peak width on detector in mm
        :param delta_theta: float or array, peak width in degrees
        :return: FWHM in mm
        """
        distance_mm = self._distance * 1000.
        return distance_mm * np.tan(np.deg2rad(delta_theta))

    def relative_position(self, vectors):
        """Return a position coordinate relative to detector centre"""
        return fg.index_coordinates(np.subtract(vectors, self.position), [self.x_axis, self.z_axis, self.normal_dir])

    def check_vector_incident(self, vectors):
        """Check a vector is in the direction of the detector"""
        vec = np.asarray(vectors, dtype=float).reshape([-1, 3])
        corner_angle = max(abs(fg.vectors_angle_degrees(self.position, self.corners())))
        vec_angles = abs(fg.vectors_angle_degrees(self.position, vec))
        return vec_angles < corner_angle

    def incident_position(self, vectors):
        """Return position of vector incident on detector"""
        directions = fg.norm(vectors).reshape([-1, 3])
        check = self.check_vector_incident(directions)
        ixyz = np.nan * np.zeros([len(directions), 3])
        for n in np.flatnonzero(check):
            ixyz[n] = fg.plane_intersection((0, 0, 0), directions[n], self.position, self.normal_dir)
        return ixyz

    def reflection_position(self, qxqyqz, wavelength_a):
        """
        Return relative position of reflection on detector
        :param qxqyqz: [nx3] array of reciprocal lattice vectors q = h.a* + k.b* + l.c*
        :param wavelength_a: incident beam wavelength in Angstroms
        :return ixyz: [nx3] array of positions on the detector
        :return uvw: [nx3] array of positions on the detector, relative to detector centre, or NaN if not incident
        :return diff: [nx1] array of wavevector difference in A-1, where 0 is the condition for diffraction.
        """
        q = np.reshape(qxqyqz, [-1, 3])
        ki = fc.wavevector(wavelength=wavelength_a) * fc.labvector([0, 1, 0], LAB=self._lab)
        kf = q + ki
        ixyz = self.incident_position(kf)
        iuvw = self.relative_position(ixyz)
        iuvw[np.any(abs(iuvw) > 0.5, axis=1)] = [np.nan, np.nan, np.nan]
        diff = fc.wavevector_difference(q, ki)  # difference in A-1
        return ixyz, iuvw, diff

    def detector_image(self, iuvw, intensity, peak_width=0.01, background=0):
        """
        Generate detector image
        :param iuvw: [nx3] array of relative positions on detector
        :param intensity: [nx1] array of intensity values
        :param peak_width: float or [nx1], peak FWHM, in m
        :param background: float, background level
        :return: xx, yy, mesh, peak_x[m], peak_y[m]
        """
        peak_x = self._x_size * iuvw[:, 0]
        peak_y = self._z_size * iuvw[:, 1]
        pixels_width = int(self._x_size / self._pixel_size) + 1

        xx, yy, mesh = fc.peaks_on_plane(
            peak_x=peak_x,
            peak_y=peak_y,
            peak_height=intensity,
            peak_width=peak_width,
            max_x=self._x_size / 2,
            max_y=self._z_size / 2,
            pixels_width=pixels_width,
            background=background
        )
        return xx, yy, mesh, peak_x, peak_y


def tk_beam(parent, xtl, callback):
    """radiation tkinter widgets, returns Latt"""
    # Lattice
    latt = Lattice(xtl)

    # tk variables
    radiation_type = tk.StringVar(parent, 'X-Ray')
    check_magnetic = tk.BooleanVar(parent, False)
    wavelength_type = tk.StringVar(parent, 'Energy [keV]')
    _prev_wavelength_type = tk.StringVar(parent, 'Energy [keV]')
    wavelength_val = tk.DoubleVar(parent, DEFAULT_ENERGY)
    edge = tk.StringVar(parent, 'Edge')
    max_gen_type = tk.StringVar(parent, u'Max Q [\u212B\u207B\u00B9]')
    _prev_max_gen_type = tk.StringVar(parent, u'Max Q [\u212B\u207B\u00B9]')
    max_val = tk.DoubleVar(parent, DEFAULT_MAXQ)
    res_val = tk.DoubleVar(parent, DEFAULT_RESOLUTION)
    res_unit = tk.StringVar(parent, 'eV')
    _prev_res_unit = tk.StringVar(parent, 'eV')
    peak_width = tk.DoubleVar(parent, DEFAULT_PEAKWIDTH)
    peak_width_type = tk.StringVar(parent, 'Peak width [Deg]')
    _prev_peak_width_type = tk.StringVar(parent, 'Peak width [Deg]')
    check_precalculate = tk.BooleanVar(parent, True)

    # radiation
    radiations = ['X-Ray', 'Neutron', 'Electron']
    wavelength_types = ['Energy [keV]', 'Energy [eV]', 'Energy [meV]', u'Wavelength [\u212B]', 'Wavelength [nm]']
    res_units = ['eV', 'meV', 'Deg', u'\u212B\u207B\u00B9', u'\u212B']
    max_types = [u'Max Q [\u212B\u207B\u00B9]', u'Max 2\u03B8 [Deg]', u'min d [\u212B]']
    peak_width_types = ['Peak width [Deg]', u'Domain size [\u212B]', u'Peak width [\u212B\u207B\u00B9]']

    # X-ray edges:
    xr_edges, xr_energies = xtl.Properties.xray_edges()
    xr_edges.insert(0, 'Cu Ka')
    xr_edges.insert(1, 'Mo Ka')
    xr_energies.insert(0, fg.Cu)
    xr_energies.insert(1, fg.Mo)

    # Functions
    def get_wavelength():
        """Return wavelength in A according to unit"""
        val = wavelength_val.get()
        rad = radiation_type.get()
        unit = _prev_wavelength_type.get()

        if unit == 'Energy [keV]':
            if rad == 'Electron':
                wavelength_a = fc.electron_wavelength(val * 1000)
            elif rad == 'Neutron':
                wavelength_a = fc.neutron_wavelength(val * 1e6)
            else:
                wavelength_a = fc.energy2wave(val)
        elif unit == 'Energy [meV]':
            if rad == 'Electron':
                wavelength_a = fc.electron_wavelength(val / 1000.)
            elif rad == 'Neutron':
                wavelength_a = fc.neutron_wavelength(val)
            else:
                wavelength_a = fc.energy2wave(val / 1.0e6)
        elif unit == 'Energy [eV]':
            if rad == 'Electron':
                wavelength_a = fc.electron_wavelength(val)
            elif rad == 'Neutron':
                wavelength_a = fc.neutron_wavelength(val * 1000)
            else:
                wavelength_a = fc.energy2wave(val / 1000.)
        elif unit == 'Wavelength [nm]':
            wavelength_a = val / 10.
        else:
            wavelength_a = val
        return wavelength_a

    def set_wavelength(wavelength_a):
        """set wavelength according to unit"""
        rad = radiation_type.get()
        unit = wavelength_type.get()

        if unit == 'Energy [keV]':
            if rad == 'Electron':
                val = fc.electron_energy(wavelength_a) / 1000.
            elif rad == 'Neutron':
                val = fc.neutron_energy(wavelength_a) / 1.0e6
            else:
                val = fc.wave2energy(wavelength_a)
        elif unit == 'Energy [meV]':
            if rad == 'Electron':
                val = fc.electron_energy(wavelength_a) * 1000
            elif rad == 'Neutron':
                val = fc.neutron_energy(wavelength_a)
            else:
                val = fc.wave2energy(wavelength_a) / 1.0e6
        elif unit == 'Energy [eV]':
            if rad == 'Electron':
                val = fc.electron_energy(wavelength_a)
            elif rad == 'Neutron':
                val = fc.neutron_energy(wavelength_a) * 1000
            else:
                val = fc.wave2energy(wavelength_a) / 1000.
        elif unit == 'Wavelength [nm]':
            val = wavelength_a * 10.,
        else:
            val = wavelength_a
        wavelength_val.set(round(val, 4))
        _prev_wavelength_type.set(unit)
        # Set max Q
        max_q = fc.calqmag(180, wavelength_a=wavelength_a)
        max_gen = max_gen_type.get()
        if max_gen == u'Max Q [\u212B\u207B\u00B9]':
            max_val.set(round(max_q, 4))
        # elif max_gen == u'Max 2\u03B8 [Deg]':
        #     energy_kev = fc.wave2energy(wavelength_a)
        #     max_val.set(round(fc.cal2theta(max_q, energy_kev), 4))
        elif max_gen == u'min d [\u212B]':
            max_val.set(round(fc.q2dspace(max_q), 4))

    def fun_radiation(event=None):
        """Set radiation"""
        rad = radiation_type.get()
        wavelength_a = get_wavelength()
        if rad == 'Neutron':
            wavelength_type.set('Energy [meV]')
        elif rad == 'Electron':
            wavelength_type.set('Energy [eV]')
            check_magnetic.set(False)
        else:
            wavelength_type.set('Energy [keV]')
        set_wavelength(wavelength_a)

    def fun_wavelength(event=None):
        """Convert previous unit"""
        wavelength_a = get_wavelength()
        set_wavelength(wavelength_a)

    def fun_edge(event=None):
        """X-ray edge option menu"""
        edge_name = edge.get()
        if edge_name in xr_edges:
            idx = xr_edges.index(edge_name)
            set_wavelength(fc.energy2wave(xr_energies[idx]))

    def get_max_q(event=None):
        """Return max val in inverse angstroms, convert if changed"""
        val = max_val.get()
        old_max_gen = _prev_max_gen_type.get()
        max_gen = max_gen_type.get()
        if old_max_gen == u'Max Q [\u212B\u207B\u00B9]':
            max_q = val
        elif old_max_gen == u'Max 2\u03B8 [Deg]':
            wavelength_a = get_wavelength()
            max_q = fc.calqmag(twotheta=val, wavelength_a=wavelength_a)
        else:  # max_gen == u'min d [\u212B]'
            max_q = fc.dspace2q(val)
        # Convert if changed
        if max_gen != old_max_gen:
            if max_gen == u'Max Q [\u212B\u207B\u00B9]':
                max_val.set(round(max_q, 4))
            elif max_gen == u'Max 2\u03B8 [Deg]':
                wavelength_a = get_wavelength()
                tth = fc.cal2theta(max_q, wavelength_a=wavelength_a)
                tth = 180. if np.isnan(tth) else tth
                max_val.set(round(tth, 4))
            else:  # max_gen == u'min d [\u212B]'
                max_val.set(round(fc.q2dspace(max_q), 4))
            _prev_max_gen_type.set(max_gen)
        return max_q

    def get_domain_size(event=None):
        """Return domain size in angstroms"""
        val = peak_width.get()
        old_type = _prev_peak_width_type.get()
        new_type = peak_width_type.get()
        if old_type == 'Peak width [Deg]':
            wavelength_a = get_wavelength()
            size_a = fc.scherrer_size(val, twotheta=45, wavelength_a=wavelength_a)
        elif old_type == u'Domain size [\u212B]':
            size_a = val
        else:  # u'Peak width [\u212B\u207B\u00B9]'
            size_a = fc.q2dspace(val)
        # Convert if changed
        if new_type != old_type:
            if new_type == 'Peak width [Deg]':
                wavelength_a = get_wavelength()
                fwhm = fc.scherrer_fwhm(size_a, twotheta=45, wavelength_a=wavelength_a)
                peak_width.set(round(fwhm, 4))
            elif new_type == u'Domain size [\u212B]':
                peak_width.set(round(size_a, 2))
            else:  # u'Peak width [\u212B\u207B\u00B9]'
                peak_width.set(round(fc.dspace2q(size_a), 4))
            _prev_peak_width_type.set(new_type)
        return size_a

    def get_resolution(event=None):
        """Return resolution in inverse Angstroms"""
        val = res_val.get()
        old_unit = _prev_res_unit.get()
        new_unit = res_unit.get()
        if old_unit == 'eV':
            res = fc.wavevector(val / 1000.)
        elif old_unit == 'meV':  # neutron
            res = fc.wavevector(wavelength=fc.neutron_wavelength(val))
        elif old_unit == 'Deg':
            wavelength = get_wavelength()
            res = fc.calqmag(val, wavelength_a=wavelength)
        elif old_unit == u'\u212B':  # Angstroms
            res = fc.dspace2q(val)
        else:  # inverse angstroms
            res = val
        # Convert if changed
        if old_unit != new_unit:
            dspace = fc.q2dspace(res)
            if new_unit == 'eV':
                res_val.set(round(fc.wave2energy(dspace) * 1000, 4))
            elif new_unit == 'meV':  # neutron
                res_val.set(round(fc.neutron_energy(dspace), 4))
            elif new_unit == 'Deg':
                wavelength_a = get_wavelength()
                res_val.set(round(fc.cal2theta(res, fc.wave2energy(wavelength_a)), 4))
            elif new_unit == u'\u212B':  # Angstroms
                res_val.set(round(dspace, 4))
            else:  # inverse angstroms
                res_val.set(round(res, 4))
            _prev_res_unit.set(new_unit)
        return res

    def help_max_q():
        msg = "Calculate reflection list upto this value\n  (lower angle is less reflections)."
        messagebox.showinfo(
            parent=parent,
            title='max-Q',
            message=msg
        )

    def help_peak_width():
        msg = "Sets the peak width (full-width at half-maximum) of the crystallite.\n"
        msg += 'The domain size can also be used in which the peak with is calculted '
        msg += 'from the Scherrer equation.'
        messagebox.showinfo(
            parent=parent,
            title='Peak Width',
            message=msg
        )

    def help_resolution():
        msg = "Sets the source/ incident beam resolution.\n"
        msg += "This is used to determine how many reflections are exicted by the incient beam.\n"
        msg += "  A large resolution (high value in eV) will produce many broad reflections\n"
        msg += "  A small resolution (low value in eV) will produce few very sharm reflections"
        messagebox.showinfo(
            parent=parent,
            title='Resolution',
            message=msg
        )

    def list_refs():
        """Show reflections"""
        s = latt.latt_str()
        StringViewer(s, 'Lattice Reflections', width=60)

    def fun_gen_refs(event=None):
        """Generate reflections + update scattering options"""
        wavelength_a = get_wavelength()
        resolution = get_resolution()
        cwv = np.ceil(fc.wavevector(wavelength=wavelength_a))
        max_q = min(get_max_q(), fc.calqmag(180., wavelength_a=wavelength_a))
        print(max_q)
        OPTIONS['maxq'] = cwv

        latt.resolution = resolution
        latt.domain_size = get_domain_size()
        latt.generate_hkl(wavelength_a=wavelength_a, max_q=max_q)

        # Calculation type
        radiation = radiation_type.get()
        magnetic = check_magnetic.get()
        if radiation == 'X-Ray':
            if magnetic:
                scattering_type = 'xray magnetic'
            else:
                scattering_type = 'xray'
        elif radiation == 'Neutron':
            if magnetic:
                scattering_type = 'neutron magnetic'
            else:
                scattering_type = 'neutron'
        else:
            scattering_type = 'electron'
        latt.xtl.Scatter.setup_scatter(scattering_type=scattering_type, output=False)

        if check_precalculate.get():
            latt.generate_intensities()

        latt.update_plots()

    def update_widgets(radiation=None, wavelength_a=None, max_q=None, resolution=None, peakwidth_deg=None):
        """Update tk widget values"""

        if radiation in radiations:
            radiation_type.set(radiation)
        if wavelength_a is not None:
            set_wavelength(wavelength_a)
        if max_q is not None:
            max_val.set(max_q)
            max_gen_type.set(max_types[0])
            _prev_max_gen_type.set(max_types[0])
        if resolution is not None:
            res_val.set(resolution)
            res_unit.set(res_units[-2])
            _prev_res_unit.set(res_units[-2])
        if peakwidth_deg is not None:
            peak_width.set(peakwidth_deg)
            peak_width_type.set(peak_width_types[0])
            _prev_peak_width_type.set(peak_width_types[0])

    # TK widgets
    OPT_WIDTH = 14
    frm = tk.LabelFrame(parent, text='Beam', relief=tk.RIDGE)
    frm.pack(fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)

    # Radiation
    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Radiation:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.OptionMenu(ln, radiation_type, *radiations, command=fun_radiation)
    var.config(font=SF, width=OPT_WIDTH, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var = tk.Checkbutton(ln, text='Magnetic', variable=check_magnetic, font=SF)
    var.pack(side=tk.LEFT, padx=6)

    # Wavelength / Energy
    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.OptionMenu(ln, wavelength_type, *wavelength_types, command=fun_wavelength)
    var.config(font=SF, width=OPT_WIDTH, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var = tk.OptionMenu(ln, edge, *xr_edges, command=fun_edge)
    var.config(font=SF, width=5, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=wavelength_val, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_wavelength)
    var.bind('<KP_Enter>', fun_wavelength)

    # Max Q
    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.OptionMenu(ln, max_gen_type, *max_types, command=get_max_q)
    var.config(font=SF, width=OPT_WIDTH, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=max_val, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_gen_refs)
    var.bind('<KP_Enter>', fun_gen_refs)
    var = tk.Button(ln, text='?', font=TF, command=help_max_q, bg=btn, activebackground=btn_active)
    var.pack(side=tk.LEFT, pady=2)

    # Peak width
    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.OptionMenu(ln, peak_width_type, *peak_width_types, command=get_domain_size)
    var.config(font=SF, width=OPT_WIDTH, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=peak_width, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_gen_refs)
    var.bind('<KP_Enter>', fun_gen_refs)
    var = tk.Button(ln, text='?', font=TF, command=help_peak_width, bg=btn, activebackground=btn_active)
    var.pack(side=tk.LEFT, pady=2)

    # Resolution
    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Resolution:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=res_val, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var = tk.OptionMenu(ln, res_unit, *res_units, command=get_resolution)
    var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_gen_refs)
    var.bind('<KP_Enter>', fun_gen_refs)
    var = tk.Button(ln, text='?', font=TF, command=help_resolution, bg=btn, activebackground=btn_active)
    var.pack(side=tk.LEFT, pady=2)

    # Update Button
    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Checkbutton(ln, text='pre-calculate intensities', variable=check_precalculate, font=SF)
    var.pack(side=tk.LEFT, padx=2)
    var = tk.Button(ln, text='Gen Refs', font=TF, command=fun_gen_refs, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.RIGHT, pady=2)
    var = tk.Button(ln, text='List Refs', font=TF, command=list_refs, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.RIGHT, pady=2)

    # Add functions
    latt.set_callbacks(
        generate_refs=fun_gen_refs,
        update_widgets=update_widgets,
        update_plots=callback,
        get_wavelength=get_wavelength,
    )
    return latt


def tk_detector(parent, callback):
    """Create detector options widget"""
    det_distance = tk.DoubleVar(parent, DEFAULT_DET_DISTANCE)
    det_width = tk.DoubleVar(parent, DEFAULT_DET_WIDTH)
    det_height = tk.DoubleVar(parent, DEFAULT_DET_HEIGHT)
    det_pixel_size = tk.DoubleVar(parent, DEFAULT_DET_PIXEL)

    det = Detector()

    def fun_update_det(event=None):
        """Update detector"""
        det.set_detector(
            distance=det_distance.get() / 1000.,
            x_size=det_width.get() / 1000.,
            z_size=det_height.get() / 1000.,
            pixel_size=det_pixel_size.get() / 1000.,
        )
        det.update_plots()  # calls plot update, assigned at end of DiffractometerGui.__init__

    def fun_list_refs():
        """List reflections"""
        s = reflist_str()
        StringViewer(s, 'Detector Reflections', width=60)

    def update_widgets(distance=None, width=None, height=None, pixel_size=None):
        """Update widgets"""
        if distance is not None:
            det_distance.set(distance)
        if width is not None:
            det_width.set(width)
        if height is not None:
            det_height.set(height)
        if pixel_size is not None:
            det_pixel_size.set(pixel_size)

    frm = tk.LabelFrame(parent, text='Detector', relief=tk.RIDGE)
    frm.pack(fill=tk.X, padx=2, pady=2)

    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
    var = tk.Label(ln, text='Distance:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=det_distance, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_update_det)
    var.bind('<KP_Enter>', fun_update_det)
    var = tk.Label(ln, text='mm', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Button(ln, text='Update', font=TF, command=fun_update_det, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.RIGHT, pady=2)

    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Width:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=det_width, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_update_det)
    var.bind('<KP_Enter>', fun_update_det)
    var = tk.Label(ln, text='mm', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Button(ln, text='List Refs.', font=TF, command=fun_list_refs, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.RIGHT, pady=2)

    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Height:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=det_height, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_update_det)
    var.bind('<KP_Enter>', fun_update_det)
    var = tk.Label(ln, text='mm', font=SF)
    var.pack(side=tk.LEFT)

    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Pixel size:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=det_pixel_size, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', fun_update_det)
    var.bind('<KP_Enter>', fun_update_det)
    var = tk.Label(ln, text='mm', font=SF)
    var.pack(side=tk.LEFT)

    # Add functions to det
    det.set_callbacks(
        generate_detector=fun_update_det,
        update_widgets=update_widgets,
        update_plots=callback
    )
    return det


def tk_angle_element(name, parent, callback, help_message='', val_range=(-180, 180), initial=0):
    """Slider element"""
    line = tk.Frame(parent)
    line.pack(fill=tk.X, pady=0)
    variable = tk.DoubleVar(line, initial)

    def button():
        msg = "Diffractometer angle: %s\n" % name
        msg += help_message
        messagebox.showinfo(
            parent=parent,
            title=name,
            message=msg
        )

    def inc():
        variable.set(variable.get() + 1)
        callback()

    def dec():
        variable.set(variable.get() - 1)
        callback()

    var = tk.Label(line, text=name, font=TF, width=8)
    var.pack(side=tk.LEFT)
    var = tk.Button(line, text='?', command=button)
    var.pack(side=tk.LEFT)
    var = tk.Button(line, text='-', command=dec)
    var.pack(side=tk.LEFT)
    var = tk.Scale(line, from_=val_range[0], to=val_range[1], variable=variable, font=BF,
                   sliderlength=30, orient=tk.HORIZONTAL, command=callback, showvalue=False,
                   repeatdelay=300, resolution=0.05, length=300)
    # var.bind("<ButtonRelease-1>", callback)
    var.pack(side=tk.LEFT, expand=tk.YES)
    var = tk.Button(line, text='+', command=inc)
    var.pack(side=tk.LEFT)
    var = tk.Entry(line, textvariable=variable, font=TF, width=6, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', callback)
    var.bind('<KP_Enter>', callback)
    return variable


def tk_hkl(parent, latt: Lattice, tth_axis: tk.DoubleVar, th_axis: tk.DoubleVar, callback):
    """reflection frame"""
    tth_fmt = u'2\u03B8 = %6.2f, \u03B8 = %6.2f Deg'
    hkl_str = tk.StringVar(parent, '0 0 2')
    tth_str = tk.StringVar(parent, tth_fmt % (0, 0))

    def update_tth(event=None):
        wavelength_a = latt.get_wavelength()
        hkl = fg.str2array(hkl_str.get())
        tth = latt.xtl.Cell.tth(hkl, wavelength_a=wavelength_a)[0]
        tth_str.set(tth_fmt % (tth, tth/2))

    def update_angles(event=None):
        wavelength_a = latt.get_wavelength()
        hkl = fg.str2array(hkl_str.get())
        tth = latt.xtl.Cell.tth(hkl, wavelength_a=wavelength_a)[0]
        tth_str.set(tth_fmt % (tth, tth / 2))
        tth_axis.set(tth)
        th_axis.set(tth / 2)
        callback()

    frm = tk.LabelFrame(parent, text='hkl', relief=tk.RIDGE)
    frm.pack(fill=tk.X, padx=2, pady=2)

    ln = tk.Frame(frm)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='hkl:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=hkl_str, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', update_tth)
    var.bind('<KP_Enter>', update_tth)
    var = tk.Label(ln, textvariable=tth_str, font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Button(ln, text='Set', font=TF, command=update_angles, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.LEFT, pady=2)


def tk_scan(parent, angle_dict, latt, det):
    default_angle = next(iter(angle_dict))
    angle_dict['Mu.2Gamma'] = None
    angle_dict['Eta.2Delta'] = None
    scan_angle = tk.StringVar(parent, default_angle)
    first = tk.DoubleVar(parent, 0)
    last = tk.DoubleVar(parent, 10)
    step = tk.DoubleVar(parent, 1)
    npoints = tk.IntVar(parent, 11)

    def update_points(event=None):
        first_val = first.get()
        last_val = last.get()
        step_val = step.get()
        npoints_val = (last_val - first_val) // step_val + 1
        npoints.set(int(npoints_val))

    def update_step(event=None):
        first_val = first.get()
        last_val = last.get()
        npoints_val = npoints.get()
        step_val = (last_val - first_val) / npoints_val
        step.set(step_val)

    def create_scan():
        first_val = first.get()
        last_val = last.get()
        step_val = step.get()
        scan_range = np.arange(first_val, last_val + step_val, step_val)
        angle_name = scan_angle.get()

        current_latt_angles = {
            'phi': angle_dict['Phi'].get(),
            'chi': angle_dict['Chi'].get(),
            'eta': angle_dict['Eta'].get(),
            'mu': angle_dict['Mu'].get(),
        }
        current_det_angles = {
            'delta': angle_dict['Delta'].get(),
            'gamma': angle_dict['Gamma'].get()
        }

        scan_sum = np.zeros_like(scan_range)
        for n, angle in enumerate(scan_range):
            # Rotate lattice + detector
            if angle_name.lower() in current_latt_angles:
                current_latt_angles[angle_name.lower()] = angle
            elif angle_name.lower() in current_det_angles:
                current_det_angles[angle_name.lower()] = angle
            elif angle_name == 'Mu.2Gamma':
                current_latt_angles['mu'] = angle / 2.
                current_det_angles['gamma'] = angle
            elif angle_name == 'Eta.2Delta':
                current_latt_angles['eta'] = angle / 2.
                current_det_angles['delta'] = angle
            latt.rotate_to(**current_latt_angles)
            det.rotate_to(**current_det_angles)

            xx, yy, mesh = generate_reflections(
                latt=latt,
                det=det,
                minimum_gaussian_intensity=0.01
            )
            scan_sum[n] = np.sum(mesh)

        # Create new frame
        newframe = tk.Toplevel(parent)
        newframe.title('%s Scan' % angle_name)
        fig = Figure(figsize=FIGURE_3D_SIZE, dpi=OPTIONS['fig_dpi'])
        canvas = FigureCanvasTkAgg(fig, newframe)
        canvas.get_tk_widget().configure(bg='black')
        canvas.draw()

        # Plot data
        ax = fig.add_subplot(111)
        ax.plot(scan_range, scan_sum)
        ax.set_xlabel('%s [Deg]' % angle_name)
        ax.set_ylabel('Detector Sum')
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

    frm = tk.LabelFrame(parent, text='Scan', relief=tk.RIDGE)
    frm.pack(fill=tk.X, padx=2, pady=2)

    var = tk.OptionMenu(frm, scan_angle, *angle_dict.keys())
    var.config(font=SF, width=10, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)

    sec = tk.Frame(frm)
    sec.pack(side=tk.LEFT)
    ln = tk.Frame(sec)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Start:', font=SF, width=6)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=first, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', update_points)
    var.bind('<KP_Enter>', update_points)

    var = tk.Label(ln, text='End:', font=SF, width=6)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=last, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', update_points)
    var.bind('<KP_Enter>', update_points)

    ln = tk.Frame(sec)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Step:', font=SF, width=6)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=step, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', update_points)
    var.bind('<KP_Enter>', update_points)

    var = tk.Label(ln, text='Points:', font=SF, width=6)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=npoints, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var.bind('<Return>', update_step)
    var.bind('<KP_Enter>', update_step)

    var = tk.Button(frm, text='Scan', font=TF, command=create_scan, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.LEFT, pady=2)


def tk_plot_options(parent, callback):
    """Plot options window"""

    # all_colormaps = plt.colormaps()
    all_colormaps = ['viridis', 'Spectral', 'plasma', 'inferno', 'Greys', 'Blues', 'winter', 'autumn',
                     'hot', 'hot_r', 'hsv', 'rainbow', 'jet']

    root = tk.Toplevel(parent)
    root.title('Plot Options')

    pl_lattice = tk.BooleanVar(root, OPTIONS['3d_lattice'])
    pl_basis = tk.BooleanVar(root, OPTIONS['3d_basis'])
    pl_det_corners = tk.BooleanVar(root, OPTIONS['3d_detector_corners'])
    pl_ref_arrows = tk.BooleanVar(root, OPTIONS['3d_ref_arrows'])
    pl_labels = tk.BooleanVar(root, OPTIONS['det_labels'])
    pl_log = tk.BooleanVar(root, OPTIONS['det_log'])
    clim_min = tk.DoubleVar(root, OPTIONS['det_clim'][0])
    clim_max = tk.DoubleVar(root, OPTIONS['det_clim'][1])
    cmap = tk.StringVar(root, OPTIONS['det_cmap'])
    dpi = tk.IntVar(root, OPTIONS['fig_dpi'])
    maxq = tk.DoubleVar(root, OPTIONS['maxq'])
    mininten = tk.DoubleVar(root, OPTIONS['min_intensity'])

    def fun_log(event=None):
        log = pl_log.get()
        cmin = clim_min.get()
        cmax = clim_max.get()
        if log:
            clim_min.set(round(np.log10(cmin+1), 3))
            clim_max.set(round(np.log10(cmax), 3))
        else:
            clim_min.set(round(10 ** cmin - 1, 3))
            clim_max.set(round(10 ** cmax, 3))

    def fun_update():
        OPTIONS['3d_lattice'] = pl_lattice.get()
        OPTIONS['3d_basis'] = pl_basis.get()
        OPTIONS['3d_detector_corners'] = pl_det_corners.get()
        OPTIONS['3d_ref_arrows'] = pl_ref_arrows.get()
        OPTIONS['det_labels'] = pl_labels.get()
        OPTIONS['det_log'] = pl_log.get()
        OPTIONS['det_clim'] = [clim_min.get(), clim_max.get()]
        OPTIONS['det_cmap'] = cmap.get()
        OPTIONS['fig_dpi'] = dpi.get()
        OPTIONS['maxq'] = maxq.get()
        OPTIONS['min_intensity'] = mininten.get()
        callback()
        root.destroy()

    ln = tk.Frame(root)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Checkbutton(ln, text='Plot 3D Lattice', variable=pl_lattice, font=SF)
    var.pack(side=tk.LEFT, padx=3)
    var = tk.Checkbutton(ln, text='Plot 3D Basis', variable=pl_basis, font=SF)
    var.pack(side=tk.LEFT, padx=3)
    var = tk.Checkbutton(ln, text='Plot 3D corners', variable=pl_det_corners, font=SF)
    var.pack(side=tk.LEFT, padx=3)

    ln = tk.Frame(root)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Checkbutton(ln, text='Detector HKL labels', variable=pl_labels, font=SF)
    var.pack(side=tk.LEFT, padx=3)
    var = tk.Checkbutton(ln, text='Plot ref arrows', variable=pl_ref_arrows, font=SF)
    var.pack(side=tk.LEFT, padx=3)

    ln = tk.Frame(root)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Detector Clim:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=clim_min, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=clim_max, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)

    ln = tk.Frame(root)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.OptionMenu(ln, cmap, *all_colormaps)
    var.config(font=SF, width=16, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var = tk.Checkbutton(ln, text='Log', variable=pl_log, font=SF, command=fun_log)
    var.pack(side=tk.LEFT, padx=1)

    ln = tk.Frame(root)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='Figure DPI:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=dpi, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var = tk.Label(ln, text='3D Plot Max:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=maxq, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var = tk.Label(ln, text='Min Intensity:', font=SF)
    var.pack(side=tk.LEFT)
    var = tk.Entry(ln, textvariable=mininten, font=TF, width=8, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)

    ln = tk.Frame(root)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Button(ln, text='Update', font=TF, command=fun_update, bg=btn,
                    activebackground=btn_active)
    var.pack(pady=2)


def tk_orientation(parent, latt: Lattice, det: Detector, callback):
    """Orientation options window"""
    root = tk.Toplevel(parent)
    root.title('Orientation Options')

    ustr = str(latt._u.tolist()).replace('], [', '\n').replace('[', '').replace(']', '')
    lstr = str(latt._lab.tolist()).replace('], [', '\n').replace('[', '').replace(']', '')

    right = tk.Frame(root, relief=tk.RAISED)
    right.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.YES, padx=6)

    # Tranformation
    def fun_transform(event=None):
        mat = transform_optoins[transform.get()]
        for m in range(3):
            for n in range(3):
                matrix_vars[m][n].set(mat[m, n])

    transform_optoins = {
        'c || z, b || y': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'c || z, b || x': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        'a || z, b || x': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        'a || z, z || x': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        'b || z, z || x': np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        'b || z, a || x': np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    }
    transform = tk.StringVar(root, 'c || z, b || y')
    var = tk.OptionMenu(right, transform, *transform_optoins, command=fun_transform)
    var.config(font=SF, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.TOP)

    # Rotation
    ln = tk.Frame(right)
    ln.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, pady=4)
    var = tk.Label(ln, text='Rotate ', font=SF)
    var.pack(side=tk.LEFT)
    angle_deg = tk.DoubleVar(root, 0)
    var = tk.Entry(ln, textvariable=angle_deg, font=TF, width=3, bg=ety, fg=ety_txt)
    var.pack(side=tk.LEFT)
    var = tk.Label(ln, text='Deg, about ', font=SF)
    var.pack(side=tk.LEFT)

    def fun_rotate():
        angle = angle_deg.get()
        rmat = rotate_options[rotate_about.get()](angle)
        cmat = np.array([[matrix_vars[m][n].get() for n in range(3)] for m in range(3)])
        nmat = np.dot(rmat, cmat.T).T
        for m in range(3):
            for n in range(3):
                matrix_vars[m][n].set(nmat[m, n])

    rotate_options = {
        'X': fc.rotmatrixx,
        'Y': fc.rotmatrixy,
        'Z': fc.rotmatrixz,
    }
    rotate_about = tk.StringVar(root, 'Z')
    var = tk.OptionMenu(ln, rotate_about, *rotate_options)
    var.config(font=SF, bg=opt, activebackground=opt_active)
    var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
    var.pack(side=tk.LEFT)
    var = tk.Button(ln, text='Go', font=TF, command=fun_rotate, bg=btn, activebackground=btn_active)
    var.pack(side=tk.LEFT)

    # Matrix
    matrix_vars = [[tk.DoubleVar(root, latt._u[m, n]) for n in range(3)] for m in range(3)]
    ln = tk.Frame(right, relief=tk.RIDGE)
    ln.pack(side=tk.TOP, fill=tk.Y, expand=tk.YES, pady=4)
    for m in range(3):
        mat_ln = tk.Frame(ln)
        mat_ln.pack(side=tk.TOP, fill=tk.X)
        for n in range(3):
            var = tk.Entry(mat_ln, textvariable=matrix_vars[m][n], font=TF, width=3, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT)

    def fun_gen_u():
        nstr = '\n'.join([' '.join(['%6.4f' % matrix_vars[m][n].get() for n in range(3)]) for m in range(3)])
        utext.delete('1.0', tk.END)
        utext.insert(tk.END, nstr)

    def fun_load_u():
        s = utext.get('1.0', tk.END)
        uary = fg.str2array(s).reshape(3, 3)
        for m in range(3):
            for n in range(3):
                matrix_vars[m][n].set(uary[m, n])

    def fun_gen_l():
        nstr = '\n'.join([' '.join(['%6.4f' % matrix_vars[m][n].get() for n in range(3)]) for m in range(3)])
        ltext.delete('1.0', tk.END)
        ltext.insert(tk.END, nstr)

    def fun_load_l():
        s = ltext.get('1.0', tk.END)
        lary = fg.str2array(s).reshape(3, 3)
        for m in range(3):
            for n in range(3):
                matrix_vars[m][n].set(lary[m, n])

    left = tk.Frame(root)
    left.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)

    ln = tk.Frame(left)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='U:', width=10, font=SF)
    var.pack(side=tk.LEFT)
    utext = tk.Text(ln, width=20, height=3, font=HF, wrap=tk.NONE, background=ety)
    utext.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
    utext.insert(tk.END, ustr)
    var = tk.Button(ln, text=u'\u25B6', font=TF, command=fun_load_u, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)
    var = tk.Button(ln, text=u'\u25C0', font=TF, command=fun_gen_u, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)

    ln = tk.Frame(left)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Label(ln, text='LabFrame:', width=10, font=SF)
    var.pack(side=tk.LEFT)
    ltext = tk.Text(ln, width=20, height=3, font=HF, wrap=tk.NONE, background=ety)
    ltext.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
    ltext.insert(tk.END, lstr)
    var = tk.Button(ln, text=u'\u25B6', font=TF, command=fun_load_l, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)
    var = tk.Button(ln, text=u'\u25C0', font=TF, command=fun_gen_l, bg=btn,
                    activebackground=btn_active)
    var.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)

    def fun_update(event=None):
        s = utext.get('1.0', tk.END)
        uary = fg.str2array(s).reshape(3, 3)
        s = ltext.get('1.0', tk.END)
        lary = fg.str2array(s).reshape(3, 3)
        latt.set_orientation(uary, lary)
        det.set_labframe(lary)
        callback()
        root.destroy()

    ln = tk.Frame(left)
    ln.pack(side=tk.TOP, fill=tk.X)
    var = tk.Button(ln, text='Update', font=TF, command=fun_update, bg=btn,
                    activebackground=btn_active)
    var.pack(pady=2)


def tk_help():
    """Display help"""
    StringViewer(__doc__, title='Dans_Diffraction diffractometer', width=121)


def generate_reflections(latt: Lattice, det: Detector, minimum_gaussian_intensity=0.01):
    """Calulate reflections on detector"""
    # find reflections on detector
    ixyz, iuvw, idiff = det.reflection_position(
        qxqyqz=latt.rotated_qxqyqz,
        wavelength_a=latt.wavelength_a
    )
    idx = ~np.isnan(iuvw[:, 0])
    latt.generate_intensities(idx)
    res = np.sqrt(latt.resolution ** 2 + fc.dspace2q(latt.domain_size) ** 2)  # combine resolutions in A-1
    scaled_inten = fc.scale_intensity(latt.intensity[idx], idiff[idx], res)
    scale = sum(fc.scale_intensity(1, idiff, res))
    scaled_inten = scaled_inten / scale  # reduce intensity by total reflected intensity
    good = scaled_inten > minimum_gaussian_intensity  # minimise the number of gaussians to generate
    fwhm_mm = det.peak_width_mm(latt.fwhm[idx][good])
    xx, yy, mesh, det_x, det_y = det.detector_image(
        iuvw=iuvw[idx, :][good, :],
        intensity=scaled_inten[good],
        peak_width=fwhm_mm / 1000.,
        background=0
    )
    # Update REFLIST (all len(det_x))
    REFLIST['hkl'] = latt.hkl[idx, :][good, :]
    REFLIST['hkl_str'] = latt.hkl_str[idx][good]
    REFLIST['qxqyqz'] = latt.rotated_qxqyqz[idx, :][good, :]
    REFLIST['fwhm'] = latt.fwhm[idx][good]
    REFLIST['intensity'] = latt.intensity[idx][good]
    REFLIST['scaled'] = scaled_inten[good]
    REFLIST['detx'] = det_x * 1000.  # mm
    REFLIST['dety'] = det_y * 1000.  # mm
    REFLIST['color'] = [next(col_cycle) for n in range(len(det_x))]
    return xx, yy, mesh


def update_3d(array, line_obj):
    """Update 3D line object"""
    line_obj.set_xdata(array[:, 0])
    line_obj.set_ydata(array[:, 1])
    line_obj.set_3d_properties(array[:, 2])


class DiffractometerGui:
    """
    View and edit the symmetry operations
    """

    _figure_3d_size = FIGURE_3D_SIZE
    _figure_det_size = FIGURE_DET_SIZE

    def __init__(self, xtl: Crystal):
        """"Initialise"""
        self.xtl = xtl
        self.scatter = self.xtl.Scatter

        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Diffractometer')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol
        )

        # ---Menu---
        menu = {
            'File': {
                'New Window': self.menu_new,
                'Exit': self.root.destroy,
            },
            'Defaults': {
                'Supernova': self.fun_supernova,
                'Wish': self.fun_wish,
                'I16': self.fun_i16,
            },
            'Options': {
                'Plot Options': self.menu_options,
                'Set Orientation': self.menu_orientation,
            },
            'Help': {
                'Docs': tk_help,
                'About': popup_about,
            }
        }
        topmenu(self.root, menu)

        # XXXXXXX GUI ELEMENTS XXXXXXX
        grid = tk.Frame(self.root)
        grid.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        grid.columnconfigure((0, 1), weight=1)
        grid.rowconfigure((0, 1), weight=1)
        border = {}  # {'highlightbackground': "black", 'highlightthickness': 1}

        # ***--- TOP ---***
        # --- Reciprocal Space Plot ---
        top_left = tk.LabelFrame(grid, text='Diffractometer', **border)
        top_left.grid(row=0, column=0, sticky='nsew')
        frm = tk.Frame(top_left)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        self.fig1 = Figure(figsize=self._figure_3d_size, dpi=OPTIONS['fig_dpi'])
        self.fig1.patch.set_facecolor('w')

        canvas = FigureCanvasTkAgg(self.fig1, frm)
        canvas.get_tk_widget().configure(bg='black')
        canvas.draw()  # this must go above 3d axes or the mouse movement doesn't work

        self.ax3d = self.fig1.add_subplot(111, projection='3d')
        self.ax3d_reciprocal_lattice, = self.ax3d.plot([], [], [], 'bo', ms=6, alpha=0.3, label='Lattice')
        self.ax3d_lattice_detector, = self.ax3d.plot([], [], [], 'bo', ms=8, label='Reflections')
        self.ax3d_beam_lines, = self.ax3d.plot([], [], [], 'k-', lw=3, label='Beam')
        self.ax3d_wavevector, = self.ax3d.plot([], [], [], 'r-', lw=3, label='Q')
        self.ax3d_detector, = self.ax3d.plot([], [], [], 'k-', lw=2, label='Beam')
        self.ax3d_qdetector, = self.ax3d.plot([], [], [], 'r-', lw=2, label='Beam')
        self.ax3d_basis_arrows = []
        self.ax3d_ref_arrows = []
        self.ax3d.set_xlabel(u'Qx [\u212B\u207B\u00B9]')
        self.ax3d.set_ylabel(u'Qy [\u212B\u207B\u00B9]')
        self.ax3d.set_zlabel(u'Qz [\u212B\u207B\u00B9]')
        self.ax3d.set_xlim([-1, 1])
        self.ax3d.set_ylim([-1, 1])
        self.ax3d.set_zlim([-1, 1])
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

        # Toolbar
        # frm = tk.Frame(sec)
        # frm.pack(expand=tk.YES)
        # self.toolbar1 = NavigationToolbar2TkAgg(canvas, frm)
        # self.toolbar1.update()
        # self.toolbar1.pack(fill=tk.X, expand=tk.YES)

        # --- Detector Plot ---
        top_right = tk.LabelFrame(grid, text='Detector', **border)
        # sec.pack(side=tk.LEFT, expand=tk.YES, padx=4, pady=4)
        top_right.grid(row=0, column=1, sticky='nsew')
        frm = tk.Frame(top_right)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        self.fig2 = Figure(figsize=self._figure_det_size, dpi=OPTIONS['fig_dpi'])
        self.fig2.patch.set_facecolor('w')
        self.axd = self.fig2.add_subplot(111)
        self.axd_image = self.axd.pcolormesh(np.zeros([10, 10]), shading='auto')
        self.axd_center, = self.axd.plot([0], [0], 'w+', ms=6, label='Centre')
        self.axd_lattice, = self.axd.plot([], [], 'r+', ms=6, label='Nearby reflections')
        self.axd_allowed, = self.axd.plot([], [], 'ko', ms=6, label='Reflections')
        self.axd_ref_str = []
        self.axd.set_xlabel(u'x-axis [mm]')
        self.axd.set_ylabel(u'z-axis [mm]')
        self.axd.set_xlim([-0.5, 0.5])
        self.axd.set_ylim([-0.5, 0.5])
        self.axd.axis('image')
        canvas = FigureCanvasTkAgg(self.fig2, frm)
        canvas.get_tk_widget().configure(bg='black')
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

        # Toolbar
        frm = tk.Frame(top_right)
        frm.pack(expand=tk.YES)
        self.toolbar2 = NavigationToolbar2TkAgg(canvas, frm)
        self.toolbar2.update()
        self.toolbar2.pack(fill=tk.X, expand=tk.YES)

        # ***--- BOTTOM ---***
        # --- Diffractometer axes ---
        bottom_left = tk.Frame(grid, **border)
        # sec.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        bottom_left.grid(row=1, column=0, sticky='nsew')

        sec = tk.LabelFrame(bottom_left, text='Angles (Deg)')
        sec.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=4, pady=4)

        # Default scattering angle
        tth = self.xtl.Cell.tth([0, 0, 2], 8)[0]

        self.phi = tk_angle_element('Phi', sec, self.update, MSG_PHI, (-180, 180), 0)
        self.chi = tk_angle_element('Chi', sec, self.update, MSG_CHI, (-90, 100), 0)
        self.eta = tk_angle_element('Eta', sec, self.update, MSG_ETA, (-180, 180), 0)
        self.mu = tk_angle_element('Mu', sec, self.update, MSG_MU, (-180, 180), tth / 2)
        self.delta = tk_angle_element('Delta', sec, self.update, MSG_DEL, (-180, 180), 0)
        self.gamma = tk_angle_element('Gamma', sec, self.update, MSG_GAM, (-180, 180), tth)
        angle_dict = {
            'Phi': self.phi,
            'Chi': self.chi,
            'Eta': self.eta,
            'Mu': self.mu,
            'Delta': self.delta,
            'Gamma': self.gamma
        }

        # --- Beam ---
        bottom_right = tk.Frame(grid, **border)
        # sec.pack(side=tk.LEFT, expand=tk.YES, padx=4, pady=4)
        bottom_right.grid(row=1, column=1, sticky='nsew')
        self.latt = tk_beam(bottom_right, self.xtl, self.update)

        # --- Detector ---
        self.det = tk_detector(bottom_right, self.update)

        # --- HKL ---
        tk_hkl(bottom_left, self.latt, self.gamma, self.mu, self.update)

        # --- Scan ---
        tk_scan(bottom_left, angle_dict, self.latt, self.det)

        # Configure expansion of Tk frame
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.latt.generate_refs()
        self.det.generate_detector()

    ###################################################################################
    ################################# MENU ############################################
    ###################################################################################

    def menu_new(self):
        """Create a new instance"""
        DiffractometerGui(self.xtl)

    def menu_options(self):
        """Set plot options"""
        tk_plot_options(self.root, self.update)

    def menu_orientation(self):
        """Set lattice orientation"""
        tk_orientation(self.root, self.latt, self.det, self.update)

    ###################################################################################
    ############################## FUNCTIONS ##########################################
    ###################################################################################

    def update(self, event=None):
        """Update plot"""

        # Rotate lattice
        self.latt.rotate_to(
            phi=self.phi.get(),
            chi=self.chi.get(),
            eta=self.eta.get(),
            mu=self.mu.get()
        )
        # Rotate detector
        self.det.rotate_to(
            delta=self.delta.get(),
            gamma=self.gamma.get()
        )

        # Generate reflections on detector (update REFLIST)
        xx, yy, mesh = generate_reflections(
            latt=self.latt,
            det=self.det,
            minimum_gaussian_intensity=OPTIONS['min_intensity']
        )

        # --- Update 3D Plot ---
        if OPTIONS['3d_lattice']:
            update_3d(self.latt.rotated_qxqyqz, self.ax3d_reciprocal_lattice)
        else:
            update_3d(np.zeros([1, 3]), self.ax3d_reciprocal_lattice)

        [artist.remove() for artist in self.ax3d_basis_arrows]
        if OPTIONS['3d_basis']:
            astar, bstar, cstar = self.latt.vectors
            self.ax3d_basis_arrows = [
                fp.Arrow3D([0, astar[0]], [0, astar[1]], [0, astar[2]], lw=3, arrowstyle="-|>", color="r"),
                fp.Arrow3D([0, bstar[0]], [0, bstar[1]], [0, bstar[2]], lw=3, arrowstyle="-|>", color="b"),
                fp.Arrow3D([0, cstar[0]], [0, cstar[1]], [0, cstar[2]], lw=3, arrowstyle="-|>", color="g"),
            ]
            [self.ax3d.add_artist(artist) for artist in self.ax3d_basis_arrows]
        else:
            self.ax3d_basis_arrows = []

        update_3d(REFLIST['qxqyqz'], self.ax3d_lattice_detector)
        [artist.remove() for artist in self.ax3d_ref_arrows]
        if OPTIONS['3d_ref_arrows']:
            ki = self.det.ki(self.latt.wavelength_a)
            self.ax3d_ref_arrows = [
                fp.Arrow3D([0, q[0] + ki[0]], [0, q[1] + ki[1]], [0, q[2] + ki[2]], lw=1, arrowstyle="-|>", color=c)
                for q, c in zip(REFLIST['qxqyqz'], REFLIST['color'])  # kf arrows
            ] + [
                fp.Arrow3D([0, q[0]], [0, q[1]], [0, q[2]], lw=1, arrowstyle="-|>", color=c)
                for q, c in zip(REFLIST['qxqyqz'], REFLIST['color'])  # q arrows
            ]
            [self.ax3d.add_artist(artist) for artist in self.ax3d_ref_arrows]
        else:
            self.ax3d_ref_arrows = []
        # update_3d(self.det.corners(), self.ax3d_detector)
        update_3d(self.det.q_shape(self.latt.wavelength_a), self.ax3d_qdetector)
        if OPTIONS['3d_detector_corners']:
            kikf, qq = self.det.kfkiq_corners(self.latt.wavelength_a)
            update_3d(kikf, self.ax3d_beam_lines)
            update_3d(qq, self.ax3d_wavevector)
        else:
            kikf, qq = self.det.kfkiq(self.latt.wavelength_a)
            update_3d(kikf, self.ax3d_beam_lines)
            update_3d(qq, self.ax3d_wavevector)

        self.ax3d.set_xlim([-OPTIONS['maxq'], OPTIONS['maxq']])
        self.ax3d.set_ylim([-OPTIONS['maxq'], OPTIONS['maxq']])
        self.ax3d.set_zlim([-OPTIONS['maxq'], OPTIONS['maxq']])

        self.fig1.set_dpi(OPTIONS['fig_dpi'])
        # self.toolbar2.update()
        self.fig1.canvas.draw()

        # --- update detector plot ---
        self.axd_image.remove()
        # convert from m to mm
        xx = xx * 1000.
        yy = yy * 1000.
        if OPTIONS['det_log']:
            self.axd_image = self.axd.pcolormesh(xx, yy, np.log10(mesh+1), shading='auto', cmap=OPTIONS['det_cmap'])
        else:
            self.axd_image = self.axd.pcolormesh(xx, yy, mesh, shading='auto', cmap=OPTIONS['det_cmap'])
        if OPTIONS['det_labels']:
            self.axd_allowed.set_data(REFLIST['detx'], REFLIST['dety'])
            for txt in self.axd_ref_str:
                txt.remove()
            self.axd_ref_str = [
                self.axd.text(REFLIST['detx'][n], REFLIST['dety'][n], REFLIST['hkl_str'][n], c='w', fontsize=12)
                for n in range(len(REFLIST['hkl']))
            ]
            print(reflist_str())
        else:
            self.axd_allowed.set_data([], [])
            for txt in self.axd_ref_str:
                txt.remove()
            self.axd_ref_str = []
        self.axd.set_xlim(self.det.width_lim())
        self.axd.set_ylim(self.det.height_lim())
        self.axd_image.set_clim(OPTIONS['det_clim'])
        self.fig2.set_dpi(OPTIONS['fig_dpi'])
        self.toolbar2.update()
        self.fig2.canvas.draw()

    ###################################################################################
    ############################### DEFAULTS ##########################################
    ###################################################################################

    def fun_i16(self):
        """"Add I16 parameters"""
        self.latt.update_widgets(
            radiation='X-Ray',
            wavelength_a=fc.energy2wave(8),
            max_q=4,
            resolution=0.0005,
            peakwidth_deg=0.1
        )
        self.det.update_widgets(
            distance=565,
            width=487 * 0.172 * np.cos(np.deg2rad(35)),
            height=195 * 0.172,
            pixel_size=0.172,
        )
        self.update()

    def fun_wish(self):
        """"Add Wish parameters"""
        self.latt.update_widgets(
            radiation='Neutron',
            wavelength_a=fc.energy2wave(8),
            max_q=4,
            resolution=0.1,
            peakwidth_deg=1
        )
        self.det.update_widgets(
            distance=500,
            width=2000,
            height=200,
            pixel_size=2,
        )
        self.update()

    def fun_supernova(self):
        """Add SuperNova parameters"""
        self.latt.update_widgets(
            radiation='X-Ray',
            wavelength_a=fc.energy2wave(fg.Mo),
            max_q=8,
            resolution=0.1,
            peakwidth_deg=1
        )
        self.det.update_widgets(
            distance=56,
            width=100,
            height=100,
            pixel_size=0.1,
        )
        self.update()

    ###################################################################################
    ################################ BUTTONS ##########################################
    ###################################################################################
