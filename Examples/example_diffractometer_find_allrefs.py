"""
Dans_Diffraction Examples
Simulate scans with a diffractometer to find reflections from a randomly oriented crystal
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt # Plotting
cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

dif.fg.nice_print()
dif.fp.set_plot_defaults()


# f = cf+'/../Dans_Diffraction/Structures/Diamond.cif'
# xtl = dif.Crystal(f)
# xtl = dif.structure_list.Na08CoO2_P63mmc()
# xtl = dif.structure_list.Ca2RuO4()
xtl = dif.structure_list.LiCoO2()

" --- Parameters --- "
energy_kev = 8.0
resolution_ev = 200
peak_width_deg = 0.5
detector_distance = 565  # mm, I16 Pilatus_100K
det_height = 195 * 0.172
det_width = 487 * 0.172 * np.cos(np.deg2rad(35))
pixel_size = 172  # um
background = 0

# Reflection seletor
refs = xtl.Scatter.get_hkl(energy_kev=energy_kev, remove_symmetric=True)
ref_tth = xtl.Cell.tth(refs, energy_kev=energy_kev)
ref_multiplicity = xtl.Symmetry.reflection_multiplyer(refs)
ref_intensity = xtl.Scatter.intensity(refs)
ref_cluster = np.array([np.sum(1 / (np.abs(th - ref_tth)+1))-1 for th in ref_tth])
ref_select = ref_intensity * ref_multiplicity / ref_cluster  # ** 2
ref_search = refs[np.argmax(ref_select), :]

# Calculated parameters
wavelength = dif.fc.energy2wave(energy_kev)
delta_value = xtl.Cell.tth(ref_search, energy_kev=energy_kev)[0]
phi_value = 0
chi_value = 90
eta_value = delta_value / 2.
domain_size = dif.fc.scherrer_size(peak_width_deg, delta_value, energy_kev=energy_kev, shape_factor=1.0)
resolution = dif.fc.wavevector_resolution(energy_range_ev=resolution_ev, domain_size_a=domain_size)
delta_width, gam_width, pixel_width = dif.fc.detector_angle_size(detector_distance, det_height, det_width, pixel_size)
total_volume = dif.fc.reciprocal_volume(delta_value+delta_width/2, delta_value-delta_width/2, wavelength_a=wavelength)


" --- scan functions --- "
def scan_eta(eta_range, phi, chi, delta):
    inten = np.zeros_like(eta_range)
    ref_hkl = np.empty([0, 3], dtype=int)
    ref_i = np.empty([0], dtype=float)
    ref_c = np.empty([0], dtype=float)
    for n, eta in enumerate(eta_range):
        xtl.Cell.orientation.rotate_6circle(eta=eta, chi=chi, phi=phi)
        xx, yy, mesh, reflist = xtl.Scatter.detector_image(
            detector_distance_mm=detector_distance,
            delta=delta,
            gamma=0,
            height_mm=det_height,
            width_mm=det_width,
            pixel_size_mm=pixel_size / 1000.,
            energy_range_ev=resolution_ev,
            peak_width_deg=peak_width_deg,
            wavelength_a=wavelength,
            background=background,
        )
        inten[n] = np.sum(mesh)
        ref_hkl = np.vstack([ref_hkl, reflist['hkl']])
        ref_i = np.append(ref_i, reflist['intensity'])
        ref_c = np.append(ref_c, len(reflist['hkl']) * [n])
    return inten, ref_hkl, ref_i, ref_c


def scan_phi(phi_range, eta, chi, delta):
    inten = np.zeros_like(phi_range)
    ref_hkl = np.empty([0, 3], dtype=int)
    ref_i = np.empty([0], dtype=float)
    ref_c = np.empty([0], dtype=float)
    for n, phi in enumerate(phi_range):
        xtl.Cell.orientation.rotate_6circle(eta=eta, chi=chi, phi=phi)
        xx, yy, mesh, reflist = xtl.Scatter.detector_image(
            detector_distance_mm=detector_distance,
            delta=delta,
            gamma=0,
            height_mm=det_height,
            width_mm=det_width,
            pixel_size_mm=pixel_size / 1000.,
            energy_range_ev=resolution_ev,
            peak_width_deg=peak_width_deg,
            wavelength_a=wavelength,
            background=background,
        )
        inten[n] = np.sum(mesh)
        ref_hkl = np.vstack([ref_hkl, reflist['hkl']])
        ref_i = np.append(ref_i, reflist['intensity'])
        ref_c = np.append(ref_c, len(reflist['hkl']) * [n])
    return inten, ref_hkl, ref_i, ref_c


" --- Set up calculation --- "
# Orient crystal
# xtl.Cell.orientation.orient(c_axis=[1, 1, 1], a_axis=[1, 1, 0])
xtl.Cell.orientation.random_orientation()

print('---- Dans_Diffraction - Search Ref Simulator ---')
print(f"Energy: {energy_kev}, resolutuion: {resolution} A-1")
print(f"Search for ref: {dif.fc.hkl2str(ref_search)} at delta={delta_value:.2f} Deg")
print(f"Crystal Structure {xtl.name} with random orientation:")
print(xtl.Cell.orientation)

phi_step, chi_step, eta_step, mu_step = dif.fc.diffractometer_step(
    wavelength_a=wavelength,
    resolution=resolution,
    phi=0,
    chi=0,  # step is smallest at chi=0
    eta=eta_value,
    delta=delta_value
)
print(f"Step size: phi = {phi_step:.2g}, eta = {eta_step:.2g}")

# Determine chi step
chi_steps = np.arange(1, 11, 1)
chi_overlaps = np.array([np.sum(dif.fc.detector_volume_scan(
        wavelength_a=wavelength, resolution=resolution,
        phi=phi_value, chi=np.arange(0, 100, step), eta=eta_value, delta=delta_value,
        detector_distance_mm=detector_distance, height_mm=det_height, width_mm=det_width
    )[2]) for step in chi_steps])
chi_overlaps[chi_overlaps < (0.1 * np.max(chi_overlaps))] = np.max(chi_overlaps)
min_chi_step = chi_steps[np.argmin(chi_overlaps)]
print(f"Minimum chi step: {min_chi_step:.2g}")

print('\n\nStart search')
tot_exposures = 0
ref_list = np.empty([0, 3], dtype=int)
ref_int = np.empty([0], dtype=float)
ref_cnt = np.empty([0], dtype=int)
ref_chi = np.empty([0], dtype=int)

# Loop Chi values
for chi_value in np.arange(96, 0, -min_chi_step):
    # use calcualted phi step to choose scan type
    phi_step, chi_step, eta_step, mu_step = dif.fc.diffractometer_step(
        wavelength_a=wavelength,
        resolution=resolution,
        phi=phi_value,
        chi=chi_value,
        eta=eta_value,
        delta=delta_value
    )
    if phi_step < 0.01 or phi_step > 10:
        scanvals = np.arange(0.25 * delta_value, 0.75 * delta_value, eta_step / 2)
        print('Eta scan at chi %s with step %.2g. nRefs=%d' % (chi_value, eta_step / 2, len(ref_list)))
        detsum, rhkl, rint, rcnt = scan_eta(scanvals, phi=phi_value, chi=chi_value, delta=delta_value)
    else:
        print('Phi scan at chi %s with step %.2g. nRefs=%d' % (chi_value, phi_step / 2, len(ref_list)))
        scanvals = np.arange(-180, 180, phi_step / 2)
        detsum, rhkl, rint, rcnt = scan_phi(scanvals, eta=eta_value, chi=chi_value, delta=delta_value)
    if len(rhkl) > 0:
        # Remove matching reflections
        rhkl, ii = np.unique(rhkl, return_index=True, axis=0)
        rint = rint[ii]
        rcnt = rcnt[ii]
        print(' Reflections found: %d' % len(rhkl))
    ref_list = np.vstack([ref_list, rhkl])
    ref_int = np.append(ref_int, rint)
    ref_cnt = np.append(ref_cnt, rcnt + tot_exposures)
    ref_chi = np.append(ref_chi, len(rhkl) * [chi_value])
    tot_exposures += len(scanvals)

" --------- "
# Show reflections
print('\n\n ------- Reflections found ------- ')
for n in range(len(ref_list)):
    print(f"{dif.fc.hkl2str(ref_list[n]):10s} exposure: {ref_cnt[n]}, intensity: {ref_int[n]:.2f}")
print('--------------\n\n')

# search_ref
idx = np.append(dif.fg.find_vector(ref_list, ref_search), dif.fg.find_vector(ref_list, -np.array(ref_search)))
if len(idx) > 1:
    print(f"Found {dif.fc.hkl2str(ref_list[idx[0]])} {len(idx)} times, first at exposure {ref_cnt[idx[0]]}")
else:
    print(f"Reflection {dif.fc.hkl2str(ref_search)} not found!")


