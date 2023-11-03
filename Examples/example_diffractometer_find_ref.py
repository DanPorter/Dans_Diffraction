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
    max_mesh = 0
    max_mesh_sum = 0
    max_mesh_eta = 0
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
        if inten[n] > max_mesh_sum:
            max_mesh = mesh
            max_mesh_sum = inten[n]
            max_mesh_eta = eta
    # Volume
    total, hkl, overlaps = dif.fc.detector_volume_scan(
        wavelength_a=wavelength, resolution=resolution,
        phi=phi, chi=chi, eta=eta_range, mu=0, delta=delta, gamma=0,
        detector_distance_mm=detector_distance, height_mm=det_height, width_mm=det_width
    )
    return inten, max_mesh, max_mesh_eta, hkl, ref_hkl, ref_i, ref_c


def scan_phi(phi_range, eta, chi, delta):
    inten = np.zeros_like(phi_range)
    max_mesh = 0
    max_mesh_sum = 0
    max_mesh_phi = 0
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
        if inten[n] > max_mesh_sum:
            max_mesh = mesh
            max_mesh_sum = inten[n]
            max_mesh_phi = phi
    # Volume
    total, hkl, overlaps = dif.fc.detector_volume_scan(
        wavelength_a=wavelength, resolution=resolution,
        phi=phi_range, chi=chi, eta=eta, mu=0, delta=delta, gamma=0,
        detector_distance_mm=detector_distance, height_mm=det_height, width_mm=det_width
    )
    return inten, max_mesh, max_mesh_phi, hkl, ref_hkl, ref_i, ref_c


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
ttl = '%s E=%.4g keV looking for: %s' % (xtl.name, energy_kev, dif.fc.hkl2str(ref_search))
ttl += '\nphi=%.4g, chi=%.4g, eta=%.4g, delta=%.4g' % (phi_value, chi_value, eta_value, delta_value)


fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=[14, 6], dpi=60)
fig.subplots_adjust(top=0.85, left=0.08, right=0.95, wspace=0.3)
fig.suptitle(ttl)

max_det = 0
max_det_sum = 0
max_det_chi = 0
max_det_phi = 0
max_det_eta = 0
scan_elements = np.empty([0, 3], dtype=int)
scan_volume = 0
tot_exposures = 0
ref_list = np.empty([0, 3], dtype=int)
ref_int = np.empty([0], dtype=float)
ref_cnt = np.empty([0], dtype=int)

# Loop Chi values
for chi_value in np.arange(96, 0, -min_chi_step):
    if max_det_sum > 0:
        break
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
        print('Eta scan at chi %s with step %.2g' % (chi_value, eta_step))
        detsum, image, max_eta, voxels, rhkl, rint, rcnt = scan_eta(scanvals, phi=phi_value, chi=chi_value, delta=delta_value)
        max_phi = phi_value
        ax1.plot(scanvals, detsum, label='chi=%s' % chi_value)
    else:
        print('Phi scan at chi %s with step %.2g' % (chi_value, phi_step / 2))
        scanvals = np.arange(-180, 180, phi_step / 2)
        detsum, image, max_phi, voxels, rhkl, rint, rcnt = scan_phi(scanvals, eta=eta_value, chi=chi_value, delta=delta_value)
        max_eta = eta_value
        ax2.plot(scanvals, detsum, label='chi=%s' % chi_value)

    ref_list = np.vstack([ref_list, rhkl])
    ref_int = np.append(ref_int, rint)
    ref_cnt = np.append(ref_cnt, rcnt + tot_exposures)
    tot_exposures += len(scanvals)
    scan_elements = np.vstack([scan_elements, voxels])
    scan_elements = np.unique(scan_elements, axis=0)
    volume = len(voxels) * resolution ** 3
    scan_volume = len(scan_elements) * resolution ** 3
    print('  Scan volume: %.2f A^-3 (%d voxels, %.2g per step), total: %.2f%%' % (
    volume, len(voxels), volume / len(scanvals), 100 * scan_volume / total_volume))
    det_sum = np.sum(image)
    if det_sum > max_det_sum:
        print('  New max! eta=%.4g, phi=%.4g, sum=%.4g' % (max_eta, max_phi, det_sum))
        max_det = image
        max_det_sum = det_sum
        max_det_chi = chi_value
        max_det_phi = max_phi
        max_det_eta = max_eta

ax1.legend()
ax1.set_xlabel('Eta [Deg]')
ax1.set_ylabel('Detector Sum')

ax2.legend()
ax2.set_xlabel('Phi [Deg]')
ax2.set_ylabel('Detector Sum')

# Max image
xtl.Cell.orientation.rotate_6circle(eta=max_det_eta, chi=max_det_chi, phi=max_det_phi)
xx, yy, mesh, reflist = xtl.Scatter.detector_image(
    detector_distance_mm=detector_distance,
    delta=delta_value,
    gamma=0,
    height_mm=det_height,
    width_mm=det_width,
    pixel_size_mm=pixel_size / 1000.,
    energy_range_ev=resolution_ev,
    peak_width_deg=peak_width_deg,
    wavelength_a=dif.fc.energy2wave(energy_kev),
    background=0
)

ax3.pcolormesh(xx, yy, mesh, vmin=0, vmax=1, shading='auto')
ax3.set_xlabel('x-axis [mm]')
ax3.set_ylabel('z-axis [mm]')
ttl2 = 'Peak found: %s\n' % (dif.fc.hkl2str(reflist['hkl'][0]))
ttl2 += 'phi=%.4g, chi=%.4g, eta=%.4g\n' % (max_det_phi, max_det_chi, max_det_eta)
ttl2 += 'Time taken: %d, total exposure: %.2f%%' % (tot_exposures, 100 * scan_volume / total_volume)
ax3.set_title(ttl2)
ax3.axis('image')
# reflection labels
for n in range(len(reflist['hkl'])):
    ax3.text(reflist['detx'][n], reflist['dety'][n], reflist['hkl_str'][n], c='k')


" --------- "
# Show reflections
print('\n\n ------- Reflections found ------- ')
for n in range(len(ref_list)):
    print(f"{dif.fc.hkl2str(ref_list[n]):10s} exposure: {ref_cnt[n]:.0f}, intensity: {ref_int[n]:.2f}")
print('--------------\n\n')


" --------- "
# Scan eta at phi position
phi_step, chi_step, eta_step, mu_step = dif.fc.diffractometer_step(
    wavelength_a=dif.fc.energy2wave(energy_kev),
    resolution=resolution,
    phi=max_det_phi,
    chi=max_det_chi,
    eta=max_det_eta,
    delta=delta_value
)
print('Scanning eta about peak with step size: %.4g' % eta_step)
eta_scan = np.arange(max_det_eta - eta_step*3, max_det_eta + eta_step*3, 0.1*eta_step)
detsum, image, max_eta, vol, rhkl, rint, rcnt = scan_eta(eta_scan, phi=max_det_phi, chi=max_det_chi, delta=delta_value)

plt.figure(figsize=[10, 6], dpi=60)
plt.subplots_adjust(top=0.85, left=0.08, right=0.95, wspace=0.3)
plt.suptitle(ttl)

plt.subplot(121)
plt.plot(eta_scan, detsum, label='chi=%s' % max_det_chi)
plt.legend()
plt.xlabel('Eta [Deg]')
plt.ylabel('Detector Sum')

xtl.Cell.orientation.rotate_6circle(eta=max_eta, chi=max_det_chi, phi=max_det_phi)
xx, yy, mesh, reflist = xtl.Scatter.detector_image(
    detector_distance_mm=detector_distance,
    delta=delta_value,
    gamma=0,
    height_mm=det_height,
    width_mm=det_width,
    pixel_size_mm=pixel_size / 1000.,
    energy_range_ev=resolution_ev,
    peak_width_deg=peak_width_deg,
    wavelength_a=dif.fc.energy2wave(energy_kev),
    background=0
)
plt.subplot(122)
plt.pcolormesh(xx, yy, mesh, vmin=0, vmax=1, shading='auto')
plt.xlabel('x-axis [mm]')
plt.ylabel('z-axis [mm]')
ttl2 = 'phi=%.4g, chi=%.4g, eta=%.4g' % (max_det_phi, max_det_chi, max_det_eta)
plt.title(ttl2)
plt.axis('image')
# reflection labels
for n in range(len(reflist['hkl'])):
    plt.text(reflist['detx'][n], reflist['dety'][n], reflist['hkl_str'][n], c='w')

plt.show()
