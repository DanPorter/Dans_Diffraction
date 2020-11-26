"""
Dans_Diffraction Examples
Calcualte number of peaks possible on a high angle detector
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

fg = dif.fg

#################################################################################################
################################## SETTINGS #####################################################
#################################################################################################

energy_kev = 8.0
max_q = 5  # inverse A

# Structure
# xtl = dif.structure_list.NaCl.build()
xtl = dif.structure_list.Na08CoO2_P63mmc.build()

# Initial rotation
ini_phi = 0
ini_eta = 0
ini_chi = 0

# Detector settings
beampipe_dist = 100  # mm
#detector_height = 83.8  # mm vertical
#detector_width = 33.5  # mm horizontal
detector_height = 100.  # mm vertical
detector_width = 100.  # mm horizontal
detector_pitch = 0  # deg inclination to vertical
#detector_dist = 300  # mm distance from sample to detector centre along the beam direction
detector_dist = 50  # mm distance from sample to detector centre along the beam direction
detector_x = 0  # mm horizontal distance from sample to detector centre
detector_y = 0  # mm vertical distance from sample to detector centre
detector_del = 63.71  # deg two-theta angle of detector
detector_gam = 0  # deg gamma angle (not implemented yet)
pixel_x = 1  # mm pixel size along width
pixel_y = 1  # mm pixel size along height

#################################################################################################
############################### CALCULATIONS ####################################################
#################################################################################################

# Calculate reciprocal lattice
k = 2 * np.pi / dif.functions_crystallography.energy2wave(energy_kev)
hkl = xtl.Cell.all_hkl(energy_kev, 180)
tth = xtl.Cell.tth(hkl, energy_kev)
q = xtl.Cell.calculateQ(hkl)
I = xtl.Scatter.intensity(hkl)

q = fg.rot3D(q, ini_phi, ini_chi, ini_eta)
q_max = (q[:, 0] <= max_q) * (q[:, 1] <= 0) * (q[:, 2] <= 2 * max_q) * (q[:, 0] >= -max_q) * (q[:, 1] >= -2 * max_q) * (
        q[:, 2] >= 0)
q_latt = q[q_max, :]

sample = [0, 0, 0]
beampipe = [0, 0, -beampipe_dist]
detector_cen = detector_dist * fg.rot3D([0, 0, 1], 0, detector_gam, -detector_del)[0] + [detector_x, detector_y, 0]
# detector_cen = detector_dist*fg.norm(np.array([np.sin(np.deg2rad(detector_gam)), np.sin(np.deg2rad(detector_del)), np.cos(np.deg2rad(detector_gam))*np.cos(np.deg2rad(detector_del))])) + [detector_x, detector_y, 0]

detector_corners_ini = np.array([
    [detector_x - detector_width / 2.0,
     detector_y - (detector_height * np.cos(np.deg2rad(detector_pitch)) / 2.),
     detector_dist + (detector_height * np.sin(np.deg2rad(detector_pitch)) / 2.)],
    [detector_x + detector_width / 2.0,
     detector_y - (detector_height * np.cos(np.deg2rad(detector_pitch)) / 2.),
     detector_dist + (detector_height * np.sin(np.deg2rad(detector_pitch)) / 2.)],
    [detector_x + detector_width / 2.0,
     detector_y + (detector_height * np.cos(np.deg2rad(detector_pitch)) / 2.),
     detector_dist - (detector_height * np.sin(np.deg2rad(detector_pitch)) / 2.)],
    [detector_x - detector_width / 2.0,
     detector_y + (detector_height * np.cos(np.deg2rad(detector_pitch)) / 2.),
     detector_dist - (detector_height * np.sin(np.deg2rad(detector_pitch)) / 2.)]])
detector_corners = fg.rot3D(detector_corners_ini, 0, detector_gam, -detector_del)
# detector_corners = np.array([detector_cen+cnr for cnr in detector_corners_ini])

pts = 11
# detector_line is used to genereate the detector shape in reciprocal space
detector_line = np.hstack([
    np.array([np.linspace(detector_corners[0, 0], detector_corners[1, 0], pts),
              np.linspace(detector_corners[0, 1], detector_corners[1, 1], pts),
              np.linspace(detector_corners[0, 2], detector_corners[1, 2], pts)]),
    np.array([np.linspace(detector_corners[1, 0], detector_corners[2, 0], pts),
              np.linspace(detector_corners[1, 1], detector_corners[2, 1], pts),
              np.linspace(detector_corners[1, 2], detector_corners[2, 2], pts)]),
    np.array([np.linspace(detector_corners[2, 0], detector_corners[3, 0], pts),
              np.linspace(detector_corners[2, 1], detector_corners[3, 1], pts),
              np.linspace(detector_corners[2, 2], detector_corners[3, 2], pts)]),
    np.array([np.linspace(detector_corners[3, 0], detector_corners[0, 0], pts),
              np.linspace(detector_corners[3, 1], detector_corners[0, 1], pts),
              np.linspace(detector_corners[3, 2], detector_corners[0, 2], pts)])]).T

# generate detector pixels
dx, dy = np.meshgrid(
    np.arange(detector_x - detector_width / 2.0, detector_x + detector_width / 2.0, pixel_x),
    np.arange(detector_y - detector_height / 2.0, detector_y + detector_height / 2.0, pixel_y))
detector_xy = np.array([dx.flatten(), dy.flatten(), detector_dist * np.ones(dx.size)]).T
detector_pixel = fg.rot3D(detector_xy, 0, detector_gam, -detector_del)
# detector_pixel = np.array([detector_cen+pxl for pxl in detector_xy])

# Determine initial and final wave-vectors, then generate wavevector transfer, q
ki = -k * fg.norm(beampipe)  # A-1, initial wavevector
kf = k * fg.norm(detector_cen)  # A-1, final wavevector (same magnitude)
q_cen = kf - ki  # A-1, wavevector transfer, centre of detector
q_det = k * fg.norm(detector_corners) - ki  # A-1, wavevector transfer, detector corners
q_lin = k * fg.norm(detector_line) - ki  # A-1, wavevector transfer, shape of detector
q_pixel = k * fg.norm(detector_pixel) - ki  # detector pixels

# Generate detector image
# the inverse distance of lattice points close to the image centre is
# added to each pixel
detector_image = np.zeros(dx.shape)
i, j = np.unravel_index(range(len(q_pixel)), dx.shape)
det_dist = np.max(fg.mag(q_det - q_cen))
q_dist = fg.mag(q_latt[:, [0, 2, 1]] - q_cen)
q_check = q_latt[q_dist < det_dist, :]
for latt_point in q_check:
    pixel_dist = fg.mag(q_pixel - latt_point[[0, 2, 1]])
    # detector_image[i,j] += det_dist/pixel_dist
    detector_image[i, j] += 1 / (1 + (pixel_dist / 0.05) ** 2)

#################################################################################################
############################### CREATE PLOT #####################################################
#################################################################################################

fig = plt.figure(figsize=[12, 10])
ax = fig.add_subplot(221, projection='3d')
ax.plot([sample[0]], [sample[2]], [sample[1]], 'k+', lw=4, ms=20, label='Sample')
ax.plot([-ki[0], 0], [-ki[2], 0], [-ki[1], 0], 'k-s', lw=4, ms=10, label='k$_i$')
pl_kf, = ax.plot([0, kf[0]], [0, kf[2]], [0, kf[1]], 'k-s', lw=4, ms=10, label='k$_f$')
pl_qc, = ax.plot([0, q_cen[0]], [0, q_cen[2]], [0, q_cen[1]], 'r-s', lw=4, ms=10, label='Q')
pl_dt, = ax.plot(q_lin[:, 0], q_lin[:, 2], q_lin[:, 1], 'k-', lw=4, ms=10, label='Q')
pl_q, = ax.plot(q_latt[:, 0], q_latt[:, 1], q_latt[:, 2], 'b+', ms=5, label='Lattice')
pl_cq, = ax.plot(q_check[:, 0], q_check[:, 1], q_check[:, 2], 'ro', ms=6, label='on detector')
ax.set_xlim3d(-max_q, max_q)
ax.set_ylim3d(0, -1.5 * max_q)
ax.set_zlim3d(0, 1.5 * max_q)
ax.set_xlabel('x', fontsize=22)
ax.set_ylabel('z', fontsize=22)
ax.set_zlabel('y', fontsize=22)

# Detector
axdet = plt.axes([0.6, 0.1, 0.4, 0.8])
plt.sca(axdet)
pl_im = plt.imshow(detector_image)
pl_im.set_clim(0, 1)

axsldr1 = plt.axes([0.1, 0.4, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr2 = plt.axes([0.1, 0.3, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr3 = plt.axes([0.1, 0.2, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr4 = plt.axes([0.1, 0.1, 0.35, 0.06], facecolor='lightgoldenrodyellow')
sldr1 = plt.Slider(axsldr1, 'phi', -180, 180, valinit=0, valfmt='%5.2f')
sldr2 = plt.Slider(axsldr2, 'chi', -98, 98, valinit=90, valfmt='%5.2f')
sldr3 = plt.Slider(axsldr3, 'eta', -40, 220, valinit=0, valfmt='%5.2f')
sldr4 = plt.Slider(axsldr4, 'delta', 0, 160, valinit=detector_del, valfmt='%5.2f')


def update_all(val):
    "Update function for pilatus image"

    phi, chi, eta, delta = sldr1.val, sldr2.val, sldr3.val, sldr4.val

    q_rot = fg.rot3D(q, phi, 90 - chi, eta)
    q_max = (q_rot[:, 0] <= max_q) * (q_rot[:, 1] <= 0) * (q_rot[:, 2] <= 2 * max_q) * (q_rot[:, 0] >= -max_q) * (
            q_rot[:, 1] >= -2 * max_q) * (q_rot[:, 2] >= 0)
    q_latt = q_rot[q_max, :]

    detector_cen = detector_dist * fg.rot3D([0, 0, 1], 0, detector_gam, -delta)[0] + [detector_x, detector_y, 0]
    detector_corners = fg.rot3D(detector_corners_ini, 0, detector_gam, -delta)
    detector_pixel = fg.rot3D(detector_xy, 0, detector_gam, -delta)
    # detector_cen = detector_dist*fg.norm(np.array([np.sin(np.deg2rad(detector_gam)), np.sin(np.deg2rad(detector_del)), np.cos(np.deg2rad(detector_gam))*np.cos(np.deg2rad(detector_del))])) + [detector_x, detector_y, 0]
    # detector_corners = np.array([detector_cen+cnr for cnr in detector_corners_ini])
    # detector_pixel = np.array([detector_cen+pxl for pxl in detector_xy])

    detector_line = np.hstack([np.array([np.linspace(detector_corners[0, 0], detector_corners[1, 0], pts),
                                         np.linspace(detector_corners[0, 1], detector_corners[1, 1], pts),
                                         np.linspace(detector_corners[0, 2], detector_corners[1, 2], pts)]),
                               np.array([np.linspace(detector_corners[1, 0], detector_corners[2, 0], pts),
                                         np.linspace(detector_corners[1, 1], detector_corners[2, 1], pts),
                                         np.linspace(detector_corners[1, 2], detector_corners[2, 2], pts)]),
                               np.array([np.linspace(detector_corners[2, 0], detector_corners[3, 0], pts),
                                         np.linspace(detector_corners[2, 1], detector_corners[3, 1], pts),
                                         np.linspace(detector_corners[2, 2], detector_corners[3, 2], pts)]),
                               np.array([np.linspace(detector_corners[3, 0], detector_corners[0, 0], pts),
                                         np.linspace(detector_corners[3, 1], detector_corners[0, 1], pts),
                                         np.linspace(detector_corners[3, 2], detector_corners[0, 2], pts)])]).T

    kf = k * fg.norm(detector_cen)
    q_cen = kf - ki
    q_det = k * fg.norm(detector_corners) - ki
    q_lin = k * fg.norm(detector_line) - ki
    q_pixel = k * fg.norm(detector_pixel) - ki

    detector_image = np.zeros(dx.shape)
    det_dist = np.max(fg.mag(q_det - q_cen))
    q_dist = fg.mag(q_latt[:, [0, 2, 1]] - q_cen)
    q_check = q_latt[q_dist < det_dist, :]
    for latt_point in q_check:
        pixel_dist = fg.mag(q_pixel - latt_point[[0, 2, 1]])
        # detector_image[i,j] += det_dist/pixel_dist
        detector_image[i, j] += 1 / (1 + (pixel_dist / 0.05) ** 2)
    print(q_dist.min(), det_dist, len(q_check), detector_image.min(), detector_image.max())

    pl_im.set_data(detector_image)
    # pl_im.set_clim(detector_image.min(), detector_image.max())

    pl_q.set_xdata(q_latt[:, 0])
    pl_q.set_ydata(q_latt[:, 1])
    pl_q.set_3d_properties(q_latt[:, 2], 'z')

    pl_cq.set_xdata(q_check[:, 0])
    pl_cq.set_ydata(q_check[:, 1])
    pl_cq.set_3d_properties(q_check[:, 2], 'z')

    pl_kf.set_xdata([0, kf[0]])
    pl_kf.set_ydata([0, kf[2]])
    pl_kf.set_3d_properties([0, kf[1]], 'z')

    pl_qc.set_xdata([0, q_cen[0]])
    pl_qc.set_ydata([0, q_cen[2]])
    pl_qc.set_3d_properties([0, q_cen[1]], 'z')

    pl_dt.set_xdata(q_lin[:, 0])
    pl_dt.set_ydata(q_lin[:, 2])
    pl_dt.set_3d_properties(q_lin[:, 1], 'z')

    plt.draw()
    # fig.canvas.draw()


sldr1.on_changed(update_all)
sldr2.on_changed(update_all)
sldr3.on_changed(update_all)
sldr4.on_changed(update_all)
plt.show()
