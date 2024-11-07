"""
Dans_Diffraction Examples
Matplotlib figure with sliders showing how different basis options change with lattice parameter
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt  # Plotting

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))

import Dans_Diffraction as dif
from Dans_Diffraction import functions_lattice as fl

dif.fg.nice_print()


latpar = (2.85, 2.85, 10.8, 90, 90, 120)
bases = {
    'MaterialsProject': fl.basis_1,
    'Vesta': fl.basis_2,
    'Busing ang Levy': fl.basis_3,
}
colours = ['k', 'b', 'r']

fig = plt.figure(figsize=[12, 10])

# Direct axes
ax1 = fig.add_subplot(221, projection='3d')


def box_coords(basis):
    uvw = np.array([[0., 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1],
                    [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0],
                    [0, 0, 1]])
    return np.dot(uvw, basis)


bases_lines = {}
bases_labels = {basis_name: [] for basis_name in bases}
for (basis_name, basis_fun), col in zip(bases.items(), colours):
    box = box_coords(basis_fun(*latpar))
    bases_lines[basis_name], = ax1.plot(box[:, 0], box[:, 1], box[:, 2], '-', c=col, lw=2, label=basis_name)
    bases_labels[basis_name] += [
        ax1.text(box[n, 0], box[n, 1], box[n, 2], lab, c=col) for n, lab in [(1, 'a'), (5, 'b'), (7, 'c')]
    ]

ax1.set_xlim3d(-10, 10)
ax1.set_ylim3d(-10, 10)
ax1.set_zlim3d(-10, 10)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.legend(bbox_to_anchor=(0.05, 0.5))
ax1.set_title('Direct Space')

# Reciprocal Axes
ax2 = fig.add_subplot(223, projection='3d')

rbases_lines = {}
rbases_labels = {basis_name: [] for basis_name in bases}
for (basis_name, basis_fun), col in zip(bases.items(), colours):
    box = box_coords(fl.reciprocal_basis(basis_fun(*latpar)))
    rbases_lines[basis_name], = ax2.plot(box[:, 0], box[:, 1], box[:, 2], '-', c=col, lw=2, label=basis_name)
    rbases_labels[basis_name] += [
        ax2.text(box[n, 0], box[n, 1], box[n, 2], lab, c=col) for n, lab in [(1, 'a*'), (5, 'b*'), (7, 'c*')]
    ]

ax2.set_xlim3d(-4, 4)
ax2.set_ylim3d(-4, 4)
ax2.set_zlim3d(-4, 4)
ax2.set_xlabel('x*')
ax2.set_ylabel('y*')
ax2.set_zlabel('z*')
ax2.set_title('Reciprocal Space')

# Sliders
axsldr1 = plt.axes([0.6, 0.8, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr2 = plt.axes([0.6, 0.7, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr3 = plt.axes([0.6, 0.6, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr4 = plt.axes([0.6, 0.5, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr5 = plt.axes([0.6, 0.4, 0.35, 0.06], facecolor='lightgoldenrodyellow')
axsldr6 = plt.axes([0.6, 0.3, 0.35, 0.06], facecolor='lightgoldenrodyellow')
sldr1 = plt.Slider(axsldr1, 'a', 1.0, 15.0, valinit=latpar[0], valfmt='%5.2f')
sldr2 = plt.Slider(axsldr2, 'b', 1.0, 15.0, valinit=latpar[1], valfmt='%5.2f')
sldr3 = plt.Slider(axsldr3, 'c', 1.0, 15.0, valinit=latpar[2], valfmt='%5.2f')
sldr4 = plt.Slider(axsldr4, 'alpha', 0, 180, valinit=latpar[3], valfmt='%5.1f')
sldr5 = plt.Slider(axsldr5, 'beta', 0, 180, valinit=latpar[4], valfmt='%5.1f')
sldr6 = plt.Slider(axsldr6, 'gamma', 0, 180, valinit=latpar[5], valfmt='%5.1f')

# Volume
axsvol = plt.axes([0.8, 0.2, 0.1, 0.06])
axsvol.set_axis_off()
vol_txt = axsvol.text(0, 0, 'Volume = %5.3f A^3' % fl.lattice_volume(*latpar))

# Reset
axreset = plt.axes([0.9, 0.1, 0.1, 0.06])
button = plt.Button(axreset, 'Reset', hovercolor='0.975')


def reset(event):
    for sldr in [sldr1, sldr2, sldr3, sldr4, sldr5, sldr6]:
        sldr.reset()


def update_all(val):
    new_lat = sldr1.val, sldr2.val, sldr3.val, sldr4.val, sldr5.val, sldr6.val

    if not fl.angles_allowed(*new_lat[3:]):
        print(f"Angle not allowed: {new_lat[3:]}")
        return

    for (basis_name, basis_fun), col in zip(bases.items(), colours):
        dir_box = box_coords(basis_fun(*new_lat))
        bases_lines[basis_name].set_data_3d(dir_box.T)
        for n, idx in enumerate((1, 5, 7)):
            bases_labels[basis_name][n].set_x(dir_box[idx, 0])
            bases_labels[basis_name][n].set_y(dir_box[idx, 1])
            bases_labels[basis_name][n].set_z(dir_box[idx, 2])

        rec_box = box_coords(fl.reciprocal_basis(basis_fun(*new_lat)))
        rbases_lines[basis_name].set_data_3d(rec_box.T)
        for n, idx in enumerate((1, 5, 7)):
            rbases_labels[basis_name][n].set_x(rec_box[idx, 0])
            rbases_labels[basis_name][n].set_y(rec_box[idx, 1])
            rbases_labels[basis_name][n].set_z(rec_box[idx, 2])

        vol_txt.set_text('Volume = %5.3f A^3' % fl.lattice_volume(*new_lat))

    plt.draw()
    fig.canvas.draw()


sldr1.on_changed(update_all)
sldr2.on_changed(update_all)
sldr3.on_changed(update_all)
sldr4.on_changed(update_all)
sldr5.on_changed(update_all)
sldr6.on_changed(update_all)
button.on_clicked(reset)
plt.show()
