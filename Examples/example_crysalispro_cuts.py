"""
Example Read recirpocal space cuts from CrysAlisPro
"""

import sys, os
import re
import numpy as np
import matplotlib.pyplot as plt

cf = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cf, '..'))
import Dans_Diffraction as dif

print(dif.version_info())


def read_image(filename, resolution=0.8):
    """
    Read uncompressed image file from CrysAlisPro
        In CrysAlisPro, write an uncompressed image with:
    >> wd inc "image.img"
        In Pyhton, read the image with:
    >> qx, qy, data = read_image("image.img")
    >> plt.pcolormesh(qx, qy, data)
    """

    # Get the file size from the header
    with open(filename, 'rb') as file:
        header = file.read()
    NHEADER = int(re.findall(b'NHEADER=\s*\d+', header)[0].strip(b'NHEADER='))
    NX = int(re.findall(b'NX=\s*\d+', header)[0].strip(b'NX='))
    NY = int(re.findall(b'NY=\s*\d+', header)[0].strip(b'NY='))

    # Separate header from data
    with open(filename, 'rb') as file:
        header = file.read(NHEADER)
        data = np.fromfile(file, np.int32)
    data = np.reshape(data, [NY, NX])

    # Determine the pixel coordinates
    qmax = 2 * np.pi / resolution
    qpixel = 2 * qmax / NX
    qxrange = np.arange(-qpixel * (NX / 2.), qpixel * (NX / 2.), qpixel)
    qyrange = np.arange(-qpixel * (NY / 2.), qpixel * (NY / 2.), qpixel)
    qx, qy = np.meshgrid(qxrange, qyrange)
    return qx, qy, data


cif_file = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\Projects\NaFeMnO2\P2-NaFeMnO2_icsd194731_fixed.cif"
img_file = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\Projects\NaFeMnO2\correct_super_uncomp.img"

qx, qy, img = read_image(img_file, 0.8)
xtl = dif.Crystal(cif_file)

P = [[4, 2, 0], [2, 4, 0], [0, 0, 3]]  # [a', b', c']=P*[a, b, c] 1/6th Supercell: a'=4a+2b, b'=2a+4b, c'=c
sup = xtl.generate_superstructure(P)

# Generate all the supercell lattice points in our recirpocal space plane
Qx, Qy, hkl = sup.Cell.reciprocal_space_plane(
    x_axis=sup.parenthkl2super([1, 1, 0]),
    y_axis=sup.parenthkl2super([0, 0, 1]),
    centre=sup.parenthkl2super([-0.5, 0.5, 0]),
    q_max=8.5,
    cut_width=0.05,
    )

plt.figure()
plt.pcolormesh(qx, qy, img, cmap=plt.get_cmap('hot_r'))
plt.clim([-1, 1e1])
plt.scatter(Qx, Qy, s=10, facecolors='none', edgecolors='b')
xtl.Plot.axis_reciprocal_lattice_lines([1, 1, 0], [0, 0, 1], [-0.5, 0.5, 0], lw=0.5, c='grey', q_max=8)
plt.axis('image')
plt.axis([-4.5, 4.5, -4.5, 4.5])
plt.show()
