# Dans_Diffaction
Reads crystallographic cif files and simulates diffraction
By Dan Porter, Diamond Light Source
2018

Usage:
from Dans_Diffraction import Crystal
xtl = Crystal('some_file.cif')
xtl.info()
xtl.Scatter.print_all_reflections(energy_kev=5)
xtl.Plot.simulate_powder(energy_kev=8)


For comments, queries or bugs - email dan.porter@diamond.ac.uk
