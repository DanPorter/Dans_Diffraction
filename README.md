# Dans_Diffaction
Reads crystallographic cif files and simulates diffraction

**Version 2.x**

By Dan Porter, Diamond Light Source
2018

TL;DR:
```python
import Dans_Diffraction as dif
xtl = dif.Crystal('some_file.cif')
xtl.info() # print Crystal structure parameters

# Print reflection list:
print(xtl.Scatter.print_all_reflections(energy_kev=5)) 

# Plot Powder pattern:
xtl.Plot.simulate_powder(energy_kev=8)
plt.show()

# Start graphical user interface:
xtl.start_gui()
```

For comments, queries or bugs - email dan.porter@diamond.ac.uk

# Installation
**Requirements:** 
Python 2.8+/3+ with packages: *Numpy*, *Matplotlib*, *Scipy*, *Tkinter*

Stable version from PyPI:
```text
$ pip install Dans-Diffraction
```

Latest version from GitHub:
```text
$ git clone https://github.com/DanPorter/Dans_Diffraction.git
```

# Operation
### Read CIF file
```python
import Dans_Diffraction as dif
xtl = dif.Crystal('some_file.cif')
xtl.info() # print Crystal structure parameters
```