# Dans_Diffraction
## Atomic Properties Data
This folder contains atomic properties & symmetry data used by Dans_Diffraction programs.

Full code documentation available [here](https://danporter.github.io/Dans_Diffraction/).

For comments, queries or bugs - email dan.porter@diamond.ac.uk


### Instructions
```text
$ ipython -i -m Dans_Diffraction
```

```python
>> import Dans_Diffraction as dif
>> print(dif.fc.atom_properties('Co', 'Weight'))
>> print(dif.fc.print_atom_properties('Fe'))
>> a = dif.fc.atom_properties() # table of all properties
```


### Datasources
The data is taken from a number of sources:
* Neutron scattering lengths: [NIST](http://www.ncnr.nist.gov/resources/n-lengths/)
* Neutron scattering lengths (Neutron Data Bookelet, 2003-2023)(default): [Periodic Table](https://github.com/pkienzle/periodictable)
* Neutron scattering lengths (V. F. Sears 1995): [ITC](https://it.iucr.org/C/), Table 4.4.4.1
* X-ray Form factor: [ITC](https://it.iucr.org/C/), p578, Table 6.1.1.4
* X-ray Form factor (Waasmaier & Kirfel, Acta Cryst. A 51, 416-431 (1995)): [diffpy](https://github.com/diffpy/libdiffpy/blob/master/src/runtime/f0_WaasKirf.dat)
* Magnetic Form Factor: [ILL](https://www.ill.eu/sites/ccsl/ffacts/ffactnode4.html)
* Element weights and radii: [Wikipedia](http://en.wikipedia.org/wiki/List_of_elements_by_atomic_properties)
* X-ray Mass Attenuation Coefficients: [NIST](https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients)
* X-ray Atomic Scattering Factors: [CXRO](http://henke.lbl.gov/optical_constants/asf.html)
* X-ray edges: [x-ray database](https://xdb.lbl.gov/Section1/Sec_1-1.html)
* Electron configurations: [Wikipedia](https://en.wikipedia.org/wiki/Electron_configurations_of_the_elements_(data_page))
* Space Group symmetry operations: [Bilbao Crystallographic Server](https://www.cryst.ehu.es/)


### Space Groups
Space group symmetry operations have been read from the [Bilbao Crystallographic Server](https://www.cryst.ehu.es/) 
and placed in a handy [json file](/SpaceGroups.json) under Dans_Diffraction/Data.

Space group information can be read from the file as follows:

```python
sg = dif.fc.spacegroup_symbol(61)  # space group number
print(sg['space group number'])
print(sg['space group name'])
print(sg['general positions'])
print(sg['positions wyckoff letter'])

# Operations can be made:
sym = dif.fc.gen_sym_pos(sg['general positions'], 0.1, 0, 0)
print(sym)
```

### Magnetic Space Groups
Magnetic space group symmetry operations have been read from the [Bilbao Crystallographic Server](https://www.cryst.ehu.es/) 
and placed in a handy [json file](/SpaceGroupsMagnetic.json) under Dans_Diffraction/Data.

Magnetic spacegroup information can be read from the file as follows:

```python
sg = dif.fc.spacegroup_symbol(61)  # space group number
sg['magnetic space groups']  # returns magnetic spacegroups associated with the spcaegroup
msg = dif.fc.spacegroup_magnetic(61.433)
print(msg['positions magnetic'])
```

### X-Ray Form Factors
X-Ray form factors calculated using the analytical form from the International Tables of Crystallography Vol. C, 
[Table 6.1.1.4](https://it.iucr.org/Cb/ch6o1v0001/table6o1o1o4.pdf).

For a particular reciprocal wave-vector (units of inverse angstroms), the form factor f(|Q|) can be calculated:

```python
q = np.arange(0,6,0.01)
f = dif.fc.xray_scattering_factor('Fe', Qmag=q)
```

![x-ray scattering factor](../../Screenshots/xray_scattering_factor.png)

### X-Ray Dispersion corrrections
X-Ray Atomic Scattering Factors are taken from the online tables at [CXRO](http://henke.lbl.gov/optical_constants/asf.html).
The dispersion corrections for resonant x-ray scattering are calculated using
<div align="center">f(Q, E) = f^0(|Q|) + f'(E) + if''(E)</div>
Where f^0(|Q|) is the standard form factor and f', f'' are the dispersion corrections. The values of f1 and f2 in the 
online tables can be converted to f', f'' using the following:
 <div align="center">f' = f1 - f^0(0)</div>
 <div align="center">f'' = -f2</div>
For a particular energy or energy range, f' and f'' can be calcualted, as well as the total scattering factor:

```python
en = np.arange(5, 10, 0.001)
# f'  and f''
f1, f2 = dif.fc.xray_dispersion_corrections('Co', en)
# Total: f0+f1+if2 (complex)
f = dif.fc.xray_scattering_factor_resonant('Co', Qmag=0, energy_kev=en)
```

### Neutron Scattering Lengths
Neutron scattering lengths of elements and isotopes has been extracted from several sources:

* [NIST Website](http://www.ncnr.nist.gov/resources/n-lengths/), with data originally from [V. F. Sears, Neutron News, Vol. 3, No. 3, 1992, pp. 29-37.](https://doi.org/10.1080/10448639208218770)
* [Periodic Table](https://github.com/pkienzle/periodictable), with data originally from [Neutron Data Booklet](https://www.ill.eu/fileadmin/user_upload/ILL/1_About_ILL/Documentation/NeutronDataBooklet.pdf), by A-J Dianoux, G. Lander (2003), with additions and corrections upto v1.7.0 (2023) 
* Table 4.4.4.1 in [International Tables Crystallography, Vol. C](https://it.iucr.org/C/) by V. F. Sears, 1995

Thie NIST website values were originally used and stored in the `Dans Element Properties.txt` file with other elemental data.
More recent values, including complex scattering lengths for isotopes are now included in `neutron_isotope_scattering_lengths.dat`.
Complex isotope scattering lengths from the table by Sears(1995) is also included, stored in `neutron_isotope_scattering_lengths_sears.dat`.

Neutron scattering lengths can be read using the built in functions:

```python

```