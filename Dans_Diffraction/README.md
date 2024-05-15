# Dans_Diffraction
Reads crystallographic cif files and simulates diffraction

By Dan Porter, Diamond Light Source Ltd. 2024

## Crystal Structure
The program creates a "Crystal" object with certain properties, attributes and functions useful for diffraction studies.
A *Crystal* object is based on a single unit cell - an arrangement of atomic positions that, when repeated infinitly in 
3 dimensions creates a perfect crystal lattice.

The *Crystal* object is typically generated from a CIF ([Crystallographic Infromation File](https://www.iucr.org/resources/cif)), 
though can also be built from scratch. From the CIF, the lattice parameters, crystal symmetries and atomic site
properties are read. The atomic positions are repeated at all symmetric positions to generate the full basis structure,
for which all functions work on.

The *Crystal* object has the structure:
```Python
xtl = dif.Crystal('file.cif')
xtl.Cell # Cell properties - lattice parameters, generation of direct and recriprocal lattice
xtl.Symmetry # Symmetry properties - contains symmetry operations and functions to apply symmetry to positions & reflections.
xtl.Atoms # Symmetically unique Atomic sites - describes atoms at symmetric sites
xtl.Structure # All Atomic sites - describes atoms at all positions in unit cell
xtl.Plot  # Plot functions - Various Plotting methods to produce graphed output
xtl.Scatter # Scattering & Diffraction functions - calculate various types of scattering based on the crystal structure.
xtl.Properties # Various functions - calculate various crystal properties
# Other available functions:
xtl.generate_structure() # Generates the Structure by applying symmetry operations from Symmetry to atomic sites in Atoms.
xtl.generate_lattice() # Creates a new Crystal object, repeated in different directions.
xtl.generate_superstructure() # Creates a new Superstructure object, based on Crystal but linked to the parent structure.
xtl.write_cif() # Write a new CIF with the current atomic site properties.
```

### Cell
Contains lattice parameters and unit cell. Provides tools to convert between orthogonal and lattice bases in real and reciprocal space.

Parameters:
```python
# Lattice parameters:
xtl.Cell.a
xtl.Cell.b
xtl.Cell.c
xtl.Cell.alpha
xtl.Cell.beta
xtl.Cell.gamma
```

Selected functions (see internal documentation for more):
 ```python
xtl.Cell.latt([2.85,2.85,10.8,90,90,120]) # Define the lattice parameters from a list
xtl.Cell.tth([0,0,12],energy_kev=8.0) # Calculate the two-theta of a reflection
xtl.Cell.lp() # Returns the current lattice parameters
xtl.Cell.volume() # Returns the calculated volume in A^3
xtl.Cell.Bmatrix() # Calculate the Busing and Levy B matrix from a real space UV
xtl.Cell.Qmag() # Returns the magnitude of wave-vector transfer of [h,k,l], in A-1
xtl.Cell.UV() # Returns the unit cell as a [3x3] array, [A,B,C]
xtl.Cell.UVstar() # Returns the reciprocal unit cell as a [3x3] array, [A*,B*,C*]
xtl.Cell.max_hkl(energy_kev, max_angle) # Returns the maximum index of h, k and l for a given energy
xtl.Cell.all_hkl(energy_kev, max_angle) # Returns an array of all (h,k,l) reflections at this energy
xtl.Cell.angle(hkl1, hkl2) # Return the angle between two reflections
xtl.Cell.calculateQ(hkl) # Convert coordinates [h,k,l], in the basis of the reciprocal lattice
xtl.Cell.calculateR(uvw) # Convert coordinates [u,v,w], in the basis of the unit cell
xtl.Cell.dspace(hkl) # Calculate the d-spacing in A
xtl.Cell.find_close_reflections() # Find reflections near to given HKL for a given two-theta or reflection angle
xtl.Cell.indexQ(xyz) # Convert reciprocal space coordinates [qx,qy,qz], to hkl
xtl.Cell.indexR(xyz) # Convert real space coorinates [x,y,z], fractional coordinates to uvw
xtl.Cell.powder_average(hkl) # Returns the powder average correction for the given hkl
xtl.Cell.reciprocal_space_plane() # Returns positions within a reciprocal space plane
```

### Symmetry
Contains symmetry information about the crystal, including the symmetry operations. 
Tabulated spacegroup symmetries are found in [data](./data/README.md).

Parameters:

```python
xtl.Symmetry.spacegroup_symbol  # spacegroup name
xtl.Symmetry.spacegroup_number  # spacegroup number
xtl.Symmetry.symmetry_operations  # list of symmetry operations (list of strings e.g. ['-x,-y,-z'])
xtl.Symmetry.symmetry_operations_magnetic  # list of magnetic symmetry operations (list of strings e.g. ['-x,-y,-z'])
```

Selected functions (see internal documentation for more):
```python
xtl.Symmetry.changesym(idx, new) # change a symmetry operation
xtl.Symmetry.invert_magsym(idx) #  Invert the time symmetry of a magnetic symmetry

xtl.Symmetry.symmetric_coordinates(uvw) # Returns array of symmetric coordinates
xtl.Symmetry.symmetric_coordinate_operations(uvw) # Returns array of symmetric operations for given position
xtl.Symmetry.print_symmetric_coordinate_operations(uvw) # Returns str of symmetric operations for given position

xtl.Symmetry.symmetric_reflections(hkl) # Returns array of symmetric reflection indices
xtl.Symmetry.symmetric_reflections_count(hkl) # Returns array of symmetric reflection indices,
xtl.Symmetry.symmetric_reflections_unique(hkl) # Returns array of symmetric reflection indices, with identical reflections removed
xtl.Symmetry.symmetric_intensity(hkl, inten) # Returns symmetric reflections with summed intensities of repeated reflections
xtl.Symmetry.average_symmetric_intensity # Return a list of reflections with symmetric reflections removed
xtl.Symmetry.remove_symmetric_reflections(hkl) # Return a list of reflections with symmetric reflections removed
xtl.Symmetry.reflection_multiplyer(hkl) # Returns the number of symmetric reflections for each hkl (multiplicity)
xtl.Symmetry.print_symmetric_vectors(hkl) # Return str of symmetric vectors
xtl.Symmetry.is_symmetric_reflection(hkl1, hkl2) # Check if reflection 1 is a symmetric equivalent of reflection 2

xtl.Symmetry.load_spacegroup(sg_number) # Load symmetry operations from a spacegroup
xtl.Symmetry.load_magnetic_spacegroup(msg_number) # Load symmetry operations from a magnetic spacegroup
xtl.Symmetry.print_magnetic_spacegroups(sg_number) # Return str of available magnetic spacegroups for this spacegroup
xtl.Symmetry.print_subgroups(sg_number) # Return str of subgroups of this spacegroup
```

### Atoms & Structure
Contains properties of atoms within the crystal. xtl.Atoms contains non-equivalent symmetric atomic sites,
xtl.Structrue contains all atomic positions in the unit cell.

Each atom has properties:

|   Property   |      |  
| ------------- | ------------- |
|   u   | Fractional atomic coordinates along direction of **a**  |
|   v  | Fractional atomic coordinates along direction of **b**   |
|   w   | Fractional atomic coordinates along direction of **c**   |
|   type    | element species, given as element name, e.g. 'Fe'   |
|   label   |   Name of atomic position, e.g. 'Fe1'   |
|   occupancy   |   Occupancy of this atom at this atomic position   |
|   uiso     |   atomic displacement factor (ADP) <u^2>   |
|   mxmymz   |   magnetic moment direction [x,y,z]   |

Functions available:
```python
xtl.Atoms.addatom() # Adds a new atom
xtl.Atoms.changeatom() # Change an atom's properties
xtl.Atoms.removeatom(idx) # Removes atom number idx from the list
xtl.Atoms.check() # Checks the validity of the contained attributes
xtl.Atoms.get() # Returns parameters: uvw, type, label, occupancy, uiso, mxmymz = xtl.Atoms.get()
xtl.Atoms.ismagnetic() # Returns True if any ions have magnetic moments assigned
xtl.Atoms.mass_fraction() # Return the mass fraction per element
xtl.Atoms.weight() # Calculate the molecular weight in g/mol of all the atoms
```

### Plot
Plotting functions for the Crystal Object. Uses matplotlib for creating figures and plots.

Selected functions (see internal documentation for more):
```python
# Plot crystal structure
xtl.Plot.plot_3Dlattice() # Plot lattice points in 3D
xtl.Plot.plot_3Dpolarisation(hkl) # Plots the scattering vectors for a particular azimuth
xtl.Plot.plot_crystal # Plot the atomic cell in 3D
xtl.Plot.plot_layers() # Separate the structure into layers along the chosen axis and plot the atoms in each layer.

# Powder diffraction
xtl.Plot.simulate_powder() # Generates a powder pattern, plots in a new figure with labels

# Single crystal diffraction
xtl.Plot.simulate_intensity_cut() # Plot a cut through reciprocal space, visualising the intensity
xtl.Plot.quick_intensity_cut() # Plot a cut through reciprocal space, visualising the intensity as different sized markers
xtl.Plot.simulate_hk0(l) # Plots the hk(L) layer of reciprocal space
xtl.Plot.simulate_h0l(k) # Plots the h(K)l layer of reciprocal space
xtl.Plot.simulate_0kl(h) # Plots the (H)kl layer of reciprocal space
xtl.Plot.simulate_hhl(h) # Plots the hhl layer of reciprocal space
xtl.Plot.simulate_ewald_coverage() # Plot ewald space coverage within a particular scattering plane

# Magnetic x-ray scattering
xtl.Plot.simulate_azimuth(hkl) # Simulate azimuthal scan of magnetic resonant x-ray scattering
xtl.Plot.simulate_azimuth_resonant(hkl) # Simulate azimuthal scan of magnetic resonant x-ray scattering
xtl.Plot.simulate_azimuth_nonresonant(hkl) # Simulate azimuthal scan of non-resonant magnetic x-ray scattering
xtl.Plot.simulate_polarisation_resonant(hkl) # Simulate azimuthal scan of resonant x-ray scattering
xtl.Plot.simulate_polarisation_nonresonant(hkl) # Simulate azimuthal scan of resonant x-ray scattering

# Multiple-Scattering
xtl.Plot.plot_ms_azimuth(hkl) # Run the multiple scattering code and plot the result
xtl.Plot.plot_multiple_scattering # Run the multiple scattering code and plot the result

# Tensor Scattering
xtl.Plot.tensor_scattering_azimuth # Plot tensor scattering intensities
xtl.Plot.tensor_scattering_stokes # Return tensor scattering intensities for non-standard polarisation
```

### Scattering
Simulate diffraction from the crystal structure, for various scattering types, including:

|   Name   |   Explanation   |
|   ----   |   -----------    |
| 'xray' | X-Ray diffraction, using atomic form factors |
| 'neutron' | Neutron difraction, using neutron scattering lengths. |
| 'neutron magnetic' | Magnetic neutron diffraction |
| 'xray magnetic' | Non-resonant x-ray magnetic diffraction |
| 'xray resonant' | Resonant x-ray magnetic diffraction |

Functions calculate the complex structure factor based on the equation:

<p style="text-align: center;">structure_factor = sum_i( sf.occ.dw.phase )</p>

Scattering factors and scattering lengths for available elements are found in [data](./data/README.md).

Setup the scattering attributes with the *xtl.Scatter.setup_scatter(parameter)* function:

|   Parameter   |   Explanation   |
|  ----------   |   -----------   |
|    type         |  'xray','neutron','xray magnetic','neutron magnetic','xray resonant'  |
|    energy_kev   |  radiation energy in keV  |
|    wavelength_a |  radiation wavelength in Angstrom  |
|    powder_units |  units to use when displaying/ plotting ['twotheta', 'd',' 'q']  |
|    min_twotheta |  minimum detector (two-theta) angle  |
|    max_twotheta |  maximum detector (two-theta) angle  |
|    min_theta    |  minimum sample angle = -opening angle  |
|    max_theta    |  maximum sample angle = opening angle  |
|    theta_offset |  sample offset angle  |
|    specular     | [h,k,l] : reflections normal to sample surface  |
|    parallel     | [h,k,l] : reflections normal to sample surface  |


Selected functions (see internal documentation for more):
```python
# Scattering angles
xtl.Scatter.hkl(hkl) # Calculate the two-theta and intensity of the given HKL, display the result
xtl.Scatter.hkl_reflection(hkl) # Calculate the theta, two-theta and intensity of the given HKL in reflection geometry, display the result
xtl.Scatter.hkl_transmission(hkl) # Calculate the theta, two-theta and intensity of the given HKL in transmission geometry, display the result

# Diffracted intensities
xtl.Scatter.structure_factor(hkl)  # Calculate the complex structure factor for the given HKL
xtl.Scatter.intensity(hkl) # Calculate the squared structure factor for the given HKL
xtl.Scatter.x_ray(hkl) # Calculate the squared structure factor for the given HKL, using x-ray scattering factors
xtl.Scatter.x_ray_fast(hkl) # Calculate the squared structure factor for the given HKL, using atomic number as scattering length
xtl.Scatter.xray_magnetic(hkl) # Calculate the non-resonant magnetic component of the structure factor 
xtl.Scatter.xray_nonresonant_magnetic(hkl) # Calculate the non-resonant magnetic component of the structure factor
xtl.Scatter.xray_resonant(hkl) # Calculate structure factors using resonant scattering factors in the dipolar approximation
xtl.Scatter.xray_resonant_magnetic(hkl) # Calculate the non-resonant magnetic component of the structure factor
xtl.Scatter.xray_resonant_scattering_factor(hkl) # Calcualte fxres, the resonant x-ray scattering factor
xtl.Scatter.neutron(hkl) # Calculate the squared structure factor for the given HKL, using neutron scattering length
xtl.Scatter.magnetic_neutron(hkl) # Calculate the magnetic component of the structure factor for the given HKL, using neutron rules and form factor
xtl.Scatter.generate_intensity_cut() # Generate a cut through reciprocal space, returns an array with centred reflections

# Powder Scattering
xtl.Scatter.powder() # Generates array of intensities along a spaced grid, equivalent to a powder pattern.
xtl.Scatter.powder_correction(hkl) # Calculate the squared structure factor for the given HKL, using neutron scattering length

# Multiple Scattering
xtl.Scatter.ms_azimuth(hkl, energy_kev) # Returns an azimuthal dependence at a particular energy
xtl.Scatter.multiple_scattering(hkl) # Run multiple scattering code, return the result.

# Display lists of reflections
xtl.Scatter.print_all_reflections() # Return str of all allowed reflections at this energy
xtl.Scatter.find_close_reflections(hkl) # Find and print list of reflections close to the given one 
xtl.Scatter.print_atomic_contributions(hkl) # Prints the atomic contributions to the structure factor
xtl.Scatter.print_intensity(hkl) # Print intensities calcualted in different ways
xtl.Scatter.print_ref_reflections() # Prints a list of all allowed reflections at this energy in the reflection geometry
xtl.Scatter.print_tran_reflections() # Prints a list of all allowed reflections at this energy in the transmission geometry
xtl.Scatter.print_symmetric_reflections(hkl) # Prints equivalent reflections
xtl.Scatter.print_symmetry_contributions(hkl) # Prints the symmetry contributions to the structure factor for each atomic site
```

### Properties
Properties functions for the *Crystal* Object. Atomic properties used here are found in [data](./data/README.md).

Selected functions (see internal documentation for more):
```python
absorption() # Returns the sample absorption coefficient in um^-1 at the requested energy in keV
atomic_neighbours(idx) # Returns the relative positions of atoms within a radius of the selected atom
density() # Return the density in g/cm
diamagnetic_susceptibility() # Calculate diamagnetic contribution to susceptibility
latex_table() # Return latex table of structure properties from CIF
molcharge() # Generate molecular charge composition of crystal
molfraction() # Display the molecular weight of a compound and atomic fractions
molname() # Generate molecular name of crystal
volume() # Returns the volume in A^3
weight() # Return the molecular weight in g/mol
xray_edges() # Returns the x-ray absorption edges available and their energies
```

