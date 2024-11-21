from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='Dans_Diffraction',
    packages=['Dans_Diffraction', 'Dans_Diffraction.tkgui'],
    version='3.3.2',
    description='Generate diffracted intensities from crystals',
    long_description_content_type='text/markdown',
    long_description=readme(),
    author='Dan Porter',
    author_email='d.g.porter@outlook.com',
    url='https://github.com/DanPorter/Dans_Diffraction',
    keywords=[
        'crystal', 'cif', 'diffraction', 'crystallography', 'science',
        'x-ray', 'neutron', 'resonant', 'magnetic', 'magnetism', 'multiple scattering',
        'fdmnes', 'super structure', 'spacegroup', 'space group', 'diffractometer'
        ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 3 - Alpha',
        ],
    install_requires=['numpy', 'matplotlib'],
    package_data={'': ['data/*.txt', 'data/*.dat', 'data/*.json', 'data/*.npy', 'Structures/*.cif', 'Structures/*.mcif']}
    )
