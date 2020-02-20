from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='Dans_Diffraction',
    packages=['Dans_Diffraction', 'Dans_Diffraction.tkgui'],
    version='1.5.2',
    description='Generate diffracted intensities from crystals',
    long_description_content_type='text/markdown',
    long_description=readme(),
    author='Dan Porter',
    author_email='d.g.porter@outlook.com',
    url='https://github.com/DanPorter/Dans_Diffraction',
    keywords=[
        'crystal', 'cif', 'diffraction', 'crystallography', 'science',
        'x-ray','neutron','resonant', 'magnetic', 'magnetism', 'multiple scattering', 
        'fdmnes', 'super structure'
        ],
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 3 - Alpha',
        ],
    install_requires=['numpy', 'matplotlib', 'scipy'],
    package_data={'': ['data/*.txt', 'data/*.dat', 'Structures/*.cif', 'Structures/*.mcif']}
    )
