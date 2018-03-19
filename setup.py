from distutils.core import setup
setup(
	name = 'Dans_Diffraction',
	packages = ['Dans_Diffraction'],
	version = '1.0.0',
	description = 'Generate diffracted intensities from crystals',
	author = 'Dan Porter',
	author_email = 'd.g.porter@outlook.com',
	url = 'https://github.com/DanPorter/Dans_Diffraction',
	download_url = 'https://github.com/DanPorter/Dans_Diffraction/archive/0.1.tar.gz',
	keywords = ['crystal','cif','diffraction','crystallography','science'],
	classifiers = [
		'Programming Language :: Python',
		'Intended Audience :: Science/Research',
		'License :: GNU GENERAL PUBLIC LICENSE v3',
		'Programming Language :: Python :: 2.7',
		'Development Status :: 3 - Alpha', 
		],
	'install_requires'=['numpy','matplotlib','scipy']
	)
