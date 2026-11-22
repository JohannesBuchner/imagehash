#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
	from setuptools import setup
except BaseException:
	from distutils.core import setup

long_description = ''
with open('README.rst', encoding='utf-8') as f:
	long_description = f.read()

setup(
	name='ImageHash',
	version='4.3.2',
	author='Johannes Buchner',
	author_email='buchner.johannes@gmx.at',
	packages=['imagehash'],
	package_data={'imagehash': ['py.typed']},
	data_files=[('images', ['tests/data/imagehash.png'])],
	scripts=['find_similar_images.py'],
	url='https://github.com/JohannesBuchner/imagehash',
	project_urls={
		'Source': 'https://github.com/JohannesBuchner/imagehash',
		'Tracker': 'https://github.com/JohannesBuchner/imagehash/issues',
	},
	license='2-clause BSD License',
	description='Image Hashing library',
	long_description=long_description,
	long_description_content_type='text/x-rst',
	python_requires='>=3.8',
	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'License :: OSI Approved :: BSD License',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3 :: Only',
		'Topic :: Multimedia :: Graphics',
		'Typing :: Typed',
	],
	install_requires=[
		'numpy',
		'scipy',		# for phash
		'pillow',		# or PIL
		'PyWavelets',  # for whash
	],
	test_suite='tests',
	tests_require=['pytest>=3'],
)
