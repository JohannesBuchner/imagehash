#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

long_description = ""
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='ImageHash',
    version='4.2.1',
    author='Johannes Buchner',
    author_email='buchner.johannes@gmx.at',
    py_modules=['imagehash'],
    data_files=[('images', ['tests/data/imagehash.png'])],
    scripts=['find_similar_images.py'],
    url='https://github.com/JohannesBuchner/imagehash',
    license='BSD 2-clause (see LICENSE file)',
    description='Image Hashing library',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        "numpy",
        "scipy",       # for phash
        "pillow",      # or PIL
        "PyWavelets",  # for whash
    ],
    test_suite='tests',
    tests_require=['pytest>=3'],
)
