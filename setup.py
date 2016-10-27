import os
import os.path

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


NAME = 'imagehash'
DESCRIPTION = 'Image Hashing Library'
DIR = os.path.abspath(os.path.dirname(__file__))


package_data = {
    NAME: [os.path.join('tests', 'data', '*')],
}


def extract_version():
    version = None
    fname = os.path.join(DIR, NAME, '__init__.py')
    with open(fname) as fi:
        for line in fi:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Removes quotation.
                break
    return version


def extract_packages():
    packages = []
    root = os.path.join(DIR, NAME)
    offset = len(os.path.dirname(root)) + 1
    for dpath, dnames, fnames in os.walk(root):
        if os.path.exists(os.path.join(dpath, '__init__.py')):
            package = dpath[offset:].replace(os.path.sep, '.')
            packages.append(package)
    return packages


def extract_description():
    description = DESCRIPTION
    fname = os.path.join(DIR, 'README.rst')
    if os.path.isfile(fname):
        with open(fname, 'r') as fi:
            description = fi.read()
    return description


setup_args = dict(
    name=NAME,
    version=extract_version(),
    author='Johannes Buchner',
    author_email='buchner.johannes@gmx.at',
    packages=extract_packages(),
    package_data=package_data,
    scripts=['find_similar_images.py'],
    url='https://github.com/JohannesBuchner/imagehash',
    license='LICENSE',
    description=DESCRIPTION,
    long_description=extract_description(),
    install_requires=[
        'six',         # for testing
        'pep8',        # for testing
        'numpy',
        'scipy',       # for phash
        'pillow',      # or PIL
        'PyWavelets',  # for whash
    ],
    test_suite = '{}.tests'.format(NAME),
)


if __name__ == '__main__':
    setup(**setup_args)
