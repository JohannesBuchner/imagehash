try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = ""
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='ImageHash',
    version='0.4',
    author='Johannes Buchner',
    author_email='buchner.johannes@gmx.at',
    packages=['imagehash'],
    scripts=['find_similar_images.py'],
    url='https://github.com/JohannesBuchner/imagehash',
    license='LICENSE',
    description='Image Hashing library',
    long_description=long_description,
    install_requires=[
        "scipy",
        "numpy",
        "pillow", # or PIL
    ],
)

