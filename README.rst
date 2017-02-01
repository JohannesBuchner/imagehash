ImageHash
===========

A image hashing library written in Python. ImageHash supports:

* average hashing (`aHash`_)
* perception hashing (`pHash`_)
* difference hashing (`dHash`_)
* wavelet hashing (`wHash`_)

|Travis|_ |Coveralls|_

Requirements
-------------
Based on PIL/Pillow Image, numpy and scipy.fftpack (for pHash)
Easy installation through `pypi`_.

Basic usage
------------
::

	>>> from PIL import Image
	>>> import imagehash
	>>> hash = imagehash.average_hash(Image.open('test.png'))
	>>> print(hash)
	d879f8f89b1bbf
	>>> otherhash = imagehash.average_hash(Image.open('other.bmp'))
	>>> print(otherhash)
	ffff3720200ffff
	>>> print(hash == otherhash)
	False
	>>> print(hash - otherhash)
	36

The demo script **find_similar_images** illustrates how to find similar images in a directory.

Source hosted at github: https://github.com/JohannesBuchner/imagehash

.. _aHash: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
.. _pHash: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
.. _dHash: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
.. _wHash: https://fullstackml.com/2016/07/02/wavelet-image-hash-in-python/
.. _pypi: https://pypi.python.org/pypi/ImageHash

Changelog
----------

* 3.0: dhash had a bug: It computed pixel differences vertically, not horizontally.
       I modified it to follow `dHash`_. The old function is available as dhash_vertical.

* 2.0: added whash

* 1.0: initial ahash, dhash, phash implementations.


.. |Travis| image:: https://travis-ci.org/JohannesBuchner/imagehash.svg?branch=master
.. _Travis: https://travis-ci.org/JohannesBuchner/imagehash

.. |Coveralls| image:: https://coveralls.io/repos/github/JohannesBuchner/imagehash/badge.svg
.. _Coveralls: https://coveralls.io/github/JohannesBuchner/imagehash
