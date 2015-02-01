"""
Image hashing library
======================

Example:

>>> import Image
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
>>> for r in range(1, 30, 5):
...     rothash = imagehash.average_hash(Image.open('test.png').rotate(r))
...     print('Rotation by %d: %d Hamming difference' % (r, hash - rothash))
... 
Rotation by 1: 2 Hamming difference
Rotation by 6: 11 Hamming difference
Rotation by 11: 13 Hamming difference
Rotation by 16: 17 Hamming difference
Rotation by 21: 19 Hamming difference
Rotation by 26: 21 Hamming difference
>>>

"""

from PIL import Image
import numpy

"""
Try to import the scipy.fftpack module. If it is not available pHash will not work, but one can still use aHash and dHash.
"""
try:
	import scipy.fftpack
	havefftpack = True
except ImportError:
	havefftpack = False

	
def binary_array_to_hex(arr):
	h = 0
	s = []
	for i,v in enumerate(arr.flatten()):
		if v: h += 2**(i % 8)
		if (i % 8) == 7:
			s.append(hex(h)[2:].rjust(2, '0'))
			h = 0
	return "".join(s)

def binary_array_to_int(arr):
	return sum([2**(i % 8) for i,v in enumerate(arr.flatten()) if v])

"""
Hash encapsulation. Can be used for dictionary keys and comparisons.
"""
class ImageHash(object):
	def __init__(self, binary_array):
		self.hash = binary_array

	def __str__(self):
		return binary_array_to_hex(self.hash)

	def __repr__(self):
		return repr(self.hash)

	def __sub__(self, other):
		if other is None:
			raise TypeError('Other hash must not be None.')
		if self.hash.shape != other.hash.shape:
			raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
		return (self.hash != other.hash).sum()

	def __eq__(self, other):
		if other is None:
			return False
		return numpy.array_equal(self.hash, other.hash)

	def __ne__(self, other):
		if other is None:
			return False
		return not numpy.array_equal(self.hash, other.hash)

	def __hash__(self):
		return binary_array_to_int(self.hash)

def hex_to_hash(hexstr):
	l = []
	if len(hexstr) != 16:
		raise ValueError('The hex string has the wrong length')
	for i in range(8):
		h = hexstr[i*2:i*2+2]
		v = int("0x" + h, 16)
		l.append([v & 2**i > 0 for i in range(8)])
	return ImageHash(numpy.array(l))


"""
Average Hash computation

Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

@image must be a PIL instance.
"""
def average_hash(image, hash_size=8):
	image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
	pixels = numpy.array(image.getdata()).reshape((hash_size, hash_size))
	avg = pixels.mean()
	diff = pixels > avg
	# make a hash
	return ImageHash(diff)

"""
Perceptual Hash computation.

Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

@image must be a PIL instance.
"""
def phash(image, hash_size=32):
	if (havefftpack == False):
		import scipy.fftpack
		
	image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
	pixels = numpy.array(image.getdata(), dtype=numpy.float).reshape((hash_size, hash_size))
	dct = scipy.fftpack.dct(pixels)
	dctlowfreq = dct[:8, 1:9]
	avg = dctlowfreq.mean()
	diff = dctlowfreq > avg
	return ImageHash(diff)

"""
Difference Hash computation.

following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

@image must be a PIL instance.
"""
def dhash(image, hash_size=8):
	image = image.convert("L").resize((hash_size + 1, hash_size), Image.ANTIALIAS)
	pixels = numpy.array(image.getdata(), dtype=numpy.float).reshape((hash_size + 1, hash_size))
	# compute differences
	diff = pixels[1:,:] > pixels[:-1,:]
	return ImageHash(diff)



__dir__ = [average_hash, dhash, phash, ImageHash]

