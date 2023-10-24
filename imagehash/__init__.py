"""
Perceptual image hashing library

Example:

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
>>> for r in range(1, 30, 5):
...	rothash = imagehash.average_hash(Image.open('test.png').rotate(r))
...	print('Rotation by %d: %d Hamming difference' % (r, hash - rothash))
...
Rotation by 1: 2 Hamming difference
Rotation by 6: 11 Hamming difference
Rotation by 11: 13 Hamming difference
Rotation by 16: 17 Hamming difference
Rotation by 21: 19 Hamming difference
Rotation by 26: 21 Hamming difference
>>>
"""


import sys

import numpy
from PIL import Image, ImageFilter
try:
	ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
	# deprecated in pillow 10
	# https://pillow.readthedocs.io/en/stable/deprecations.html
	ANTIALIAS = Image.ANTIALIAS

__version__ = '4.3.1'

"""
You may copy this file, if you keep the copyright information below:


Copyright (c) 2013-2022, Johannes Buchner
https://github.com/JohannesBuchner/imagehash

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


def _binary_array_to_hex(arr):
	"""
	internal function to make a hex string out of a binary array.
	"""
	bit_string = ''.join(str(b) for b in 1 * arr.flatten())
	width = int(numpy.ceil(len(bit_string) / 4))
	return '{:0>{width}x}'.format(int(bit_string, 2), width=width)


class ImageHash:
	"""
	Hash encapsulation. Can be used for dictionary keys and comparisons.
	"""

	def __init__(self, binary_array):
		# type: (NDArray) -> None
		self.hash = binary_array  # type: NDArray

	def __str__(self):
		return _binary_array_to_hex(self.hash.flatten())

	def __repr__(self):
		return repr(self.hash)

	def __sub__(self, other):
		# type: (ImageHash) -> int
		if other is None:
			raise TypeError('Other hash must not be None.')
		if self.hash.size != other.hash.size:
			raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
		return numpy.count_nonzero(self.hash.flatten() != other.hash.flatten())

	def __eq__(self, other):
		# type: (object) -> bool
		if other is None:
			return False
		return numpy.array_equal(self.hash.flatten(), other.hash.flatten())  # type: ignore

	def __ne__(self, other):
		# type: (object) -> bool
		if other is None:
			return False
		return not numpy.array_equal(self.hash.flatten(), other.hash.flatten())  # type: ignore

	def __hash__(self):
		# this returns a 8 bit integer, intentionally shortening the information
		return sum([2**(i % 8) for i, v in enumerate(self.hash.flatten()) if v])

	def __len__(self):
		# Returns the bit length of the hash
		return self.hash.size


# dynamic code for typing
try:
	# specify allowed values if possible (py3.8+)
	from typing import Literal
	WhashMode = Literal['haar', 'db4']  # type: ignore
except ImportError:
	WhashMode = str  # type: ignore

try:
	# enable numpy array typing (py3.7+)
	import numpy.typing
	NDArray = numpy.typing.NDArray[numpy.bool_]
except (AttributeError, ImportError):
	NDArray = list  # type: ignore

# type of Callable
if sys.version_info >= (3, 3):
	if sys.version_info >= (3, 9, 0) and sys.version_info <= (3, 9, 1):
		# https://stackoverflow.com/questions/65858528/is-collections-abc-callable-bugged-in-python-3-9-1
		from typing import Callable
	else:
		from collections.abc import Callable
	try:
		MeanFunc = Callable[[NDArray], float]
		HashFunc = Callable[[Image.Image], ImageHash]
	except TypeError:
		MeanFunc = Callable  # type: ignore
		HashFunc = Callable  # type: ignore
# end of dynamic code for typing


def hex_to_hash(hexstr):
	# type: (str) -> ImageHash
	"""
	Convert a stored hash (hex, as retrieved from str(Imagehash))
	back to a Imagehash object.

	Notes:
	1. This algorithm assumes all hashes are either
			bidimensional arrays with dimensions hash_size * hash_size,
			or onedimensional arrays with dimensions binbits * 14.
	2. This algorithm does not work for hash_size < 2.
	"""
	hash_size = int(numpy.sqrt(len(hexstr) * 4))
	# assert hash_size == numpy.sqrt(len(hexstr)*4)
	binary_array = '{:0>{width}b}'.format(int(hexstr, 16), width=hash_size * hash_size)
	bit_rows = [binary_array[i:i + hash_size] for i in range(0, len(binary_array), hash_size)]
	hash_array = numpy.array([[bool(int(d)) for d in row] for row in bit_rows])
	return ImageHash(hash_array)


def hex_to_flathash(hexstr, hashsize):
	# type: (str, int) -> ImageHash
	hash_size = int(len(hexstr) * 4 / (hashsize))
	binary_array = '{:0>{width}b}'.format(int(hexstr, 16), width=hash_size * hashsize)
	hash_array = numpy.array([[bool(int(d)) for d in binary_array]])[-hash_size * hashsize:]
	return ImageHash(hash_array)


def hex_to_multihash(hexstr):
	# type: (str) -> ImageMultiHash
	"""
	Convert a stored multihash (hex, as retrieved from str(ImageMultiHash))
	back to an ImageMultiHash object.

	This function is based on hex_to_hash so the same caveats apply. Namely:

	1. This algorithm assumes all hashes are either
			bidimensional arrays with dimensions hash_size * hash_size,
			or onedimensional arrays with dimensions binbits * 14.
	2. This algorithm does not work for hash_size < 2.
	"""
	split = hexstr.split(',')
	hashes = [hex_to_hash(x) for x in split]
	return ImageMultiHash(hashes)


def old_hex_to_hash(hexstr, hash_size=8):
	# type: (str, int) -> ImageHash
	"""
	Convert a stored hash (hex, as retrieved from str(Imagehash))
	back to a Imagehash object. This method should be used for
	hashes generated by ImageHash up to version 3.7. For hashes
	generated by newer versions of ImageHash, hex_to_hash should
	be used instead.
	"""
	arr = []
	count = hash_size * (hash_size // 4)
	if len(hexstr) != count:
		emsg = 'Expected hex string size of {}.'
		raise ValueError(emsg.format(count))
	for i in range(count // 2):
		h = hexstr[i * 2:i * 2 + 2]
		v = int('0x' + h, 16)
		arr.append([v & 2**i > 0 for i in range(8)])
	return ImageHash(numpy.array(arr))


def average_hash(image, hash_size=8, mean=numpy.mean):
	# type: (Image.Image, int, MeanFunc) -> ImageHash
	"""
	Average Hash computation

	Implementation follows https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	Step by step explanation: https://web.archive.org/web/20171112054354/https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/ # noqa: E501

	@image must be a PIL instance.
	@mean how to determine the average luminescence. can try numpy.median instead.
	"""
	if hash_size < 2:
		raise ValueError('Hash size must be greater than or equal to 2')

	# reduce size and complexity, then convert to grayscale
	image = image.convert('L').resize((hash_size, hash_size), ANTIALIAS)

	# find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
	pixels = numpy.asarray(image)
	avg = mean(pixels)

	# create string of bits
	diff = pixels > avg
	# make a hash
	return ImageHash(diff)


def phash(image, hash_size=8, highfreq_factor=4):
	# type: (Image.Image, int, int) -> ImageHash
	"""
	Perceptual Hash computation.

	Implementation follows https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	@image must be a PIL instance.
	"""
	if hash_size < 2:
		raise ValueError('Hash size must be greater than or equal to 2')

	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = image.convert('L').resize((img_size, img_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
	dctlowfreq = dct[:hash_size, :hash_size]
	med = numpy.median(dctlowfreq)
	diff = dctlowfreq > med
	return ImageHash(diff)


def phash_simple(image, hash_size=8, highfreq_factor=4):
	# type: (Image.Image, int, int) -> ImageHash
	"""
	Perceptual Hash computation.

	Implementation follows https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	@image must be a PIL instance.
	"""
	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = image.convert('L').resize((img_size, img_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(pixels)
	dctlowfreq = dct[:hash_size, 1:hash_size + 1]
	avg = dctlowfreq.mean()
	diff = dctlowfreq > avg
	return ImageHash(diff)


def dhash(image, hash_size=8):
	# type: (Image.Image, int) -> ImageHash
	"""
	Difference Hash computation.

	following https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

	computes differences horizontally

	@image must be a PIL instance.
	"""
	# resize(w, h), but numpy.array((h, w))
	if hash_size < 2:
		raise ValueError('Hash size must be greater than or equal to 2')

	image = image.convert('L').resize((hash_size + 1, hash_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between columns
	diff = pixels[:, 1:] > pixels[:, :-1]
	return ImageHash(diff)


def dhash_vertical(image, hash_size=8):
	# type: (Image.Image, int) -> ImageHash
	"""
	Difference Hash computation.

	following https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

	computes differences vertically

	@image must be a PIL instance.
	"""
	# resize(w, h), but numpy.array((h, w))
	image = image.convert('L').resize((hash_size, hash_size + 1), ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between rows
	diff = pixels[1:, :] > pixels[:-1, :]
	return ImageHash(diff)


def whash(image, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True):
	# type: (Image.Image, int, int | None, WhashMode, bool) -> ImageHash
	"""
	Wavelet Hash computation.

	based on https://www.kaggle.com/c/avito-duplicate-ads-detection/

	@image must be a PIL instance.
	@hash_size must be a power of 2 and less than @image_scale.
	@image_scale must be power of 2 and less than image size. By default is equal to max
					power of 2 for an input image.
	@mode (see modes in pywt library):
					'haar' - Haar wavelets, by default
					'db4' - Daubechies wavelets
	@remove_max_haar_ll - remove the lowest low level (LL) frequency using Haar wavelet.
	"""
	import pywt
	if image_scale is not None:
		assert image_scale & (image_scale - 1) == 0, 'image_scale is not power of 2'
	else:
		image_natural_scale = 2**int(numpy.log2(min(image.size)))
		image_scale = max(image_natural_scale, hash_size)

	ll_max_level = int(numpy.log2(image_scale))

	level = int(numpy.log2(hash_size))
	assert hash_size & (hash_size - 1) == 0, 'hash_size is not power of 2'
	assert level <= ll_max_level, 'hash_size in a wrong range'
	dwt_level = ll_max_level - level

	image = image.convert('L').resize((image_scale, image_scale), ANTIALIAS)
	pixels = numpy.asarray(image) / 255.

	# Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
	if remove_max_haar_ll:
		coeffs = pywt.wavedec2(pixels, 'haar', level=ll_max_level)
		coeffs = list(coeffs)
		coeffs[0] *= 0
		pixels = pywt.waverec2(coeffs, 'haar')

	# Use LL(K) as freq, where K is log2(@hash_size)
	coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
	dwt_low = coeffs[0]

	# Subtract median and compute hash
	med = numpy.median(dwt_low)
	diff = dwt_low > med
	return ImageHash(diff)


def colorhash(image, binbits=3):
	# type: (Image.Image, int) -> ImageHash
	"""
	Color Hash computation.

	Computes fractions of image in intensity, hue and saturation bins:

	* the first binbits encode the black fraction of the image
	* the next binbits encode the gray fraction of the remaining image (low saturation)
	* the next 6*binbits encode the fraction in 6 bins of saturation, for highly saturated parts of the remaining image
	* the next 6*binbits encode the fraction in 6 bins of saturation, for mildly saturated parts of the remaining image

	@binbits number of bits to use to encode each pixel fractions
	"""

	# bin in hsv space:
	intensity = numpy.asarray(image.convert('L')).flatten()
	h, s, v = [numpy.asarray(v).flatten() for v in image.convert('HSV').split()]
	# black bin
	mask_black = intensity < 256 // 8
	frac_black = mask_black.mean()
	# gray bin (low saturation, but not black)
	mask_gray = s < 256 // 3
	frac_gray = numpy.logical_and(~mask_black, mask_gray).mean()
	# two color bins (medium and high saturation, not in the two above)
	mask_colors = numpy.logical_and(~mask_black, ~mask_gray)
	mask_faint_colors = numpy.logical_and(mask_colors, s < 256 * 2 // 3)
	mask_bright_colors = numpy.logical_and(mask_colors, s > 256 * 2 // 3)

	c = max(1, mask_colors.sum())
	# in the color bins, make sub-bins by hue
	hue_bins = numpy.linspace(0, 255, 6 + 1)
	if mask_faint_colors.any():
		h_faint_counts, _ = numpy.histogram(h[mask_faint_colors], bins=hue_bins)
	else:
		h_faint_counts = numpy.zeros(len(hue_bins) - 1)
	if mask_bright_colors.any():
		h_bright_counts, _ = numpy.histogram(h[mask_bright_colors], bins=hue_bins)
	else:
		h_bright_counts = numpy.zeros(len(hue_bins) - 1)

	# now we have fractions in each category (6*2 + 2 = 14 bins)
	# convert to hash and discretize:
	maxvalue = 2**binbits
	values = [min(maxvalue - 1, int(frac_black * maxvalue)), min(maxvalue - 1, int(frac_gray * maxvalue))]
	for counts in list(h_faint_counts) + list(h_bright_counts):
		values.append(min(maxvalue - 1, int(counts * maxvalue * 1. / c)))
	# print(values)
	bitarray = []
	for v in values:
		bitarray += [v // (2**(binbits - i - 1)) % 2**(binbits - i) > 0 for i in range(binbits)]
	return ImageHash(numpy.asarray(bitarray).reshape((-1, binbits)))


class ImageMultiHash:
	"""
	This is an image hash containing a list of individual hashes for segments of the image.
	The matching logic is implemented as described in Efficient Cropping-Resistant Robust Image Hashing
	"""

	def __init__(self, hashes):
		# type: (list[ImageHash]) -> None
		self.segment_hashes = hashes  # type: list[ImageHash]

	def __eq__(self, other):
		# type: (object) -> bool
		if other is None:
			return False
		return self.matches(other)  # type: ignore

	def __ne__(self, other):
		# type: (object) -> bool
		return not self.matches(other)  # type: ignore

	def __sub__(self, other, hamming_cutoff=None, bit_error_rate=None):
		# type: (ImageMultiHash, float | None, float | None) -> float
		matches, sum_distance = self.hash_diff(other, hamming_cutoff, bit_error_rate)
		max_difference = len(self.segment_hashes)
		if matches == 0:
			return max_difference
		max_distance = matches * len(self.segment_hashes[0])
		tie_breaker = 0 - (float(sum_distance) / max_distance)
		match_score = matches + tie_breaker
		return max_difference - match_score

	def __hash__(self):
		return hash(tuple(hash(segment) for segment in self.segment_hashes))

	def __str__(self):
		return ','.join(str(x) for x in self.segment_hashes)

	def __repr__(self):
		return repr(self.segment_hashes)

	def hash_diff(self, other_hash, hamming_cutoff=None, bit_error_rate=None):
		# type: (ImageMultiHash, float | None, float | None) -> tuple[int, int]
		"""
		Gets the difference between two multi-hashes, as a tuple. The first element of the tuple is the number of
		matching segments, and the second element is the sum of the hamming distances of matching hashes.
		NOTE: Do not order directly by this tuple, as higher is better for matches, and worse for hamming cutoff.
		:param other_hash: The image multi hash to compare against
		:param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
		:param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff. The
		default of 0.25 means that the segment hashes can be up to 25% different
		"""
		# Set default hamming cutoff if it's not set.
		if hamming_cutoff is None:
			if bit_error_rate is None:
				bit_error_rate = 0.25
			hamming_cutoff = len(self.segment_hashes[0]) * bit_error_rate
		# Get the hash distance for each region hash within cutoff
		distances = []
		for segment_hash in self.segment_hashes:
			lowest_distance = min(
				segment_hash - other_segment_hash
				for other_segment_hash in other_hash.segment_hashes
			)
			if lowest_distance > hamming_cutoff:
				continue
			distances.append(lowest_distance)
		return len(distances), sum(distances)

	def matches(self, other_hash, region_cutoff=1, hamming_cutoff=None, bit_error_rate=None):
		# type: (ImageMultiHash, int, float | None, float | None) -> bool
		"""
		Checks whether this hash matches another crop resistant hash, `other_hash`.
		:param other_hash: The image multi hash to compare against
		:param region_cutoff: The minimum number of regions which must have a matching hash
		:param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
		:param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff. The
		default of 0.25 means that the segment hashes can be up to 25% different
		"""
		matches, _ = self.hash_diff(other_hash, hamming_cutoff, bit_error_rate)
		return matches >= region_cutoff

	def best_match(self, other_hashes, hamming_cutoff=None, bit_error_rate=None):
		# type: (list[ImageMultiHash], float | None, float | None) -> ImageMultiHash
		"""
		Returns the hash in a list which is the best match to the current hash
		:param other_hashes: A list of image multi hashes to compare against
		:param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
		:param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff.
		Defaults to 0.25 if unset, which means the hash can be 25% different
		"""
		return min(
			other_hashes,
			key=lambda other_hash: self.__sub__(other_hash, hamming_cutoff, bit_error_rate)
		)


def _find_region(remaining_pixels, segmented_pixels):
	"""
	Finds a region and returns a set of pixel coordinates for it.
	:param remaining_pixels: A numpy bool array, with True meaning the pixels are remaining to segment
	:param segmented_pixels: A set of pixel coordinates which have already been assigned to segment. This will be
	updated with the new pixels added to the returned segment.
	"""
	in_region = set()
	not_in_region = set()
	# Find the first pixel in remaining_pixels with a value of True
	available_pixels = numpy.transpose(numpy.nonzero(remaining_pixels))
	start = tuple(available_pixels[0])
	in_region.add(start)
	new_pixels = in_region.copy()
	while True:
		try_next = set()
		# Find surrounding pixels
		for pixel in new_pixels:
			x, y = pixel
			neighbours = [
				(x - 1, y),
				(x + 1, y),
				(x, y - 1),
				(x, y + 1)
			]
			try_next.update(neighbours)
		# Remove pixels we have already seen
		try_next.difference_update(segmented_pixels, not_in_region)
		# If there's no more pixels to try, the region is complete
		if not try_next:
			break
		# Empty new pixels set, so we know whose neighbour's to check next time
		new_pixels = set()
		# Check new pixels
		for pixel in try_next:
			if remaining_pixels[pixel]:
				in_region.add(pixel)
				new_pixels.add(pixel)
				segmented_pixels.add(pixel)
			else:
				not_in_region.add(pixel)
	return in_region


def _find_all_segments(pixels, segment_threshold, min_segment_size):
	"""
	Finds all the regions within an image pixel array, and returns a list of the regions.

	Note: Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
	rounding in the greyscale conversion.
	:param pixels: A numpy array of the pixel brightnesses.
	:param segment_threshold: The brightness threshold to use when differentiating between hills and valleys.
	:param min_segment_size: The minimum number of pixels for a segment.
	"""
	img_width, img_height = pixels.shape
	# threshold pixels
	threshold_pixels = pixels > segment_threshold
	unassigned_pixels = numpy.full(pixels.shape, True, dtype=bool)

	segments = []
	already_segmented = set()

	# Add all the pixels around the border outside the image:
	already_segmented.update([(-1, z) for z in range(img_height)])
	already_segmented.update([(z, -1) for z in range(img_width)])
	already_segmented.update([(img_width, z) for z in range(img_height)])
	already_segmented.update([(z, img_height) for z in range(img_width)])

	# Find all the "hill" regions
	while numpy.bitwise_and(threshold_pixels, unassigned_pixels).any():
		remaining_pixels = numpy.bitwise_and(threshold_pixels, unassigned_pixels)
		segment = _find_region(remaining_pixels, already_segmented)
		# Apply segment
		if len(segment) > min_segment_size:
			segments.append(segment)
		for pix in segment:
			unassigned_pixels[pix] = False

	# Invert the threshold matrix, and find "valleys"
	threshold_pixels_i = numpy.invert(threshold_pixels)
	while len(already_segmented) < img_width * img_height:
		remaining_pixels = numpy.bitwise_and(threshold_pixels_i, unassigned_pixels)
		segment = _find_region(remaining_pixels, already_segmented)
		# Apply segment
		if len(segment) > min_segment_size:
			segments.append(segment)
		for pix in segment:
			unassigned_pixels[pix] = False

	return segments


def crop_resistant_hash(
	image,  # type: Image.Image
	hash_func=None,  # type: HashFunc
	limit_segments=None,  # type: int | None
	segment_threshold=128,  # type: int
	min_segment_size=500,  # type: int
	segmentation_image_size=300  # type: int
):
	# type: (...) -> ImageMultiHash
	"""
	Creates a CropResistantHash object, by the algorithm described in the paper "Efficient Cropping-Resistant Robust
	Image Hashing". DOI 10.1109/ARES.2014.85
	This algorithm partitions the image into bright and dark segments, using a watershed-like algorithm, and then does
	an image hash on each segment. This makes the image much more resistant to cropping than other algorithms, with
	the paper claiming resistance to up to 50% cropping, while most other algorithms stop at about 5% cropping.

	Note: Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
	rounding in the greyscale conversion. This leads to a slightly different result.
	:param image: The image to hash
	:param hash_func: The hashing function to use
	:param limit_segments: If you have storage requirements, you can limit to hashing only the M largest segments
	:param segment_threshold: Brightness threshold between hills and valleys. This should be static, putting it between
	peak and through dynamically breaks the matching
	:param min_segment_size: Minimum number of pixels for a hashable segment
	:param segmentation_image_size: Size which the image is resized to before segmentation
	"""
	if hash_func is None:
		hash_func = dhash

	orig_image = image.copy()
	# Convert to gray scale and resize
	image = image.convert('L').resize((segmentation_image_size, segmentation_image_size), ANTIALIAS)
	# Add filters
	image = image.filter(ImageFilter.GaussianBlur()).filter(ImageFilter.MedianFilter())
	pixels = numpy.array(image).astype(numpy.float32)

	segments = _find_all_segments(pixels, segment_threshold, min_segment_size)

	# If there are no segments, have 1 segment including the whole image
	if not segments:
		full_image_segment = {(0, 0), (segmentation_image_size - 1, segmentation_image_size - 1)}
		segments.append(full_image_segment)

	# If segment limit is set, discard the smaller segments
	if limit_segments:
		segments = sorted(segments, key=lambda s: len(s), reverse=True)[:limit_segments]

	# Create bounding box for each segment
	hashes = []
	for segment in segments:
		orig_w, orig_h = orig_image.size
		scale_w = float(orig_w) / segmentation_image_size
		scale_h = float(orig_h) / segmentation_image_size
		min_y = min(coord[0] for coord in segment) * scale_h
		min_x = min(coord[1] for coord in segment) * scale_w
		max_y = (max(coord[0] for coord in segment) + 1) * scale_h
		max_x = (max(coord[1] for coord in segment) + 1) * scale_w
		# Compute robust hash for each bounding box
		bounding_box = orig_image.crop((min_x, min_y, max_x, max_y))
		hashes.append(hash_func(bounding_box))
		# Show bounding box
		# im_segment = image.copy()
		# for pix in segment:
		# 	im_segment.putpixel(pix[::-1], 255)
		# im_segment.show()
		# bounding_box.show()

	return ImageMultiHash(hashes)
