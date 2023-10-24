import unittest

import imagehash

from .utils import TestImageHash

CHECK_HASH_DEFAULT = range(2, 5)
CHECK_HASH_SIZE_DEFAULT = range(-1, 1)


class Test(TestImageHash):
	def setUp(self):
		self.image = self.get_data_image()
		self.func = imagehash.colorhash

	def test_colorhash(self):
		self.check_hash_algorithm(self.func, self.image)

	def check_hash_algorithm(self, func, image):
		original_hash = func(image)
		rotate_image = image.rotate(-1)
		rotate_hash = func(rotate_image)
		distance = original_hash - rotate_hash
		emsg = ('slightly rotated image should have similar hash {} {} {}'.format(original_hash, rotate_hash, distance))
		self.assertTrue(distance <= 10, emsg)
		self.assertEqual(original_hash, rotate_hash, emsg)
		rotate_image = image.rotate(180)
		rotate_hash = func(rotate_image)
		emsg = ('flipped image should have same hash {} {}'.format(original_hash, rotate_hash))
		self.assertEqual(original_hash, rotate_hash, emsg)

	def test_colorhash_stored(self):
		self.check_hash_stored(self.func, self.image)

	def test_colorhash_length(self):
		self.check_hash_length(self.func, self.image)

	def test_colorhash_size(self):
		self.check_hash_size(self.func, self.image)

	def check_hash_stored(self, func, image, binbits=CHECK_HASH_DEFAULT):
		for bit in binbits:
			image_hash = func(image, bit)
			other_hash = imagehash.hex_to_flathash(str(image_hash), bit * (2 + 6 * 2))
			emsg = 'stringified hash {} != original hash {}'.format(other_hash, image_hash)
			self.assertEqual(image_hash, other_hash, emsg)
			distance = image_hash - other_hash
			emsg = ('unexpected hamming distance {}: original hash {} - stringified hash {}'.format(distance, image_hash, other_hash))
			self.assertEqual(distance, 0, emsg)

	def check_hash_length(self, func, image, binbits=CHECK_HASH_DEFAULT):
		for bit in binbits:
			image_hash = func(image, bit)
			emsg = 'bit={} is not respected'.format(bit)
			self.assertEqual(image_hash.hash.size, (2 + 6 * 2) * bit, emsg)

	def check_hash_size(self, func, image, binbits=CHECK_HASH_SIZE_DEFAULT):
		for bit in binbits:
			with self.assertRaises(ValueError):
				func(image, bit)


if __name__ == '__main__':
	unittest.main()
