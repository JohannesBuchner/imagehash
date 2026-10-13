import unittest

import imagehash

from .utils import TestImageHash


class Test(TestImageHash):
	def setUp(self):
		self.image = self.get_data_image()
		self.func = imagehash.dhash

	def test_dhash(self):
		self.check_hash_algorithm(self.func, self.image)

	def test_dhash_length(self):
		self.check_hash_length(self.func, self.image)

	def test_dhash_stored(self):
		self.check_hash_stored(self.func, self.image)

	def test_dhash_size(self):
		self.check_hash_size(self.func, self.image)


if __name__ == '__main__':
	unittest.main()
