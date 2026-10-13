import unittest

import imagehash

from .utils import TestImageHash


class Test(TestImageHash):
	def setUp(self):
		self.image = self.get_data_image()
		self.func = imagehash.phash

	def test_phash(self):
		self.check_hash_algorithm(self.func, self.image)

	def test_phash_length(self):
		self.check_hash_length(self.func, self.image)

	def test_phash_stored(self):
		self.check_hash_stored(self.func, self.image)

	def test_phash_size(self):
		self.check_hash_size(self.func, self.image)


if __name__ == '__main__':
	unittest.main()
