import unittest

import numpy as np

import imagehash


class TestCompareToNone(unittest.TestCase):
	"""`==` and `!=` must disagree, including when the other side is None (#213)."""

	def setUp(self):
		self.hash = imagehash.ImageHash(np.zeros((8, 8), dtype=bool))
		self.other = imagehash.ImageHash(np.ones((8, 8), dtype=bool))
		self.multi = imagehash.ImageMultiHash([self.hash])

	def test_imagehash_is_not_equal_to_none(self):
		self.assertFalse(self.hash == None)  # noqa: E711
		self.assertTrue(self.hash != None)  # noqa: E711

	def test_multihash_is_not_equal_to_none(self):
		self.assertFalse(self.multi == None)  # noqa: E711
		self.assertTrue(self.multi != None)  # noqa: E711

	def test_imagehash_is_not_equal_to_other_types(self):
		for other in [str(self.hash), 0, self.multi]:
			self.assertFalse(self.hash == other)
			self.assertTrue(self.hash != other)
		self.assertNotIn(self.hash, [str(self.hash), self.other])
		self.assertIn(self.hash, [str(self.hash), self.hash])

	def test_multihash_is_not_equal_to_other_types(self):
		for other in [str(self.multi), 0, self.hash]:
			self.assertFalse(self.multi == other)
			self.assertTrue(self.multi != other)

	def test_imagehash_comparisons_are_unchanged(self):
		same = imagehash.ImageHash(np.zeros((8, 8), dtype=bool))
		self.assertTrue(self.hash == same)
		self.assertFalse(self.hash != same)
		self.assertFalse(self.hash == self.other)
		self.assertTrue(self.hash != self.other)

	def test_multihash_comparisons_are_unchanged(self):
		same = imagehash.ImageMultiHash([imagehash.ImageHash(np.zeros((8, 8), dtype=bool))])
		different = imagehash.ImageMultiHash([self.other])
		self.assertTrue(self.multi == same)
		self.assertFalse(self.multi != same)
		self.assertFalse(self.multi == different)
		self.assertTrue(self.multi != different)


if __name__ == '__main__':
	unittest.main()
