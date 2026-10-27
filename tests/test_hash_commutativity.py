import unittest

import numpy as np

import imagehash

from .utils import TestImageHash


def _segment(seed):
	return imagehash.ImageHash(np.random.RandomState(seed).randint(0, 2, (8, 8)).astype(bool))


class TestHashCommutativity(TestImageHash):
	"""`a - b` must equal `b - a` for every hash type (#152)."""

	def setUp(self):
		self.image = self.get_data_image()
		self.peppers = self.get_data_image('peppers.png')

	def test_imagehash_sub_is_commutative(self):
		hash_funcs = [
			imagehash.average_hash, imagehash.phash, imagehash.phash_simple,
			imagehash.dhash, imagehash.dhash_vertical, imagehash.whash, imagehash.colorhash,
		]
		for hash_func in hash_funcs:
			with self.subTest(hash_func=hash_func.__name__):
				a = hash_func(self.image)
				b = hash_func(self.peppers)
				self.assertEqual(a - b, b - a)

	def test_multihash_sub_is_commutative(self):
		width, height = self.peppers.size
		hashes = {
			'imagehash': imagehash.crop_resistant_hash(self.image),
			'peppers': imagehash.crop_resistant_hash(self.peppers),
			'rotated': imagehash.crop_resistant_hash(self.peppers.rotate(5)),
			'cropped': imagehash.crop_resistant_hash(self.peppers.crop((0, 0.3 * height, 0.4 * width, 0.75 * height))),
		}
		names = list(hashes)
		for i, a in enumerate(names):
			for b in names[i + 1:]:
				with self.subTest(a=a, b=b):
					self.assertEqual(hashes[a] - hashes[b], hashes[b] - hashes[a])

	def test_multihash_sub_different_segment_counts(self):
		# `small` has one segment, which is also in `large`; `large` has a second
		# segment that matches nothing in `small`.
		small = imagehash.ImageMultiHash([_segment(1)])
		large = imagehash.ImageMultiHash([_segment(1), imagehash.ImageHash(~_segment(1).hash)])
		self.assertEqual(small - large, large - small)
		# zero only when both hashes match all of their segments with zero distance
		self.assertGreater(small - large, 0)
		self.assertEqual(small - small, 0)
		self.assertEqual(large - large, 0)

	def test_multihash_sub_asymmetric_matches(self):
		# Same segment count, but two segments of `a` match the same segment of `b`,
		# so a -> b finds two matches and b -> a only one.
		base = _segment(2)
		near = base.hash.copy()
		near[0, 0] = not near[0, 0]
		a = imagehash.ImageMultiHash([base, imagehash.ImageHash(near)])
		b = imagehash.ImageMultiHash([base, imagehash.ImageHash(~base.hash)])
		self.assertNotEqual(a.hash_diff(b), b.hash_diff(a))
		self.assertEqual(a - b, b - a)

	def test_multihash_sub_cutoff_arguments(self):
		small = imagehash.ImageMultiHash([_segment(3)])
		large = imagehash.ImageMultiHash([_segment(3), _segment(4)])
		for kwargs in ({'hamming_cutoff': 64}, {'bit_error_rate': 1.0}):
			with self.subTest(**kwargs):
				self.assertEqual(small.__sub__(large, **kwargs), large.__sub__(small, **kwargs))

	def test_best_match_is_unchanged(self):
		# best_match ranks candidates by how many of the query's segments they match,
		# so a candidate containing every query segment wins even if it has extra
		# segments of its own.
		query = imagehash.ImageMultiHash([_segment(5), _segment(6)])
		superset = imagehash.ImageMultiHash([_segment(5), _segment(6), _segment(7), _segment(8), _segment(9)])
		partial = imagehash.ImageMultiHash([_segment(5), imagehash.ImageHash(~_segment(6).hash)])
		self.assertIs(query.best_match([partial, superset]), superset)
		self.assertIs(query.best_match([superset, partial]), superset)


if __name__ == '__main__':
	unittest.main()
