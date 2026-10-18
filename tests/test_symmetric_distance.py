import unittest

import numpy as np

import imagehash

from .utils import TestImageHash

SINGLE_HASH_FUNCS = [
	imagehash.average_hash, imagehash.phash, imagehash.phash_simple,
	imagehash.dhash, imagehash.dhash_vertical, imagehash.whash, imagehash.colorhash,
]


def _segment(seed):
	return imagehash.ImageHash(np.random.RandomState(seed).randint(0, 2, (8, 8)).astype(bool))


def _flip_first_bit(segment):
	flipped = segment.hash.copy()
	flipped[0, 0] = not flipped[0, 0]
	return imagehash.ImageHash(flipped)


class TestSymmetricDistance(TestImageHash):
	"""`a.symmetric_distance(b)` must equal `b.symmetric_distance(a)` for every hash type (#152)."""

	def setUp(self):
		self.image = self.get_data_image()
		self.peppers = self.get_data_image('peppers.png')

	def test_imagehash_symmetric_distance_is_commutative(self):
		for hash_func in SINGLE_HASH_FUNCS:
			with self.subTest(hash_func=hash_func.__name__):
				a = hash_func(self.image)
				b = hash_func(self.peppers)
				self.assertEqual(a.symmetric_distance(b), b.symmetric_distance(a))

	def test_imagehash_symmetric_distance_equals_sub(self):
		for hash_func in SINGLE_HASH_FUNCS:
			with self.subTest(hash_func=hash_func.__name__):
				a = hash_func(self.image)
				b = hash_func(self.peppers)
				self.assertEqual(a.symmetric_distance(b), a - b)
				self.assertEqual(a.symmetric_distance(b), b - a)
				self.assertEqual(a.symmetric_distance(a), 0)

	def test_multihash_symmetric_distance_is_commutative(self):
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
					distance = hashes[a].symmetric_distance(hashes[b])
					self.assertEqual(distance, hashes[b].symmetric_distance(hashes[a]))
					self.assertEqual(distance, max(hashes[a] - hashes[b], hashes[b] - hashes[a]))

	def test_multihash_symmetric_distance_different_segment_counts(self):
		# `small` has one segment, which is also in `large`; `large` has a second
		# segment that matches nothing in `small`.
		small = imagehash.ImageMultiHash([_segment(1)])
		large = imagehash.ImageMultiHash([_segment(1), imagehash.ImageHash(~_segment(1).hash)])
		self.assertEqual(small.symmetric_distance(large), large.symmetric_distance(small))
		# zero only when both hashes match all of their segments with zero distance
		self.assertEqual(small.symmetric_distance(large), 1.0)
		self.assertEqual(small.symmetric_distance(small), 0)
		self.assertEqual(large.symmetric_distance(large), 0)

	def test_multihash_symmetric_distance_asymmetric_matches(self):
		# Same segment count, but two segments of `a` match the same segment of `b`,
		# so a -> b finds two matches and b -> a only one.
		base = _segment(2)
		a = imagehash.ImageMultiHash([base, _flip_first_bit(base)])
		b = imagehash.ImageMultiHash([base, imagehash.ImageHash(~base.hash)])
		self.assertNotEqual(a.hash_diff(b), b.hash_diff(a))
		self.assertEqual(a.symmetric_distance(b), b.symmetric_distance(a))
		self.assertEqual(a.symmetric_distance(b), 1.0)

	def test_multihash_symmetric_distance_cutoff_arguments(self):
		small = imagehash.ImageMultiHash([_segment(3)])
		large = imagehash.ImageMultiHash([_segment(3), _segment(4)])
		for kwargs in ({'hamming_cutoff': 64}, {'bit_error_rate': 1.0}, {'hamming_cutoff': 0}):
			with self.subTest(**kwargs):
				distance = small.symmetric_distance(large, **kwargs)
				self.assertEqual(distance, large.symmetric_distance(small, **kwargs))
				self.assertEqual(distance, max(small.__sub__(large, **kwargs), large.__sub__(small, **kwargs)))

	def test_multihash_sub_is_unchanged(self):
		# `a - b` stays one-sided, normalised by the segments of `a`; these are the values
		# it has always returned, so stored thresholds keep working.
		small = imagehash.ImageMultiHash([_segment(1)])
		large = imagehash.ImageMultiHash([_segment(1), imagehash.ImageHash(~_segment(1).hash)])
		self.assertEqual(small - large, 0.0)
		self.assertEqual(large - small, 1.0)

		base = _segment(2)
		a = imagehash.ImageMultiHash([base, _flip_first_bit(base)])
		b = imagehash.ImageMultiHash([base, imagehash.ImageHash(~base.hash)])
		self.assertEqual(a - b, 0.0078125)
		self.assertEqual(b - a, 1.0)

		small = imagehash.ImageMultiHash([_segment(3)])
		large = imagehash.ImageMultiHash([_segment(3), _segment(4)])
		self.assertEqual(small.__sub__(large, hamming_cutoff=64), 0.0)
		self.assertEqual(large.__sub__(small, hamming_cutoff=64), 0.2109375)
		self.assertEqual(large.__sub__(small, bit_error_rate=1.0), 0.2109375)
		self.assertEqual(large.__sub__(small, hamming_cutoff=0), 1.0)

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
