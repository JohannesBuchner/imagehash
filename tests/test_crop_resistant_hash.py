from __future__ import absolute_import, division, print_function

import unittest
from datetime import datetime
from unittest.mock import patch

import imagehash

from .utils import TestImageHash


class Test(TestImageHash):
	def setUp(self):
		self.image = self.get_data_image()
		self.peppers = self.get_data_image('peppers.png')

	def test_segmented_hash(self):
		original_hash = imagehash.crop_resistant_hash(self.image)
		rotate_image = self.image.rotate(-1)
		small_rotate_hash = imagehash.crop_resistant_hash(rotate_image)
		emsg = ('slightly rotated image should have similar hash {} {}'.format(original_hash, small_rotate_hash))
		self.assertTrue(original_hash.matches(small_rotate_hash), emsg)
		rotate_image = self.image.rotate(-90)
		large_rotate_hash = imagehash.crop_resistant_hash(rotate_image)
		emsg = ('rotated image should have different hash {} {}'.format(original_hash, large_rotate_hash))
		self.assertFalse(original_hash.matches(large_rotate_hash), emsg)

		other_hashes = [small_rotate_hash, large_rotate_hash]
		self.assertEqual(
			original_hash.best_match(other_hashes),
			small_rotate_hash,
			'Hash of the slightly rotated image should be a better match than for the more heavily rotated image.'
		)

	def test_segmented_hash__hash_func(self):
		segmented_ahash = imagehash.crop_resistant_hash(self.image, imagehash.average_hash)
		segmented_dhash = imagehash.crop_resistant_hash(self.image, imagehash.dhash)
		self.assertFalse(
			segmented_ahash.matches(segmented_dhash),
			'Segmented hash should not match when the underlying hashing method is not the same'
		)

	def test_segmented_hash__empty_crops(self):
		image = self.peppers.resize((150, 150))
		# At scale 0.5, both 37.5 and 38.5 round to 38 in Pillow (#205).
		empty_segments = [
			{(75, 0), (76, 299)},
			{(0, 75), (299, 76)},
			{(75, 75), (76, 76)},
		]
		full_segment = {(0, 0), (299, 299)}
		for hash_func in (imagehash.whash, imagehash.dhash):
			with self.subTest(hash_func=hash_func.__name__):
				with patch.object(imagehash, '_find_all_segments', return_value=empty_segments + [full_segment]):
					result = imagehash.crop_resistant_hash(image, hash_func)
				self.assertEqual(len(result.segment_hashes), 1)
				self.assertEqual(str(result), str(hash_func(image)))

	def test_segmented_hash__all_empty_crops(self):
		image = self.peppers.resize((150, 150))
		empty_segments = [{(75, 0), (76, 299)}, {(0, 75), (299, 76)}]
		for limit_segments in (None, 1):
			with self.subTest(limit_segments=limit_segments):
				with patch.object(imagehash, '_find_all_segments', return_value=empty_segments):
					result = imagehash.crop_resistant_hash(image, imagehash.whash, limit_segments=limit_segments)
				self.assertEqual(len(result.segment_hashes), 1)
				self.assertEqual(str(result), str(imagehash.whash(image)))
				self.assertTrue(result.matches(result))
				self.assertEqual(result - result, 0)
				self.assertEqual(str(imagehash.hex_to_multihash(str(result))), str(result))

	def test_segmented_hash__small_image(self):
		# Exercise resizing, filtering, segmentation, and wavelet hashing together.
		image = self.peppers.resize((4, 4))
		result = imagehash.crop_resistant_hash(
			image, imagehash.whash, min_segment_size=1, segmentation_image_size=30
		)
		self.assertGreater(len(result.segment_hashes), 0)
		self.assertTrue(result.matches(result))
		self.assertEqual(str(imagehash.hex_to_multihash(str(result))), str(result))

	def test_segmented_hash__limit_segments(self):
		segmented_orig = imagehash.crop_resistant_hash(self.image)
		segmented_limit = imagehash.crop_resistant_hash(self.image, limit_segments=1)
		self.assertGreaterEqual(
			len(segmented_orig.segment_hashes), len(segmented_limit.segment_hashes),
			'Limit segments should mean there are fewer segments'
		)
		self.assertEqual(
			len(segmented_limit.segment_hashes), 1,
			'Limit segments should correctly limit the segment count'
		)

	def test_segmented_hash__segment_threshold(self):
		segmented_low_threshold = imagehash.crop_resistant_hash(self.image, segment_threshold=20)
		segmented_high_threshold = imagehash.crop_resistant_hash(self.image, segment_threshold=250)
		self.assertFalse(
			segmented_low_threshold.matches(segmented_high_threshold, region_cutoff=3),
			'Segmented hash should not match when segment threshold is changed'
		)

	def test_segmentation_image_size(self):
		start_time = datetime.now()
		imagehash.crop_resistant_hash(self.image, segmentation_image_size=200)
		small_timed = datetime.now() - start_time

		start_time = datetime.now()
		imagehash.crop_resistant_hash(self.image, segmentation_image_size=400)
		large_timed = datetime.now() - start_time

		self.assertGreater(large_timed, small_timed, 'Hashing should take longer when the segmentation image is larger')

	def test_min_segment_size(self):
		small_segments_hash = imagehash.crop_resistant_hash(self.peppers, min_segment_size=100)
		big_segments_hash = imagehash.crop_resistant_hash(self.peppers, min_segment_size=1000)

		self.assertGreater(
			len(small_segments_hash.segment_hashes),
			len(big_segments_hash.segment_hashes),
			'Small segment size limit should lead to larger number of segments detected.'
		)
		self.assertEqual(
			small_segments_hash,
			big_segments_hash,
			'Hashes should still match, as large segments are present in both'
		)

	def test_crop_resistance(self):
		full_image = self.peppers
		width, height = full_image.size
		crop_10 = full_image.crop((0.05 * width, 0.05 * height, 0.95 * width, 0.95 * height))
		crop_40 = full_image.crop((0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height))
		crop_asymmetric = full_image.crop((0, 0.3 * height, 0.4 * width, 0.75 * height))

		full_hash = imagehash.crop_resistant_hash(full_image, min_segment_size=200)
		crop_hash_10 = imagehash.crop_resistant_hash(crop_10)
		crop_hash_40 = imagehash.crop_resistant_hash(crop_40)
		crop_hash_asymmetric = imagehash.crop_resistant_hash(crop_asymmetric)

		self.assertEqual(crop_hash_10, full_hash, 'Slightly cropped image hash should match full image hash')
		self.assertEqual(crop_hash_40, full_hash, 'Heavily cropped image hash should match full image hash')
		self.assertEqual(
			crop_hash_asymmetric, full_hash, 'Asymmetrically cropped image hash should match full image hash'
		)


if __name__ == '__main__':
	unittest.main()
