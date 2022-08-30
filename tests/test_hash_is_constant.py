import hashlib

import numpy
from PIL import ImageFilter
from PIL import Image
import PIL
from packaging import version

import imagehash
from tests import TestImageHash


def _calculate_segment_properties(segment):
	length = len(segment)
	min_y = min(coord[0] for coord in segment)
	min_x = min(coord[1] for coord in segment)
	max_y = max(coord[0] for coord in segment)
	max_x = max(coord[1] for coord in segment)
	return {
		"length": length,
		"min_x": min_x,
		"min_y": min_y,
		"max_x": max_x,
		"max_y": max_y
	}


def _pillow_has_convert_fix():
	"""
	Pillow version 7.0.0 introduced a fix for a rounding error in Image.convert("L") which means that segmentation is
	slightly different after this release.
	The PR which fixed the rounding and caused this inconsistency is https://github.com/python-pillow/Pillow/pull/4320
	"""
	return version.parse(PIL.__version__) >= version.parse("7.0.0")


class Test(TestImageHash):
	def setUp(self):
		self.image = self.get_data_image()
		self.peppers = self.get_data_image("peppers.png")

	def test_average_hash(self):
		result_hash = imagehash.average_hash(self.image)
		known_hash = "ffd7918181c9ffff"
		self.assertEqual(str(result_hash), known_hash)

	def test_phash(self):
		result_hash = imagehash.phash(self.image)
		known_hash = "ba8c84536bd3c366"
		self.assertEqual(str(result_hash), known_hash)

	def test_dhash(self):
		result_hash = imagehash.dhash(self.image)
		known_hash = "0026273b2b19550e"
		self.assertEqual(str(result_hash), known_hash)

	def test_whash(self):
		result_hash = imagehash.whash(self.image)
		known_hash = "ffd391818181a5e7"
		self.assertEqual(str(result_hash), known_hash)

	def test_color_hash(self):
		result_hash = imagehash.colorhash(self.image)
		known_hash = "07007000000"
		self.assertEqual(str(result_hash), known_hash)

	def test_crop_resistant_hash(self):
		result_hash = imagehash.crop_resistant_hash(self.peppers)
		if _pillow_has_convert_fix():
			known_hash = "c4d9f3e3e1c18101,706c6e66464c99b9,98d8f1ecd8f0f0e1,a082c0c49acc6dbd,f1f39b99c1c1b1b1,3a7ece1c9df4fcb9"
		else:
			known_hash = "c4d9f1e3e1c18101,706c6e66464c99b9,98d8f1ecd8f0f0e1,a282c0c49acc6dbd,b1f39b99e1c1b1b1,3a7ece1c9df4fcb9"
		self.assertEqual(str(result_hash), known_hash)

	def test_crop_resistant_segmentation(self):
		# Image pre-processing
		image = self.peppers.convert("L")
		if _pillow_has_convert_fix():
			known_bw_md5 = "61db06218cc8b9aba14812d965869120"
		else:
			known_bw_md5 = "61442e74c83cfea67d182481c24c5f3e"
		self.assertEqual(
			hashlib.md5(image.tobytes()).hexdigest(),
			known_bw_md5,
			"This hash should match, unless pillow have changed Convert('L') again"
		)
		image = image.resize((300, 300), Image.Resampling.LANCZOS)
		# Add filters
		image = image.filter(ImageFilter.GaussianBlur()).filter(ImageFilter.MedianFilter())
		pixels = numpy.array(image).astype(numpy.float32)
		# Segment
		segments = imagehash._find_all_segments(pixels, 128, 500)
		known_segment_count = 6
		self.assertEqual(len(segments), known_segment_count)
		if _pillow_has_convert_fix():
			known_segments = sorted([
				{'length': 595, 'min_x': 20, 'min_y': 0, 'max_x': 60, 'max_y': 31},
				{'length': 1458, 'min_x': 61, 'min_y': 0, 'max_x': 156, 'max_y': 58},
				{'length': 3505, 'min_x': 0, 'min_y': 111, 'max_x': 97, 'max_y': 191},
				{'length': 8789, 'min_x': 112, 'min_y': 145, 'max_x': 299, 'max_y': 260},
				{'length': 12153, 'min_x': 157, 'min_y': 0, 'max_x': 299, 'max_y': 148},
				{'length': 60916, 'min_x': 0, 'min_y': 0, 'max_x': 299, 'max_y': 299}
			], key=lambda x: (x["length"], x["min_x"], x["min_y"], x["max_x"], x["max_y"]))
		else:
			known_segments = sorted([
				{'length': 591, 'min_x': 20, 'min_y': 0, 'max_x': 60, 'max_y': 31},
				{'length': 1451, 'min_x': 61, 'min_y': 0, 'max_x': 156, 'max_y': 58},
				{'length': 12040, 'min_x': 157, 'min_y': 0, 'max_x': 299, 'max_y': 147},
				{'length': 3452, 'min_x': 0, 'min_y': 111, 'max_x': 97, 'max_y': 191},
				{'length': 8701, 'min_x': 112, 'min_y': 145, 'max_x': 299, 'max_y': 259},
				{'length': 61179, 'min_x': 0, 'min_y': 0, 'max_x': 299, 'max_y': 299}
			], key=lambda x: (x["length"], x["min_x"], x["min_y"], x["max_x"], x["max_y"]))
		segment_properties = sorted([
			_calculate_segment_properties(segment) for segment in segments
		], key=lambda x: (x["length"], x["min_x"], x["min_y"], x["max_x"], x["max_y"]))
		self.assertEqual(segment_properties, known_segments)
