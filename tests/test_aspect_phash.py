from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

import imagehash

from .utils import TestImageHash


class Test(TestImageHash):
	def test_aspect_phash(self):
		img = self.get_data_image()
		actual = imagehash.aspect_phash(img)
		etalon = imagehash.ImageHash(
			np.frombuffer(bytes.fromhex('7d404140414040404040404040404040'), dtype=np.uint8)
		)  # actual hash computed by `aspect` tool
		self.assertLess(etalon - actual, 3)


if __name__ == "__main__":
	unittest.main()
