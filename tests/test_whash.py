from __future__ import (absolute_import, division, print_function)

import sys

from PIL import Image
import unittest

import imagehash
from .utils import TestImageHash


# extracted from 'six'
# https://github.com/benjaminp/six/blob/3974f0c4f6700a5821b451abddff8b3ba6b2a04f/six.py#L642-L689
if sys.version_info[0] == 3:
    if sys.version_info[1] <= 1:
        _assertRaisesRegex = 'assertRaisesRegexp'
    else:
        _assertRaisesRegex = 'assertRaisesRegex'
else:
    _assertRaisesRegex = 'assertRaisesRegexp'
def assertRaisesRegex(self, *args, **kwargs):
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)


class TestBasic(TestImageHash):

    def setUp(self):
        self.image = self.get_data_image()
        self.func = imagehash.whash

    def test_whash(self):
        self.check_hash_algorithm(self.func, self.image)

    def test_whash_length(self):
        self.check_hash_length(self.func, self.image, sizes=[2,4,8,16,32,64])

    def test_whash_stored(self):
        self.check_hash_stored(self.func, self.image, sizes=[2,4,8,16,32,64])


class Test(unittest.TestCase):
    def setUp(self):
        self.image = self._get_white_image()

    def _get_white_image(self, size=None):
        if size is None:
            size = (512, 512)
        return Image.new("RGB", size, "white")

    def test_hash_size_2power(self):
        for hash_size in [4, 8, 16]:
            hash = imagehash.whash(self.image, hash_size=hash_size)
            self.assertEqual(hash.hash.size, hash_size**2)

    def test_hash_size_for_small_images(self):
        default_hash_size = 8
        for image_size in [(1, 25), (7, 5)]:
            image = self._get_white_image(image_size)
            hash = imagehash.whash(image)
            self.assertEqual(hash.hash.size, default_hash_size**2)

    def test_hash_size_not_2power(self):
        emsg = 'hash_size is not power of 2'
        for hash_size in [3, 7, 12]:
            with assertRaisesRegex(self, AssertionError, emsg):
                imagehash.whash(self.image, hash_size=hash_size)

    def test_hash_size_is_less_than_image_scale(self):
        image = self._get_white_image((120, 200))
        emsg = 'hash_size in a wrong range'
        for hash_size in [128, 512]:
            with assertRaisesRegex(self, AssertionError, emsg):
                imagehash.whash(image, hash_size=hash_size, image_scale=64)

    def test_custom_hash_size_and_scale(self):
        hash_size = 16
        hash = imagehash.whash(self.image, hash_size=hash_size, image_scale=64)
        self.assertEqual(hash.hash.size, hash_size**2)

    def test_hash_size_more_than_scale(self):
        emsg = 'hash_size in a wrong range'
        with assertRaisesRegex(self, AssertionError, emsg):
            imagehash.whash(self.image, hash_size=32, image_scale=16)

    def test_image_scale_not_2power(self):
        emsg = 'image_scale is not power of 2'
        for image_scale in [4, 8, 16]:
            with assertRaisesRegex(self, AssertionError, emsg):
                imagehash.whash(self.image, image_scale=image_scale+1)


if __name__ == '__main__':
    unittest.main()
