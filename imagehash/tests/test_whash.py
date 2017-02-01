from __future__ import (absolute_import, division, print_function)

from PIL import Image
import six
import unittest

import imagehash


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

    def test_hash_size_not_2power(self):
        emsg = 'hash_size is not power of 2'
        for hash_size in [3, 7, 12]:
            with six.assertRaisesRegex(self, AssertionError, emsg):
                imagehash.whash(self.image, hash_size=hash_size)

    def test_hash_size_is_less_than_image_size(self):
        image = self._get_white_image((120, 200))
        emsg = 'hash_size in a wrong range'
        for hash_size in [128, 512]:
            with six.assertRaisesRegex(self, AssertionError, emsg):
                imagehash.whash(image, hash_size=hash_size)

    def test_custom_hash_size_and_scale(self):
        hash_size = 16
        hash = imagehash.whash(self.image, hash_size=hash_size, image_scale=64)
        self.assertEqual(hash.hash.size, hash_size**2)

    def test_hash_size_more_than_scale(self):
        emsg = 'hash_size in a wrong range'
        with six.assertRaisesRegex(self, AssertionError, emsg):
            imagehash.whash(self.image, hash_size=32, image_scale=16)

    def test_image_scale_not_2power(self):
        emsg = 'image_scale is not power of 2'
        for image_scale in [4, 8, 16]:
            with six.assertRaisesRegex(self, AssertionError, emsg):
                imagehash.whash(self.image, image_scale=image_scale+1)

    def test_whash_small_image(self):
        image_small = self._get_white_image((4, 9))
        try:
            imagehash.whash(image_small)
        except Exception as e:
            self.fail(e)


if __name__ == '__main__':
    unittest.main()
