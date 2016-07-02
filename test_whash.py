from PIL import Image
import imagehash

import unittest

class WHashTestCase(unittest.TestCase):

    def get_white_image(self, size):
        return Image.new("RGB", size, "white")

    def test_hash_size_2power(self):
        image = self.get_white_image( (512, 512) )
        for hash_size in [4, 8, 16]:
            hash = imagehash.whash(image, hash_size=hash_size)
            self.assertEqual(hash.hash.size, hash_size**2)

    def test_hash_size_not_2power(self):
        image = self.get_white_image( (512, 512) )
        for hash_size in [3, 7, 12]:
            self.assertRaises(AssertionError, imagehash.whash, image, hash_size=hash_size)

    def test_hash_size_is_less_than_image_size(self):
        image = self.get_white_image( (120, 200) )
        for hash_size in [128, 512]:
            self.assertRaises(AssertionError, imagehash.whash, image, hash_size=hash_size)

    def test_custom_hash_size_and_scale(self):
        image = self.get_white_image( (512, 512) )
        hash_size = 16
        hash = imagehash.whash(image, hash_size=hash_size, image_scale=64)
        self.assertEqual(hash.hash.size, hash_size**2)

    def test_hash_size_more_than_scale(self):
        image = self.get_white_image( (512, 512) )
        self.assertRaises(AssertionError, imagehash.whash, image, hash_size=32, image_scale=16)

if __name__ == '__main__':
    unittest.main()
