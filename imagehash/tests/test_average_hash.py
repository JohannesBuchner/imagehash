from __future__ import (absolute_import, division, print_function)

from PIL import Image
import unittest

import imagehash
import imagehash.tests as tests


class Test(tests.TestImageHash):
    def setUp(self):
        self.image = self.get_data_image()
        self.func = imagehash.average_hash

    def test_average_hash(self):
        self.check_hash_algorithm(self.func, self.image)

    def test_average_hash_length(self):
        self.check_hash_length(self.func, self.image, [8, 20])

    def test_average_hash_stored(self):
        self.check_hash_stored(self.func, self.image)


if __name__ == '__main__':
    unittest.main()
