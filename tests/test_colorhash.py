from __future__ import (absolute_import, division, print_function)

import unittest

import imagehash
from .utils import TestImageHash


class Test(TestImageHash):
    def setUp(self):
        self.image = self.get_data_image()
        self.func = imagehash.colorhash

    def tearDown(self):
        self.image.close()

    def test_colorhash(self):
        self.check_hash_algorithm(self.func, self.image)

    def check_hash_algorithm(self, func, image):
        original_hash = func(image)
        rotate_image = image.rotate(-1)
        rotate_hash = func(rotate_image)
        distance = original_hash - rotate_hash
        emsg = ('slightly rotated image should have '
                'similar hash {} {} {}'.format(original_hash, rotate_hash,
                                               distance))
        self.assertTrue(distance <= 10, emsg)
        self.assertEqual(original_hash, rotate_hash, emsg)
        rotate_image = image.rotate(180)
        rotate_hash = func(rotate_image)
        emsg = ('flipped image should have same '
                'hash {} {}'.format(original_hash, rotate_hash))
        self.assertEqual(original_hash, rotate_hash, emsg)

    def test_colorhash_stored(self):
        self.check_hash_stored(self.func, self.image)

    def test_colorhash_length(self):
        self.check_hash_length(self.func, self.image)

    def test_colorhash_size(self):
        self.check_hash_size(self.func, self.image)

    def check_hash_stored(self, func, image, binbits=range(2,5)):
        for bit in binbits:
            image_hash = func(image, bit)
            other_hash = imagehash.hex_to_flathash(str(image_hash), bit * (2 + 6*2))
            emsg = 'stringified hash {} != original hash {}'.format(other_hash,
                                                                    image_hash)
            self.assertEqual(image_hash, other_hash, emsg)
            distance = image_hash - other_hash
            emsg = ('unexpected hamming distance {}: original hash {} '
                    '- stringified hash {}'.format(distance, image_hash,
                                                   other_hash))
            self.assertEqual(distance, 0, emsg)

    def check_hash_length(self, func, image, binbits=range(2,5)):
        for bit in binbits:
            image_hash = func(image, bit)
            emsg = 'bit={} is not respected'.format(bit)
            self.assertEqual(image_hash.hash.size, (2 + 6 * 2)*bit, emsg)

    def check_hash_size(self, func, image, binbits=range(-1,1)):
        for bit in binbits:
            with self.assertRaises(ValueError):
                func(image, bit)

    def test_colorhash_icc_support(self):
        image = self.get_data_image('bigbuckbunny.png')
        image_with_icc = self.get_data_image('bigbuckbunny-with-diff-icc.png')
        with image, image_with_icc:
            self.check_icc_support(self.func, image, image_with_icc)

    def check_icc_support(self, func, orig_image, image_with_diff_icc):
        orig_data_hash = func(orig_image)
        diff_icc_hash = func(image_with_diff_icc)
        emsg = ('same image data but with a different ICC profile should '
                'result in different hashes: {} == {}'.format(orig_data_hash,
                                                              diff_icc_hash))
        self.assertNotEqual(orig_data_hash, diff_icc_hash, emsg)

    def test_colorhash_icc_ignored(self):
        image = self.get_data_image('bigbuckbunny.png')
        image_with_icc = self.get_data_image('bigbuckbunny-with-diff-icc.png')
        with image, image_with_icc:
            self.check_icc_ignored(self.func, image, image_with_icc)

    def check_icc_ignored(self, func, orig_image, image_with_diff_icc):
        orig_data_hash = func(orig_image)
        diff_icc_hash = func(image_with_diff_icc, ignore_icc=True)
        emsg = ('ignoring the ICC profile in an image with the same image '
                'data as one without a profile should result in the same hash: '
                '{} != {}'.format(orig_data_hash, diff_icc_hash))
        self.assertEqual(orig_data_hash, diff_icc_hash, emsg)

if __name__ == '__main__':
    unittest.main()
