from __future__ import (absolute_import, division, print_function)

import os
import os.path
import unittest

from PIL import Image

import imagehash


class TestImageHash(unittest.TestCase):
    @staticmethod
    def get_data_image(fname=None):
        if fname is None:
            fname = 'imagehash.png'
        dname = os.path.abspath(os.path.dirname(__file__))
        target = os.path.join(dname, 'data', fname)
        if not os.path.isfile(target):
            emsg = 'Unknown test image file: {!r}'
            raise ValueError(emsg.format(target))
        return Image.open(target)

    def check_hash_algorithm(self, func, image):
        original_hash = func(image)
        rotate_image = image.rotate(-1)
        rotate_hash = func(rotate_image)
        distance = original_hash - rotate_hash
        emsg = ('slightly rotated image should have '
                'similar hash {} {} {}'.format(original_hash, rotate_hash,
                                               distance))
        self.assertTrue(distance <= 10, emsg)
        rotate_image = image.rotate(-90)
        rotate_hash = func(rotate_image)
        emsg = ('rotated image should have different '
                'hash {} {}'.format(original_hash, rotate_hash))
        self.assertNotEqual(original_hash, rotate_hash, emsg)
        distance = original_hash - rotate_hash
        emsg = ('rotated image should have larger different '
                'hash {} {} {}'.format(original_hash, rotate_hash,
                                       distance))
        self.assertTrue(distance > 10, emsg)

    def check_hash_length(self, func, image, sizes):
        for hash_size in sizes:
            image_hash = func(image, hash_size=hash_size)
            emsg = 'hash_size={} is not respected'.format(hash_size)
            self.assertEqual(image_hash.hash.size, hash_size**2, emsg)

    def check_hash_stored(self, func, image):
        image_hash = func(image)
        other_hash = imagehash.hex_to_hash(str(image_hash))
        emsg = 'stringified hash {} != original hash {}'.format(other_hash,
                                                                image_hash)
        self.assertEqual(image_hash, other_hash, emsg)
        distance = image_hash - other_hash
        emsg = ('unexpected hamming distance {}: original hash {} '
                '- stringified hash {}'.format(distance, image_hash,
                                               other_hash))
        self.assertEqual(distance, 0, emsg)
