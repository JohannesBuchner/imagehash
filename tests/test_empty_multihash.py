import unittest

import imagehash


class TestEmptyMultiHash(unittest.TestCase):
    def test_hash_diff_empty_segments(self):
        empty = imagehash.ImageMultiHash([])
        other = imagehash.hex_to_multihash('aa' * 8)
        self.assertEqual(empty.hash_diff(other), (0, 0))
        self.assertEqual(other.hash_diff(empty), (0, 0))
        self.assertEqual(empty.hash_diff(empty), (0, 0))
        self.assertEqual(empty - other, 0.0)
        self.assertFalse(empty.matches(other))
