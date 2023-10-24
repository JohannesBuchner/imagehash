import unittest

import imagehash
from .utils import TestImageHash


class Test (TestImageHash):
    def setUp(self):
        self.image = self.get_data_image()

    def test_hex_to_multi_hash(self):
        generated_hash = imagehash.crop_resistant_hash(
            self.image,
            # these arguments are required for a true multi image hash with multiple segments
            min_segment_size=500, segmentation_image_size=1000
        )
        string = str(generated_hash)
        emsg = ('Stringified multihash did not match original hash')
        self.assertEqual(
            generated_hash,
            imagehash.hex_to_multihash(string),
            emsg
        )
        string = '0026273b2b19550e,6286373334662535,6636192c47639573,999d6d67a3e82125,27a327c38191a4ad,938971382b328a46'
        emsg = ('Stringified multihash did not match hardcoded original hash')
        self.assertEqual(
            generated_hash,
            imagehash.hex_to_multihash(string),
            emsg
        )


if __name__ == '__main__':
    unittest.main()
