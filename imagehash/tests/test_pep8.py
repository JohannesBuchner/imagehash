"""
Perform a PEP8 conformance test of the imagehash code base.
"""
from __future__ import (absolute_import, division, print_function)

import os
import unittest

import pep8

import imagehash


class TestCodeFormat(unittest.TestCase):
    def test_pep8_conformance(self):
        # Tests the imagehash code base against the "pep8" tool.
        #
        # Users can add their own excluded files (should files exist in the
        # local directory which is not in the repository) by adding a
        # ".pep8_test_exclude.txt" file in the same directory as this test.
        # The file should be a line separated list of filenames/directories
        # as can be passed to the "pep8" tool's exclude list.

        pep8style = pep8.StyleGuide(quiet=False)

        # Allow users to add their own exclude list.
        extra_exclude_fname = os.path.join(os.path.dirname(__file__),
                                           '.pep8_test_exclude.txt')
        if os.path.isfile(extra_exclude_fname):
            with open(extra_exclude_file, 'r') as fi:
                extra_exclude = [line.strip()
                                 for line in fi if line.strip()]
            pep8style.options.exclude.extend(extra_exclude)

        root = os.path.abspath(imagehash.__file__)
        result = pep8style.check_files([os.path.dirname(root)])
        emsg = 'Found {} pep8 code syntax errors (and warnings).'
        self.assertEqual(result.total_errors, 0,
                         emsg.format(result.total_errors))


if __name__ == '__main__':
    unittest.main()
