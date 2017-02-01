#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
from PIL import Image
import six

import imagehash

"""
Demo of hashing
"""
def find_similar_images(userpath, hashfunc = imagehash.average_hash):
    import os
    def is_image(filename):
    	f = filename.lower()
    	return f.endswith(".png") or f.endswith(".jpg") or \
    		f.endswith(".jpeg") or f.endswith(".bmp") or f.endswith(".gif")
    
    image_filenames = [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
    images = {}
    for img in sorted(image_filenames):
    	hash = hashfunc(Image.open(img))
    	images[hash] = images.get(hash, []) + [img]
    
    for k, img_list in six.iteritems(images):
    	if len(img_list) > 1:
    		print(" ".join(img_list))


if __name__ == '__main__':
    import sys, os
    def usage():
    	sys.stderr.write("""SYNOPSIS: %s [ahash|phash|dhash|...] [<directory>]

Identifies similar images in the directory.

Method: 
  ahash:      Average hash
  phash:      Perceptual hash
  dhash:      Difference hash
  whash-haar: Haar wavelet hash
  whash-db4:  Daubechies wavelet hash

(C) Johannes Buchner, 2013
""" % sys.argv[0])
    	sys.exit(1)
    
    hashmethod = sys.argv[1] if len(sys.argv) > 1 else usage()
    if hashmethod == 'ahash':
    	hashfunc = imagehash.average_hash
    elif hashmethod == 'phash':
    	hashfunc = imagehash.phash
    elif hashmethod == 'dhash':
    	hashfunc = imagehash.dhash
    elif hashmethod == 'whash-haar':
    	hashfunc = imagehash.whash
    elif hashmethod == 'whash-db4':
    	hashfunc = lambda img: imagehash.whash(img, mode='db4')
    else:
    	usage()
    userpath = sys.argv[2] if len(sys.argv) > 2 else "."
    find_similar_images(userpath=userpath, hashfunc=hashfunc)
    

