#!/usr/bin/env python
from PIL import Image
import imagehash
import os
import sys

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}

def find_similar_images(userpath, hashfunc = imagehash.average_hash):
    def is_image(filename):
        return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    
    image_filenames = [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
    images = {}
    for img in sorted(image_filenames):
        hash = hashfunc(Image.open(img))
        images[hash] = images.get(hash, []) + [img]
    
    for k, img_list in images.items():
        if len(img_list) > 1:
            print(" ".join(img_list))

def usage():
    sys.stderr.write(
"""SYNOPSIS: %s [ahash|phash|dhash|...] [<directory>]

Identifies similar images in the directory.

Method: 
  ahash:      Average hash
  phash:      Perceptual hash
  dhash:      Difference hash
  whash-haar: Haar wavelet hash
  whash-db4:  Daubechies wavelet hash

(C) Johannes Buchner, 2013
""" % sys.argv[0]
    )
    sys.exit(1)

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc == 2:
        userpath = "."
    elif argc == 3:
        userpath = sys.argv[2]
    else:
        usage()
    hashmethod = sys.argv[1]
    try:
        hashfunc = {
            'ahash': imagehash.average_hash,
            'phash': imagehash.phash,
            'dhash': imagehash.dhash,
            'whash-haar': imagehash.whash,
            'whash-db4': lambda img: imagehash.whash(img, mode='db4'),
        }[hashmethod]
    except KeyError:
        usage()
    find_similar_images(userpath=userpath, hashfunc=hashfunc)
