import numpy

import imagehash
from PIL import Image, ImageFilter

IMAGE_FILE = "../tests/data/peppers.png"
IMG_SIZE = 300
SEGMENT_THRESHOLD = 128
MIN_SEGMENT_SIZE = 500
RAINBOW = [
	(141, 211, 199),
	(255, 255, 179),
	(190, 186, 218),
	(251, 128, 114),
	(128, 177, 211),
	(253, 180, 98),
	(179, 222, 105),
	(252, 205, 229),
	(217, 217, 217),
	(188, 128, 189)
]

# Load image
full_image = Image.open(IMAGE_FILE)
width, height = full_image.size
# Image pre-processing
image = full_image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
# Add filters
image = image.filter(ImageFilter.GaussianBlur()).filter(ImageFilter.MedianFilter())
pixels = numpy.array(image).astype(numpy.float32)
# Split segments
segments = imagehash._find_all_segments(pixels, SEGMENT_THRESHOLD, MIN_SEGMENT_SIZE)
# Change back to RGB
image = image.convert("RGB")
# Colour in segments
for num, segment in enumerate(segments):
	for x, y in segment:
		image.putpixel((y, x), RAINBOW[num % len(RAINBOW)])
image.show()
