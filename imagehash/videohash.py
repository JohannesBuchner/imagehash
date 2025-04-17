import cv2
import io
import imagehash
from PIL import Image
import numpy


class VideoHash(imagehash.ImageHash):
    def __init__(self, binary_array):
        imagehash.ImageHash.__init__(self, binary_array)


def _save_image_to_buffer(image):
    # Encode the image to JPEG format
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise Exception("Failed to encode image")

    # Write the encoded image to a memory buffer
    buffer = io.BytesIO()
    buffer.write(encoded_image)
    return buffer


def _extract_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if ret:
        return _save_image_to_buffer(frame)


def _get_frames(video, frames_quantity):
    # type: (cv2.VideoCapture, int) -> Image.Image

    # Get total frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    if total_frames < frames_quantity:
        frames_quantity = total_frames
    frames_step = total_frames // (frames_quantity - 1)

    # Iterate over frames
    for i in range(0, total_frames, frames_step):
        yield Image.open(_extract_frame(video, i))

    # Release the VideoCapture object
    video.release()


def average_hash(video, hash_size=8, mean=numpy.mean, frames_quantity=8):
    # type: (cv2.VideoCapture, int, MeanFunc, int) -> VideoHash
    """
    Average Hash computation

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    Step by step explanation: https://web.archive.org/web/20171112054354/https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/ # noqa: E501


    @video must be a cv2.VideoCapture instance.
    @mean how to determine the average luminescence. can try numpy.median instead.
    @frames_quantity split the video into a certain number of frames
    """
    hashed_frames = []

    # Hash all frames
    for frame in _get_frames(video, frames_quantity):
        image_hash = imagehash.average_hash(frame, hash_size)
        hashed_frames.append(image_hash.hash)

    return VideoHash(numpy.array(hashed_frames))


def phash(video, hash_size=8, highfreq_factor=4, frames_quantity=8):
    # type: (cv2.VideoCapture, int, int, int) -> VideoHash
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @video must be a cv2.VideoCapture instance.
    @frames_quantity split the video into a certain number of frames
    """
    hashed_frames = []

    # Hash all frames
    for frame in _get_frames(video, frames_quantity):
        image_hash = imagehash.phash(frame, hash_size, highfreq_factor)
        hashed_frames.append(image_hash.hash)

    return VideoHash(numpy.array(hashed_frames))


def phash_simple(video, hash_size=8, highfreq_factor=4, frames_quantity=8):
    # type: (cv2.VideoCapture, int, int, int) -> VideoHash
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @video must be a cv2.VideoCapture instance.
    @frames_quantity split the video into a certain number of frames
    """
    hashed_frames = []

    # Hash all frames
    for frame in _get_frames(video, frames_quantity):
        image_hash = imagehash.phash_simple(frame, hash_size, highfreq_factor)
        hashed_frames.append(image_hash.hash)

    return VideoHash(numpy.array(hashed_frames))


def dhash(video, hash_size=8, frames_quantity=8):
    # type: (cv2.VideoCapture, int, int) -> VideoHash
    """
    Difference Hash computation.

    following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

    computes differences horizontally

    @video must be a cv2.VideoCapture instance.
    @frames_quantity split the video into a certain number of frames
    """
    hashed_frames = []

    # Hash all frames
    for frame in _get_frames(video, frames_quantity):
        image_hash = imagehash.dhash(frame, hash_size)
        hashed_frames.append(image_hash.hash)

    return VideoHash(numpy.array(hashed_frames))


def dhash_vertical(video, hash_size=8, frames_quantity=8):
    # type: (cv2.VideoCapture, int, int) -> VideoHash
    """
    Difference Hash computation.

    following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

    computes differences vertically

    @video must be a cv2.VideoCapture instance.
    @frames_quantity split the video into a certain number of frames
    """
    hashed_frames = []

    # Hash all frames
    for frame in _get_frames(video, frames_quantity):
        image_hash = imagehash.dhash_vertical(frame, hash_size)
        hashed_frames.append(image_hash.hash)

    return VideoHash(numpy.array(hashed_frames))


def whash(video, hash_size=8, image_scale=None, mode="haar", remove_max_haar_ll=True, frames_quantity=8):
    # type: (cv2.VideoCapture, int, int | None, WhashMode, bool, int) -> VideoHash
    """
    Wavelet Hash computation.

    based on https://www.kaggle.com/c/avito-duplicate-ads-detection/

    @video must be a cv2.VideoCapture instance.
    @hash_size must be a power of 2 and less than @image_scale.
    @image_scale must be power of 2 and less than image size. By default is equal to max
                    power of 2 for an input image.
    @mode (see modes in pywt library):
                    'haar' - Haar wavelets, by default
                    'db4' - Daubechies wavelets
    @remove_max_haar_ll - remove the lowest low level (LL) frequency using Haar wavelet.
    @frames_quantity split the video into a certain number of frames
    """
    hashed_frames = []

    # Hash all frames
    for frame in _get_frames(video, frames_quantity):
        image_hash = imagehash.whash(frame, hash_size, image_scale, mode, remove_max_haar_ll)
        hashed_frames.append(image_hash.hash)

    return VideoHash(numpy.array(hashed_frames))


def colorhash(video, binbits=3, frames_quantity=8):
    # type: (cv2.VideoCapture, int, int) -> VideoHash
    """
    Color Hash computation.

    Computes fractions of image in intensity, hue and saturation bins:

    * the first binbits encode the black fraction of the image
    * the next binbits encode the gray fraction of the remaining image (low saturation)
    * the next 6*binbits encode the fraction in 6 bins of saturation, for highly saturated parts of the remaining image
    * the next 6*binbits encode the fraction in 6 bins of saturation, for mildly saturated parts of the remaining image

    @video must be a cv2.VideoCapture instance.
    @binbits number of bits to use to encode each pixel fractions
    @frames_quantity split the video into a certain number of frames
    """
    hashed_frames = []

    # Hash all frames
    for frame in _get_frames(video, frames_quantity):
        image_hash = imagehash.colorhash(frame, binbits)
        hashed_frames.append(image_hash.hash)

    return VideoHash(numpy.array(hashed_frames))
