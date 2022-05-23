from _typeshed import Incomplete
from typing import Callable, Optional, Tuple, List

import numpy as np
import numpy.typing as npt
from PIL.Image import Image


class ImageHash:
    hash: npt.NDArray[np.bool]
    def __init__(self, binary_array: npt.NDArray[np.bool]) -> None: ...
    def __sub__(self, other: ImageHash) -> int: ...
    def __eq__(self, other: ImageHash) -> bool: ...
    def __ne__(self, other: ImageHash) -> bool: ...
    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...

def hex_to_hash(hexstr: str) -> ImageHash: ...
def hex_to_flathash(hexstr: str, hashsize: int) -> ImageHash: ...
def old_hex_to_hash(hexstr: str, hash_size: int = ...) -> ImageHash: ...
def average_hash(image: Image, hash_size: int = ..., mean: Callable[[npt.NDArray], float] = ...) -> ImageHash: ...
def phash(image: Image, hash_size: int = ..., highfreq_factor: int = ...) -> ImageHash: ...
def phash_simple(image: Image, hash_size: int = ..., highfreq_factor: int = ...) -> ImageHash: ...
def dhash(image: Image, hash_size: int = ...) -> ImageHash: ...
def dhash_vertical(image: Image, hash_size: int = ...) -> ImageHash: ...
def whash(image: Image, hash_size: int = ..., image_scale: Optional[int] = ..., mode: str = ..., remove_max_haar_ll: bool = ...) -> ImageHash: ...
def colorhash(image: Image, binbits: int = ...) -> ImageHash: ...

class ImageMultiHash:
    segment_hashes: Incomplete
    def __init__(self, hashes: ImageHash) -> None: ...
    def __eq__(self, other: ImageMultiHash) -> bool: ...
    def __ne__(self, other: ImageMultiHash) -> bool: ...
    def __sub__(self, other: ImageMultiHash, hamming_cutoff: Optional[int] = ..., bit_error_rate: Optional[float] = ...) -> int: ...
    def __hash__(self) -> int: ...
    def hash_diff(self, other_hash: ImageMultiHash, hamming_cutoff: Optional[int] = ..., bit_error_rate: Optional[float] = ...) -> Tuple[int, int]: ...
    def matches(self, other_hash: ImageMultiHash, region_cutoff: int = ..., hamming_cutoff: Optional[int] = ..., bit_error_rate: Optional[float] = ...) -> bool: ...
    def best_match(self, other_hashes: List[ImageMultiHash], hamming_cutoff: Optional[int] = ..., bit_error_rate: Optional[float] = ...) -> ImageMultiHash: ...

def crop_resistant_hash(image: Image, hash_func: Callable[[Image], ImageHash] = ..., limit_segments: Optional[int] = ..., segment_threshold: int = ..., min_segment_size: int = ..., segmentation_image_size: int = ...) -> ImageMultiHash: ...
