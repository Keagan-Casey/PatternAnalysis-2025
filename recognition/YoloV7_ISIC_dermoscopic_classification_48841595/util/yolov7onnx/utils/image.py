from typing import Tuple

import time
import cv2
import numpy as np


def letterbox(image: np.ndarray, output_size: Tuple[int, int], fill_value: int = 0) -> np.ndarray:
    """Resizes the input image while maintaining its aspect ratio and pad with fill_value.

    Args:
        image: Input image as a numpy array.
        output_size: Desired output height and width as a tuple.
        fill_value: Padding grayscale value [0, 255]. Default is 0.

    Returns:
        Numpy array representing the letterboxed image.
    """
    input_height, input_width = image.shape[:2]
    output_height, output_width = output_size

    # Calculate the scaling factor to maintain aspect ratio
    scale = min(output_height / input_height, output_width / input_width)
    new_height, new_width = int(input_height * scale), int(input_width * scale)

    # Choose interpolation method based on upscaling or downscaling
    #inter = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Calculate padding values to centralize the resized image
    top_pad = (output_height - new_height) // 2
    bottom_pad = output_height - new_height - top_pad
    left_pad = (output_width - new_width) // 2
    right_pad = output_width - new_width - left_pad

    # Use numpy's pad function for efficient padding
    padded_image = np.pad(
        resized_image,
        ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )

    return padded_image
