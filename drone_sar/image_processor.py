from typing import List
import numpy as np
from PIL import Image
import math


class ImageProcessor:
    def __init__(self):
        pass

    def split_image(
        self, image: Image.Image, width: int, height: int, overlap_percent: int
    ) -> List[Image.Image]:
        """
        Splits a high-resolution image into smaller sub-images with the given width, height, and overlap percentage.

        Args:
            image: The PIL Image to be split.
            width: The desired width of each sub-image.
            height: The desired height of each sub-image.
            overlap_percent: The percentage of overlap between adjacent sub-images.

        Returns:
            A list of PIL Image objects, each representing a sub-image of the original image.
        """
        # Compute the number of columns and rows
        num_cols = int(
            math.ceil((image.width - width) / (width - overlap_percent / 100 * width))
        )
        num_rows = int(
            math.ceil(
                (image.height - height) / (height - overlap_percent / 100 * height)
            )
        )

        # Split the image
        sub_images = []
        for row in range(num_rows):
            for col in range(num_cols):
                # Compute the bounding box for the sub-image
                left = int((width - overlap_percent * width) * col)
                upper = int((height - overlap_percent * height) * row)
                right = min(left + width, image.width)
                lower = min(upper + height, image.height)
                bbox = (left, upper, right, lower)

                # Crop the sub-image
                sub_image = image.crop(bbox)
                sub_images.append(sub_image)

        return sub_images

    def histogram_equalization(self, image: Image.Image) -> Image.Image:
        """
        Applies histogram equalization to the input image.

        Args:
            image: The PIL Image to be equalized.

        Returns:
            A PIL Image object with the equalized histogram.
        """
        # Convert the image to grayscale
        gray_image = image.convert("L")

        # Compute the histogram
        hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])

        # Compute the cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        # Apply the equalization
        equalized_image = np.interp(gray_image, bins[:-1], cdf_normalized)

        # Convert the equalized image back to PIL format
        equalized_image = Image.fromarray(equalized_image.astype("uint8"))

        return equalized_image

    def contrast_stretching(
        self, image: Image.Image, low: int, high: int
    ) -> Image.Image:
        """
        Applies contrast stretching to the given PIL Image.

        Args:
            image: The PIL Image to be stretched.
            low: The low end of the stretch range.
            high: The high end of the stretch range.

        Returns:
            A new PIL Image that has been contrast stretched.
        """
        # Convert image to grayscale
        gray_image = image.convert("L")

        # Compute the minimum and maximum pixel values in the image
        min_val, max_val = gray_image.getextrema()

        # Compute the scale factor for contrast stretching
        scale_factor = (high - low) / (max_val - min_val)

        # Apply contrast stretching
        stretched_image = gray_image.point(
            lambda x: int(low + (x - min_val) * scale_factor)
        )

        return stretched_image
