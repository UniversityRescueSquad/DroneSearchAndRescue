import numpy as np

class SyntheticApertureImage:
    def __init__(self, num_images, edge_length):
        """
        Initializes the SyntheticApertureImage class with a given number of images and edge length.

        Args:
            num_images (int): The number of images to combine.
            edge_length (int): The size of the border or edge around the central part of each image 
            that will be used to create the synthetic aperture image. This border is added to allow 
            for overlap between adjacent images when they are combined into the final image. The 
            edge_length argument is used to calculate the size of the overlap between adjacent images, 
            and it should be chosen based on the expected size of the objects or features in the scene, 
            as well as the distance between the camera and the scene. A larger edge length will result 
            in more overlap between adjacent images and a smoother final image, but it will also 
            increase the processing time and memory requirements.
        """
        self.num_images = num_images
        self.edge_length = edge_length
        self.images = []

    def add_image(self, image):
        """
        Adds a new image to the list of images for the SyntheticApertureImage.

        Args:
            image (numpy.ndarray): The image to add to the list of images.
        """
        self.images.append(image)
        if len(self.images) > self.num_images:
            self.images.pop(0)

    def create_sai(self):
        """
        Creates a Synthetic Aperture Image (SAI) from the list of images.

        Returns:
            Synthetic Aperture Image (numpy.ndarray): The SAI created from the list of images.
        """
        if not self.images:
            return None

        sai = np.zeros_like(self.images[0])  # Initialize the SAI as a zero-filled array.
        weight = np.zeros_like(self.images[0])  # Initialize the weight as a zero-filled array.

        for i, image in enumerate(self.images):
            # Compute the edge length in x and y directions.
            dx = dy = self.edge_length // 2
            if i == 0:
                dx = dy = 0
            elif i == self.num_images - 1:
                dx = dy = self.edge_length - 1

            # Add to the weight and SAI.
            weight[dy:-dy or None, dx:-dx or None] += 1
            sai[dy:-dy or None, dx:-dx or None] += image[:sai.shape[0], :sai.shape[1]]

        # Normalize the SAI by the weight and handle NaN values.
        sai = sai.astype(np.float64)
        sai /= weight
        sai[np.isnan(sai)] = 0
        sai = np.nan_to_num(sai).astype(np.uint8)

        return sai