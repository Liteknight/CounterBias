import numpy as np


class ToFloatMNIST(object):
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        return (image / 255).astype('f4')


class ToFloatUKBB(object):
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        try:
            image = image.astype('f8')
            maxv = np.max(image)
            #minv = np.min(image)
            return (image / maxv).astype('f4')
        except:
            return image

class ToFloatMitacs(object):
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        try:
            image = image.astype('f4')
            #minv = np.min(image)
            return image
        except:
            return image

class Crop3D(object):
    """Crop to size informed."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        original_size = image.shape
        if (len(original_size) != 3):
            raise Exception(f'Crop3D only works with 3 dimensions. Input has {len(original_size)} dimensions')
        x_init = round((original_size[0]-self.size[0])/2)
        x_end = original_size[0] - x_init
        y_init = round((original_size[1]-self.size[1])/2)
        y_end = original_size[1] - y_init
        z_init = round((original_size[2]-self.size[2])/2)
        z_end = original_size[2] - z_init
        image = image[x_init:x_end,y_init:y_end,z_init:z_end]
        if (image.shape != self.size):
            raise Exception(f'Transformations got {image.shape}, but it should be P{self.size}')
        return image

