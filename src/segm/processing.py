import cv2
import numpy as np
from PIL import Image


def relative2absolute(coords, image_shape):
    left_top_x = int(coords[0] * image_shape[1])
    left_top_y = int(coords[1] * image_shape[0])
    right_bot_x = int(coords[2] * image_shape[1])
    right_bot_y = int(coords[3] * image_shape[0])
    return left_top_x, left_top_y, right_bot_x, right_bot_y


def mask2area_and_centroid(mask):
    moments = cv2.moments(mask)
    centroid_x = (moments["m10"] / moments["m00"]) / mask.shape[1]
    centroid_y = (moments["m01"] / moments["m00"]) / mask.shape[0]
    area = moments['m00'] / np.prod(mask.shape)
    return area, (centroid_x, centroid_y)


def get_region(area, centroid, size_coef=1.0,
               shift_x_coef=0.0, shift_y_coef=0.0):
    centroid_x, centroid_y = centroid
    side = np.sqrt(area) * size_coef / 2
    left_top_x = centroid_x - side + shift_x_coef
    left_top_y = centroid_y - side + shift_y_coef
    right_bot_x = centroid_x + side + shift_x_coef
    right_bot_y = centroid_y + side + shift_y_coef
    return left_top_x, left_top_y, right_bot_x, right_bot_y


def croc_region(np_image, region):
    pil_image = Image.fromarray(np_image)
    pil_image = pil_image.crop(box=region)
    np_image = np.array(pil_image)
    return np_image
