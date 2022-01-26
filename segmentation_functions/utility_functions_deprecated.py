import numpy as np
import cv2
import tensorflow as tf

SIZE = (288, 384)
SHAPE_X = [SIZE[0], SIZE[1], 3]
SHAPE_Y_WNET = [SIZE[0], SIZE[1], 2]
SHAPE_Y_UNET = [SIZE[0], SIZE[1], 1]




def _read_image_wnet(img_path):
    img_path = img_path.decode()
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    return image


def _read_image_normal_wnet(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def _read_mask_wnet(img_path):
    img_path = img_path.decode()
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = mask/255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def _read_mask_normal_wnet(img_path):
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def _read_image_unet(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, SIZE)
    image = image/255.0
    return image


def _read_mask_unet(img_path):
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, SIZE)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def _read_image_unet_train(img_path):
    img_path = img_path.decode()
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, SIZE)
    image = image/255.0
    return image


def _read_mask_unet_train(img_path):
    img_path = img_path.decode()
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, SIZE)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask



def read_image_mask(img_path, typos):
    """Read image and mask according to specific types
    Valid typos:
        unet-image
        unet-mask
        wnet-image
        wnet-mask
        wnet-image-normal
        wnet-mask-normal
    Args:
        img_path (str): image path
        typos (str): combination of model-image types

    Raises:
        ValueError: wrong typos error

    Returns:
        image: image or mask
    """

    if typos == 'wnet-image':
        return _read_image_wnet(img_path)

    if typos == 'wnet-image-normal':
        return _read_image_normal_wnet(img_path)

    if typos == 'wnet-mask':
        return _read_mask_wnet(img_path)

    if typos == 'wnet-mask-normal':
        return _read_mask_normal_wnet(img_path)

    if typos == 'unet-image':
        return _read_image_unet(img_path)

    if typos == 'unet-mask':
        return _read_mask_unet(img_path)

    if typos == 'unet-image-train':
        return _read_image_unet_train(img_path)

    if typos == 'unet-mask-train':
        return _read_mask_unet_train(img_path)

    raise ValueError(f"typos:{typos} is not correct")



