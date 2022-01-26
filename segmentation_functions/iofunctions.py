import os
import cv2
from glob import glob
from sklearn.model_selection import train_test_split


def image_mask_from_paths(image_path, mask_path):
    """ Read the image and mask from the given path. """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    return image, mask


def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def image_resize(image, size, keep_aspect_ratio=True):
    """Resize image array

    Args:
        image: array
        size: tuple(height, width)
        keep_aspect_ratio (bool, optional): Defaults to True

    Returns:
        image array
    """
    width, height = size
    h, w = image.shape[:2]

    if h == height and w == width:
        return image

    if keep_aspect_ratio:
        pad_bottom, pad_right = 0, 0
        ratio = w / h

        if h > height or w > width:
            # shrinking image algorithm
            interp = cv2.INTER_AREA
        else:
            # stretching image algorithm
            interp = cv2.INTER_CUBIC

        w = width
        h = round(w / ratio)
        if h > height:
            h = height
            w = round(h * ratio)
        pad_bottom = abs(height - h)
        pad_right = abs(width - w)

        scaled_img = cv2.resize(image, (w, h), interpolation=interp)
        padded_img = cv2.copyMakeBorder(
            scaled_img,
            0,
            pad_bottom,
            0, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        return padded_img
    return cv2.resize(image, size)


def image_mask_resize(image, mask, size, keep_aspect_ratio=True):
    """
    Resize image and corresponding mask concurrently
    image: image array
    mask: mask array
    size: (height, width)

    returns touple(image array, mask array)
    """
    img = image_resize(image, size, keep_aspect_ratio)
    msk = image_resize(mask, size, keep_aspect_ratio)
    return img, msk


def load_images(path, subdir):
    """ Load all the data and then split them into train and valid dataset. """
    img_paths = glob(os.path.join(path, f"{subdir}/*"))
    img_paths.sort()

    return img_paths


def split_dataset(images_paths, masks_paths, validation_percent=0.1, test_percent=0.1):
    """returns (train_x, train_y), (valid_x, valid_y), (test_x, test_y)"""
    len_ids = len(images_paths)
    len_masks = len(masks_paths)
    assert len_ids == len_masks

    valid_size = int(validation_percent * len_ids)
    test_size = int(test_percent * len_ids)

    train_x, test_x = train_test_split(
        images_paths, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(
        masks_paths, test_size=test_size, random_state=42)

    train_x, valid_x = train_test_split(
        train_x, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(
        train_y, test_size=valid_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def image_pure_name(image_path, image_type='.jpg'):
    return os.path.basename(image_path).replace(image_type, '')


def create_save_path(root, subdir, name):
    return os.path.join(root, subdir, name)


def joinp(dest_path, subdir):
    return os.path.join(dest_path, subdir)
