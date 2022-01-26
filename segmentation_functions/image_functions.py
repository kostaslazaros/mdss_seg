# import os
import cv2
import albumentations as alb
from tqdm import tqdm
from segmentation_functions import iofunctions as iof


def image_resize(image, size, keep_aspect_ratio=True):
    img = image
    width, height = size
    h, w = img.shape[:2]

    if h == height and w == width:
        return img

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

        scaled_img = cv2.resize(img, (w, h), interpolation=interp)
        padded_img = cv2.copyMakeBorder(
            scaled_img,
            0,
            pad_bottom,
            0,pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0,0,0]
        )
        return padded_img
    return cv2.resize(image, size)


def image_mask_resize(image, mask, size, keep_aspect_ratio=True):
    img = image_resize(image, size, keep_aspect_ratio)
    msk = image_resize(mask, size, keep_aspect_ratio)
    return img, msk


def augment_data_to_disk(images, masks, save_path, size, augment=True):
    """Creating augmentation images and saving to disk

    Args:
        images (image paths)
        masks (image paths)
        save_path (root save path)
        size (tuple(height, width))
        augment (bool, optional): Defaults to True
    """
    tile_size = size[0] // 12
    crop_size = (size[1]-tile_size, size[0]-tile_size)
    ## Crop Parameters
    x_min = y_min = 0
    x_max, y_max = x_min + size[0], y_min + size[1]

    augmentations = (
        alb.CenterCrop(p=1, height=crop_size[0], width=crop_size[1]),
        alb.Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
        alb.RandomRotate90(p=1),
        alb.Transpose(p=1),
        alb.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        alb.GridDistortion(p=1),
        alb.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
        alb.VerticalFlip(p=1),
        alb.HorizontalFlip(p=1),
        alb.RandomBrightnessContrast(p=1),
        alb.RandomGamma(p=1),
        alb.HueSaturationValue(p=1),
        alb.RGBShift(p=1),
        alb.RandomBrightness(p=1),
        alb.RandomContrast(p=1),
        alb.MotionBlur(p=1, blur_limit=7),
        alb.MedianBlur(p=1, blur_limit=9),
        alb.GaussianBlur(p=1, blur_limit=9),
        alb.GaussNoise(p=1),
        alb.ChannelShuffle(p=1),
        alb.CoarseDropout(p=1, max_holes=8, max_height=tile_size, max_width=tile_size)
    )

    augmentations_gray = (
        alb.CenterCrop(p=1, height=crop_size[0], width=crop_size[1]),
        alb.VerticalFlip(p=1),
        alb.HorizontalFlip(p=1)
    )

    for image, mask in tqdm(zip(images, masks), total=len(images)):
        image_name = iof.image_pure_name(image)
        mask_name = iof.image_pure_name(mask)

        x, y = iof.image_mask_from_paths(image, mask)
        try:
            h, w, c = x.shape
        except Exception as e:
            image = image[:-1]
            x, y = iof.image_mask_from_paths(image, mask)
            h, w, c = x.shape

        if augment == True:
            ## Grayscale
            img_gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            msk_gray = y  # This is already gray

            xydic = []

            xydic.append({'image': x, 'mask': y})
            xydic.append({'image': img_gray, 'mask': msk_gray})
            for daug in augmentations:
                xydic.append(daug(image=x, mask=y))
            for daug in augmentations_gray:
                xydic.append(daug(image=img_gray, mask=msk_gray))

            images = []
            masks = []
            for aug in xydic:
                images.append(aug['image'])
                masks.append(aug['mask'])

        else:
            images = [x]
            masks  = [y]

        for idx, image_mask_tuple in enumerate(zip(images, masks)):
            i, m = image_mask_tuple
            i, m = image_mask_resize(i, m, size, True)
            tmp_image_name = f"{image_name}_{idx}.jpg"
            tmp_mask_name  = f"{mask_name}_{idx}.jpg"

            image_path =  iof.create_save_path(save_path, "image", tmp_image_name)
            mask_path  = iof.create_save_path(save_path, "mask", tmp_mask_name)
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)


def transform_dataset(dest_path, origin_path="CVC-ClinicDB", size=(384, 288)):
    images_paths = iof.load_images(origin_path, 'images')
    masks_paths = iof.load_images(origin_path, 'masks')

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = iof.split_dataset(images_paths, masks_paths)

    iof.create_dir(f"{dest_path}/train/image/")
    iof.create_dir(f"{dest_path}/train/mask/")
    iof.create_dir(f"{dest_path}/valid/image/")
    iof.create_dir(f"{dest_path}/valid/mask/")
    iof.create_dir(f"{dest_path}/test/image/")
    iof.create_dir(f"{dest_path}/test/mask/")

    augment_data_to_disk(train_x, train_y, iof.joinp(dest_path, "train"), size, augment=True)
    augment_data_to_disk(valid_x, valid_y, iof.joinp(dest_path, "valid"), size, augment=False)
    augment_data_to_disk(test_x, test_y, iof.joinp(dest_path, "test"), size, augment=False)


def transform_dataset2folders(dest_path, origin_path="CVC-ClinicDB", size=(384, 288)):
    """
    Image and mask augmentation function.
    It also puts augmented data into corresponding folders.
    ATTENTION: not correct for training; creates data leakage.
    Use above function instead.

    Args:
        dest_path (str): Destination path
        origin_path (str): Original dataset path. Defaults to "CVC-ClinicDB".
        size (tuple, optional): Image size. Defaults to (384, 288).
    """
    images_paths = iof.load_images(origin_path, 'images')
    masks_paths = iof.load_images(origin_path, 'masks')
    iof.create_dir(f"{dest_path}/image/")
    iof.create_dir(f"{dest_path}/mask/")
    augment_data_to_disk(images_paths, masks_paths, dest_path, size, augment=True)
