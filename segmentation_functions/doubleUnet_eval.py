import os
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from segmentation_functions.metrics import *
from segmentation_functions.iofunctions import create_dir
from segmentation_functions.parameters import SIZE


def read_image_normal(image):
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask_normal(mask):
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def read_image(image):
    image = image.decode()
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    return image


def read_mask(mask):
    mask = mask.decode()
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask = mask/255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def parse_prediction(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred


def load_wnet_model_weights(path):
    with CustomObjectScope({
        'dice_coef': dice_coef,
        'iou': iou,
        'dice_loss': dice_loss,
        'bce_dice_loss': bce_dice_loss,
        'focal_loss': focal_loss,
    }):
        model = load_model(path)
    return model


def parse_data_wnet(image, mask):
    def _parse(image, mask):
        image = read_image(image)
        mask = read_mask(mask)
        mask = np.concatenate([mask, mask], axis=-1)
        return image, mask

    image, mask = tf.numpy_function(
        _parse, [image, mask], [tf.float32, tf.float32])
    image.set_shape([SIZE[1], SIZE[0], 3])
    mask.set_shape([SIZE[1], SIZE[0], 2])
    return image, mask


def tf_dataset(image, mask, batch_size=8, parse_data_function=parse_data_wnet):
    dataset = tf.data.Dataset.from_tensor_slices((image, mask))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data_function)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def evaluate_doubleu(model, x_data, y_data, result_dir):
    create_dir(result_dir)
    for i, (img_path, msk_path) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        filenumber, *_ = os.path.basename(img_path).split('.')
        img = read_image_normal(img_path)
        msk = read_mask_normal(msk_path)
        # _, h, w, _ = img.shape

        y_pred1 = parse_prediction(model.predict(img)[0][..., -2])
        y_pred2 = parse_prediction(model.predict(img)[0][..., -1])

        white_line = vertical_whiteline(img, line_width=10)

        input_image = img[0] * 255.0
        ground_truth = mask_to_3d(msk) * 255.0
        predict1 = mask_to_3d(y_pred1) * 255.0
        predict2 = mask_to_3d(y_pred2) * 255.0
        added_image = cv2.addWeighted(input_image, 0.6, predict2, 0.1, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        all_images = [
            input_image,
            added_image,
            ground_truth,
            predict1,
            predict2
        ]

        titles = ('Image', 'Overlayed', 'Ground Truth',
                  'Prediction1', 'Prediction2')
        image = create_result_image(all_images, white_line, titles)
        save2disk(image, img_path, result_dir)


def vertical_whiteline(image, line_width=10):
    _, h, w, _ = image.shape
    white_line = np.ones((h, line_width, 3)) * 255.0
    return white_line


def save2disk(image, img_path, result_dir):
    filenumber, *_ = os.path.basename(img_path).split('.')
    filename = os.path.join(result_dir, f"{filenumber}.png")
    cv2.imwrite(filename, image)


def create_result_image(images, vertical_line, titles):
    assert len(images) == len(titles)
    final = []
    for image in images:
        final.append(image)
        final.append(vertical_line)
    image = np.concatenate(final[:-1], axis=1)
    create_titles(image, titles)
    return image


def create_titles(image, titles):
    image_width = SIZE[0]
    line_width = 10
    y_pad = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, title in enumerate(titles):
        y_pos = (image_width + line_width) * i + y_pad
        cv2.putText(image, title, (y_pos, 20), font,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)


def get_data(origin_path, batch_size=8):
    np.random.seed(42)
    tf.random.set_seed(42)
    test_path = os.path.join(origin_path, "test")
    test_x = sorted(glob(os.path.join(test_path, "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(test_path, "mask", "*.jpg")))
    test_dataset = tf_dataset(
        test_x, test_y, batch_size=batch_size, parse_data_function=parse_data_wnet)

    test_steps = (len(test_x)//batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1
    return test_x, test_y, test_dataset, test_steps
