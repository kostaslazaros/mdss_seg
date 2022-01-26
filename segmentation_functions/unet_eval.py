import os
import numpy as np
from tqdm import tqdm
from glob import glob
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from segmentation_functions.iofunctions import create_dir
from segmentation_functions.parameters import SIZE
from segmentation_functions.metrics import iou


def evaluate_normal(model, x_data, y_data, result_dir):
    create_dir(result_dir)
    for i, (img_path, msk_path) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        img = read_image_plain(img_path)
        msk = read_mask_plain(msk_path)
        ground_truth = mask_to_3d(msk)

        predicted_mask = unet_mask_predict(img/255.0, model)
        predict_resized = cv2.resize(
            predicted_mask, (img.shape[1], img.shape[0]))
        image_overlayed = cv2.addWeighted(
            img, 0.7, predict_resized, 0.3, 0, dtype=cv2.CV_64F)
        white_line = vertical_whiteline(img)

        all_images = [
            img,
            image_overlayed,
            ground_truth,
            predicted_mask
        ]
        titles = ('Image', 'Overlayed', 'Ground Truth', 'Prediction')
        image = create_result_image(all_images, white_line, titles)
        save2disk(image, img_path, result_dir)


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


def unet_mask_predict(image, model):
    pred = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    pred255 = mask_to_3d(pred) * 255.0
    pred255_resized = cv2.resize(pred255, (image.shape[1], image.shape[0]))
    return pred255_resized


def vertical_whiteline(image, line_width=10):
    h, _, _ = image.shape
    white_line = np.ones((h, line_width, 3)) * 255.0
    return white_line


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def read_image_plain(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, SIZE)
    return image


def read_mask_plain(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, SIZE)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def read_image_decoded(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, SIZE)
    image = image.astype(np.double)
    return image


def read_mask_decoded(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, SIZE)
    mask = mask.astype(np.double)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def tf_parse(image, mask):
    def _parse(image, mask):
        image = read_image_decoded(image)
        mask = read_mask_decoded(mask)
        return image, mask

    image, mask = tf.numpy_function(
        _parse, [image, mask], [tf.float64, tf.float64])
    image.set_shape([SIZE[1], SIZE[0], 3])
    mask.set_shape([SIZE[1], SIZE[0], 1])
    return image, mask


def tf_dataset(image, mask, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((image, mask))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset


def get_data(origin_path, batch_size=8):
    np.random.seed(42)
    tf.random.set_seed(42)
    test_path = os.path.join(origin_path, "test")
    test_x = sorted(glob(os.path.join(test_path, "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(test_path, "mask", "*.jpg")))
    test_dataset = tf_dataset(test_x, test_y, batch_size=batch_size)

    test_steps = (len(test_x)//batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1
    return test_x, test_y, test_dataset, test_steps


def get_pretrained_model(model_path):
    with CustomObjectScope({'iou': iou}):
        model = load_model(model_path)
        return model
