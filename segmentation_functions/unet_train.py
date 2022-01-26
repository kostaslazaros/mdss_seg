import os
from glob import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers as lay
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from segmentation_functions.parameters import SIZE
from segmentation_functions.metrics import iou


def conv_block(x, num_filters):
    x = lay.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = lay.BatchNormalization()(x)
    x = lay.Activation("relu")(x)
    x = lay.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = lay.BatchNormalization()(x)
    x = lay.Activation("relu")(x)
    return x


def build_unet_model():
    num_filters = [16, 32, 48, 64]
    inputs = lay.Input((SIZE[1], SIZE[0], 3))
    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = lay.MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])
    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = lay.UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = lay.Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = lay.Conv2D(1, (1, 1), padding="same")(x)
    x = lay.Activation("sigmoid")(x)

    return Model(inputs, x)


def load_data(path):
    split_tuples = {}
    for tpath in ('test', 'train', 'valid'):
        images = sorted(glob(os.path.join(path, tpath,"image/*")))
        masks = sorted(glob(os.path.join(path, tpath, "mask/*")))
        split_tuples[tpath] = (images, masks)
    return split_tuples['train'], split_tuples['valid'], split_tuples['test']


def read_image_mask(path, typos='image'):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if typos == 'mask':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, SIZE)
    image = image/255.0
    if typos == 'mask':
        image = np.expand_dims(image, axis=-1)
    return image


def tf_parse(image, mask):
    def _parse(image, mask):
        image = read_image_mask(image, typos='image')
        mask = read_image_mask(mask, typos='mask')
        return image, mask

    image, mask = tf.numpy_function(_parse, [image, mask], [tf.float64, tf.float64])
    image.set_shape([SIZE[1], SIZE[0], 3])
    mask.set_shape([SIZE[1], SIZE[0], 1])
    return image, mask


def tf_dataset(image, mask, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((image, mask))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


def train(model, modelh5, csv_logger, epochs=20, dataset_path='cvc_ready'):
    (train_x, train_y), (valid_x, valid_y), _ = load_data(dataset_path)

    ## Hyperparameters
    batch = 8
    lr = 1e-4

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint(modelh5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger(csv_logger),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )