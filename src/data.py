# data loader placeholder
import os
import random
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image

def list_image_paths_labels(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Walk IDC_regular tree. Label = 1 if filename contains 'class1', else 0.
    """
    paths, labels = [], []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                label = 1 if "class1" in f.lower() else 0
                paths.append(os.path.join(root, f))
                labels.append(label)
    return paths, labels

def split_paths(
    paths: List[str],
    labels: List[int],
    test_size: float,
    val_size: float,
    seed: int
):
    # First split train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        paths, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    # Then split train vs val from trainval
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=seed, stratify=y_trainval
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _load_and_preprocess(path, label, image_size):
    # Read file
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, depth=2)

def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label

def make_tf_dataset(
    paths: List[str],
    labels: List[int],
    image_size: int,
    batch_size: int,
    buffer_size: int,
    augment: bool = False,
    shuffle: bool = True
):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.map(lambda p, y: _load_and_preprocess(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
