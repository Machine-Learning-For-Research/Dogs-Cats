import tensorflow as tf
import numpy as np
import os

PATH_TRAIN = "data/train"
PATH_TEST = "data/test"


def read_data(data_dir):
    if not os.path.exists(data_dir):
        raise IOError("Not dir found.")
    images_cat, images_dog = [], []
    labels_cat, labels_dog = [], []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if file_name.split(".")[0] == "cat":
            images_cat.append(file_path)
            labels_cat.append(0)
        else:
            images_dog.append(file_path)
            labels_dog.append(1)

    images = np.hstack([images_cat, images_dog])
    labels = np.hstack([labels_cat, labels_dog])
    temp = np.array([images, labels]).transpose()

    images_list = list(temp[:, 0])
    labels_list = [int(label) for label in temp[:, 1]]
    # labels_list = [[1 - int(label), int(label)] for label in temp[:, 1]]

    return images_list, labels_list


def get_batch(images, labels, image_W, image_H, batch_size, capacity, num_threads=64, standardization=True):
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)

    queue = tf.train.slice_input_producer([images, labels])
    label = queue[1]
    image_path = queue[0]
    image_data = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)
    if standardization:
        image = tf.image.per_image_standardization(image)

    batch_image, batch_label = tf.train.batch([image, label], batch_size, num_threads, capacity)
    batch_image = tf.cast(batch_image, tf.float32)
    return batch_image, batch_label



# images, labels = read_data(PATH_TRAIN)
# batch_image, batch_label = get_batch(images, labels, 100, 100, 3, 100)
# print batch_image.get_shape()
# print len(labels)