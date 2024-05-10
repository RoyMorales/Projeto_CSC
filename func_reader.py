# Functions

# Imports 
import gzip
import numpy as np

def read_dataset_images_train(num_images):
    image_dataset_train = gzip.open("./Dataset/train-images-idx3-ubyte.gz")
    image_size = 28

    image_dataset_train.read(16)
    buffer = image_dataset_train.read(image_size * image_size * num_images)
    data_image_train = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.float32)
    data_image_train = data_image_train.reshape(num_images, image_size, image_size)

    return data_image_train


def read_dataset_images_test(num_images):
    image_dataset_test = gzip.open("./Dataset/t10k-images-idx3-ubyte.gz")
    image_size = 28

    image_dataset_test.read(16)
    buffer = image_dataset_test.read(image_size * image_size * num_images)
    data_image_test = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.float32)
    data_image_test = data_image_test.reshape(num_images, image_size, image_size)

    return data_image_test


def read_dataset_labels_train(num_labels):
    labels_dataset_train = gzip.open("./Dataset/train-labels-idx1-ubyte.gz")
    labels_dataset_train.read(8)

    buffer = labels_dataset_train.read(num_labels)
    labels = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.uint64)

    return labels


def read_dataset_labels_test(num_labels):
    labels_dataset_test = gzip.open("./Dataset/t10k-labels-idx1-ubyte.gz")
    labels_dataset_test.read(8)

    buffer = labels_dataset_test.read(num_labels)
    labels = np.frombuffer(buffer, dtype=np.uint8)  # .astype(np.uint64)

    return labels

