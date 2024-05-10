# CNN MNIST

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
from func_reader import *


def load_data():
    dataset_size = 60000
    dataset_size_test = 10000

    dataset_images_train = read_dataset_images_train(dataset_size)
    dataset_labels_train = read_dataset_labels_train(dataset_size)
    dataset_images_test = read_dataset_images_test(dataset_size_test)
    dataset_labels_test = read_dataset_labels_test(dataset_size_test)

    return (
        dataset_images_train,
        dataset_labels_train,
        dataset_images_test,
        dataset_labels_test,
    )


def visualize_data(images_train, labels_train):
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(images_train[i]))
        plt.xlabel(classes[labels_train[i]])
    plt.show()


def prepare_data():
    dataset_images_train, labels_train, dataset_images_test, labels_test = load_data()

    images_train = dataset_images_train.astype("float32") / 255
    images_test = dataset_images_test.astype("float32") / 255
    images_train = dataset_images_train.reshape(
        dataset_images_train.shape[0], 28, 28, 1
    )
    images_test = dataset_images_test.reshape(dataset_images_test.shape[0], 28, 28, 1)

    return images_train, labels_train, images_test, labels_test


def create_cnn(num_classes):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            (1, 1),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    model.summary()

    return model


def compile_and_fit(
    model,
    images_train,
    labels_train,
    images_test,
    labels_test,
    batch_size,
    epochs,
    apply_data_augmentation,
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )

    if not apply_data_augmentation:
        print("No data augmentation")

        history = model.fit(
            images_train,
            labels_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(images_test, labels_test),
            shuffle=True,
        )
    else:
        print("Not Done")

    return model, history


if __name__ == "__main__":
    num_classes = 10
    batch_size = 128
    epochs = 10
    apply_data_augmentation = False
    num_predictions = 20

    images_train, labels_train, images_test, labels_test = prepare_data()
    visualize_data(images_train, labels_train)

    cnn_model = create_cnn(num_classes)

    cnn_model, history = compile_and_fit(
        cnn_model,
        images_train,
        labels_train,
        images_test,
        labels_test,
        batch_size,
        epochs,
        apply_data_augmentation,
    )

    score = cnn_model.evaluate(images_test, labels_test)
    print("Evaluation Loss: ", score[0])
    print("Evaluation Accuracy: ", score[1])
