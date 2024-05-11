# CNN MNIST

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import KFold
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils


gcs_utils._is_gcs_disabled = True
os_path = "cnn_flowers_weights"


def load_prepare_data():
    dataset, info = tfds.load(
        "oxford_flowers102", shuffle_files=True, as_supervised=True, with_info=True
    )

    dataset_train = dataset["train"].map(resize_normalize_image)
    dataset_test = dataset["test"].map(resize_normalize_image)

    return dataset_train, dataset_test


def visualize_data(dataset_train, dataset_test):
    fig = tfds.show_examples(dataset_train, dataset_test, rows=4, cols=4)


def resize_normalize_image(image, label):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_cnn(num_classes, dim_layer):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            dim_layer,
            (3, 3),
            (1, 1),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(
        tf.keras.layers.Conv2D(dim_layer, (3, 3), padding="same", activation="relu")
    )
    model.add(
        tf.keras.layers.Conv2D(dim_layer, (3, 3), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(
        tf.keras.layers.Conv2D(dim_layer, (3, 3), padding="same", activation="relu")
    )
    model.add(
        tf.keras.layers.Conv2D(dim_layer, (3, 3), padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dim_layer, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    return model


def compile_and_fit(
    model,
    dataset_train,
    dataset_test,
    batch_size,
    epochs,
    num_classes,
    dim_layer,
):
    accuracy_scores = []
    fold_models = []
    history_list = []

    num_folds = 3
    kf = KFold(n_splits=num_folds)

    for fold, (train_index, val_index) in enumerate(kf.split(dataset_train)):
        print(f"Fold {fold + 1}/{num_folds}")

        train_dataset = dataset.take(train_index)
        val_dataset = dataset.take(val_index)

        model = create_cnn(num_classes, dim_layer)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.005),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=["accuracy"],
        )

        history = model.fit(
            images_train,
            labels_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(images_test, labels_test),
            shuffle=True,
        )
        print(history.history)

        _, val_acc = model.evaluate(images_val_fold, labels_val_fold)
        accuracy_scores.append(val_acc)
        fold_models.append(model)
        history_list.append(history)

    best_model_index = np.argmax(accuracy_scores)
    best_model = fold_models[best_model_index]
    best_history = history_list[best_model_index]

    return best_model, best_history


def plot_learning_curves(history, epochs, image_path, dim_layer):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training / Validation")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training / Validation Loss ")

    plt.savefig(image_path)


if __name__ == "__main__":
    if not os.path.exists(os_path):
        os.makedirs(os_path)

    dim_layer_list = [8, 16, 32, 64]

    for dim_layer in dim_layer_list:
        image_path = os.path.join(os_path, "plot" + str(dim_layer) + ".png")

        num_classes = 102
        batch_size = 128
        epochs = 30

        dataset_train, dataset_test = load_prepare_data()

        print("\n\n dataset shape: ", type(dataset_train))
        cnn_model = create_cnn(num_classes, dim_layer)

        cnn_model, history = compile_and_fit(
            cnn_model,
            dataset_train,
            dataset_test,
            batch_size,
            epochs,
            num_classes,
            dim_layer,
        )

        model_path = os.path.join(os_path, "flowers_model_" + str(dim_layer) + ".h5")
        cnn_model.save(model_path)

        plot_learning_curves(history, epochs, image_path, dim_layer)
