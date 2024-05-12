# CNN Dog

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle

data_dir = "./Dataset_Dog/"
os_path = "cnn_dog_weights"


def load_prepare_data():
    classes = os.listdir(data_dir)
    class_labels = {class_name: i for i, class_name in enumerate(classes)}

    images = []
    labels = []

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        print("Class: ", class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            img = np.asarray(img).astype(np.float32)
            images.append(img)
            labels.append(class_labels[class_name])

    images = np.array(images)
    labels = np.array(labels)
    print("Shape Images: ", images.shape)
    print("Shape Labels: ", labels.shape)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    return train_images, train_labels, test_images, test_labels

def create_cnn(num_classes, dim_layer):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Conv2D(
            dim_layer,
            (3, 3),
            (1, 1),
            padding="same",
            activation="relu",
            input_shape=(128, 128, 3),
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
    images_train,
    labels_train,
    images_test,
    labels_test,
    batch_size,
    epochs,
    num_classes,
    dim_layer,
):
    accuracy_scores = []
    fold_models = []
    history_list = []

    num_folds = 2
    kf = KFold(n_splits=num_folds)

    for fold, (train_index, val_index) in enumerate(kf.split(images_train)):
        print(f"Fold {fold + 1}/{num_folds}")

        imges_train_fold, images_val_fold = (
                images_train[train_index],
                images_train[val_index],
                )

        labels_train_fold, labels_val_fold = (
                images_train[train_index],
                labels_train[val_index],
                )

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

    print("\n\nDone")

    plt.figure(figsize=(8,8))
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

    print("\n\nDone")

    plt.savefig(image_path)


if __name__ == "__main__":
    if not os.path.exists(os_path):
        os.makedirs(os_path)

    dim_layer_list = [32, 64]
    images_train, labels_train, images_test, labels_test = load_prepare_data()


    num_classes = 120
    batch_size = 4
    epochs = 10

    for dim_layer in dim_layer_list:
        print("Dim Layer: ", dim_layer)
        image_path = os.path.join(os_path, "plot" + str(dim_layer) + ".png")

        cnn_model, history = compile_and_fit(
            images_train,
            labels_train,
            images_test,
            labels_test,
            batch_size,
            epochs,
            num_classes,
            dim_layer
        )

        model_path = os.path.join(os_path, "dog_model_" + str(dim_layer) + ".h5")
        history_path = os.path.join(os_path, "dog_history_" + str(dim_layer) + ".pkl")
        cnn_model.save(model_path)

        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

        #plot_learning_curves(history, epochs, image_path, dim_layer)
