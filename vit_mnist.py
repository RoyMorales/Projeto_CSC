# Projeto -> Vit Mnist

# Imports
import tensorflow as tf
from model.vision_transformer import create_vit_classifier
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from func_reader import *

os_path = "vit_mnist_weights"


def load_data(): 
    (dataset_images_train, dataset_labels_train), (dataset_images_test, dataset_labels_test) = tf.keras.datasets.mnist.load_data()

    return dataset_images_train, dataset_labels_train, dataset_images_test, dataset_labels_test


def visualize_data(images_train, labels_train):
    classes = ['0','1','2','3','4','5','6','7','8','9']

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

    return images_train, labels_train , images_test, labels_test, input_shape


def compile_and_fit(
    images_train,
    labels_train,
    images_test,
    labels_test,
    batch_size,
    epochs,
    num_patches,
    size_patches
):
    accuracy_scores = []
    fold_models = []
    history_list = []

    num_folds = 2
    kf = KFold(n_splits=num_folds, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kf.split(images_train)):
        print(f"Fold {fold + 1}/{num_folds}")
        images_train_fold, images_val_fold = (
            images_train[train_index],
            images_train[val_index],
        )
        labels_train_fold, labels_val_fold = (
            labels_train[train_index],
            labels_train[val_index],
        )

        model = create_vit_classifier(input_shape=[28, 28, 1],
                                      num_classes=10,
                                      image_size=28,
                                      patch_size=size_patches,
                                      num_patches=num_patches,
                                      projection_dim=256,
                                      dropout=0.2,
                                      n_transformer_layers=1,
                                      num_heads=4,
                                      transformer_units=[512, 256,],
                                      mlp_head_units=[64])

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


def plot_learning_curves(history, epochs, image_path):
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

    num_classes = 10
    batch_size = 64
    epochs = 10

    size_patches_list = [7, 4, 2]

    for size_patches in size_patches_list:
        num_patches = (28 // size_patches) ** 2 
        print("\nNumber of Patches: ", num_patches)
        image_path = os.path.join(os_path, "plot" + str(num_patches) + ".png")
        images_train, labels_train, images_test, labels_test, input_shape = prepare_data()
        # visualize_data(images_train, labels_train)

        vit_model, history = compile_and_fit(
            images_train,
            labels_train,
            images_test,
            labels_test,
            batch_size,
            epochs,
            num_patches,
            size_patches
        )

        model_path = os.path.join(os_path, "mnist_model_" + str(num_patches) + ".h5")
        vit_model.save(model_path)

        plot_learning_curves(history, epochs, image_path)



