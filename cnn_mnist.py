# CNN MNIST

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    return (x_train, y_train), (x_test, y_test), classes


def visualize_data(x_train, y_train, classes):
    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(x_train[i]))
        plt.xlabel(classes[y_train[i]])
    plt.show()


def prepare_data():
    (x_train, y_train), (x_test, y_test), classes = load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return x_train, y_train, x_test, y_test, classes


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
    model, x_train, y_train, x_test, y_test, batch_size, epochs, apply_data_augmentation
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )

    if not apply_data_augmentation:
        print("No data augmentation")

        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
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

    x_train, y_train, x_test, y_test, classes = prepare_data()

    cnn_model = create_cnn(num_classes)

    cnn_model, history = compile_and_fit(
        cnn_model,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size,
        epochs,
        apply_data_augmentation,
    )

    score = cnn_model.evaluate(x_test, y_test)
    print("Evaluation Loss: ", score[0])
    print("Evaluation Accuracy: ", score[1])
