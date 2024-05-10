# Projeto -> DeepVit Mnist

# Imports
import tensorflow as tf
from vit import ViT

def load_data():
    dataset_size = 60000
    dataset_size_test = 10000

    dataset_images_train = read_dataset_images_train(dataset_size)
    dataset_labels_train = read_dataset_labels_train(dataset_size)
    dataset_images_test = read_dataset_images_test(dataset_size_test)
    dataset_labels_test = read_dataset_labels_test(dataset_size_test)

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
    images_train = dataset_images_train.reshape(dataset_images_train.shape[0], 28, 28, 1)
    images_test = dataset_images_test.reshape(dataset_images_test.shape[0], 28, 28, 1)

    return images_train, labels_train , images_test, labels_test

def create_vit(config):

    model = ViT(config)
    model.summary()
    
    return model

def compile_and_fit(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test,y_test),
                        shuffle=True)

    return model, history




if __name__ == "__main__":
    config = {}
    config["num_layers"] = 5
    config["hidden_dim"] = 64
    config["mlp_dim"] = 128
    config["num_heads"] = 4
    config["dropout_rate"] = 0.1
    config["num_patches"] = 49
    config["patch_size"] = 4
    config["num_channels"] = 1

    batch_size=128
    epochs=10
    num_predictions=20


    x_train, y_train, x_test, y_test, classes = prepare_data()
    vit_model = create_vit(config)

    patches_x_train = tf.image.extract_patches()

    vit_model, history = compile_and_fit(vit_model, x_train, y_train, x_test, y_test, batch_size, epochs)

    score = vit_model.evaluate(x_test, y_test)
    print('Evaluation: ', score[0])
    print('Evaluation: ', score[1])




