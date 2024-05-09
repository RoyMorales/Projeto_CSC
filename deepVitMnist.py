# Projeto -> DeepVit Mnist

# Imports
import tensorflow as tf
from vit import ViT

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return (x_train, y_train), (x_test, y_test), classes

def prepare_data():
    # Load Data
    (x_train, y_train), (x_test, y_test), classes = load_data()

    # Normalize Data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Reshape Data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return x_train, y_train, x_test, y_test, classes

def create_vit(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):

    model = ViT(image_size=image_size,
                patch_size=patch_size,
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                emb_dropout=emb_dropout)
    
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
    image_size = 28
    patch_size = 2
    num_classes = 10
    dim = 16
    depth = 4
    heads = 16
    mlp_dim = 128
    dropout = 0.1
    emb_dropout = 0.1

    batch_size=128
    epochs=10
    num_predictions=20


    x_train, y_train, x_test, y_test, classes = prepare_data()
    vit_model = create_vit(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout)

    vit_model, history = compile_and_fit(vit_model, x_train, y_train, x_test, y_test, batch_size, epochs)

    score = vit_model.evaluate(x_test, y_test)
    print('Evaluation: ', score[0])
    print('Evaluation: ', score[1])




