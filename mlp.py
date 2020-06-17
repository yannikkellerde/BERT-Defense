import tensorflow as tf
import keras
import numpy as np
import os
import pickle as pkl
from util import read_labeled_data
from sentence_Embeddings import init_model_roberta, simple_sentence_embedder



def load_all_data():
    model = init_model_roberta()
    if os.path.exists("storage_third_version/embeddings_robert.pkl"):
        with open("storage_third_version/embeddings_robert.pkl","rb") as f:
            test_set = pkl.load(f)
    else:
        test_set = []
        print("Pls run __main__.py for test_set")
    test_1, test_2 = zip(*test_set)
    train_score, train_1, train_2 = read_labeled_data("DATA/training-dataset.txt")
    dev_score, dev_1, dev_2 = read_labeled_data("DATA/development-dataset.txt")
    if os.path.exists("mlp_model/embeddings_t_1.pkl"):
        with open("mlp_model/embeddings_t_1.pkl", "rb") as f:
            train_1 = pkl.load(f)
        with open("mlp_model/embeddings_t_2.pkl", "rb") as f:
            train_2 = pkl.load(f)
    else:
        train_1 = simple_sentence_embedder(model, train_1)
        train_2 = simple_sentence_embedder(model, train_2)
        with open("mlp_model/embeddings_t_1.pkl", "wb") as f:
            pkl.dump(train_1, f)
        with open("mlp_model/embeddings_t_2.pkl", "wb") as f:
            pkl.dump(train_2, f)   
    if os.path.exists("mlp_model/embeddings_d_1.pkl"):
        with open("mlp_model/embeddings_d_1.pkl", "rb") as f:
            dev_1 = pkl.load(f)
        with open("mlp_model/embeddings_d_2.pkl", "rb") as f:
            dev_2 = pkl.load(f)
    else:
        dev_1 = simple_sentence_embedder(model, dev_1)
        dev_2 = simple_sentence_embedder(model, dev_2)
        with open("mlp_model/embeddings_d_1.pkl", "wb") as f:
            pkl.dump(dev_1, f)
        with open("mlp_model/embeddings_d_2.pkl", "wb") as f:
            pkl.dump(dev_2, f)
    return train_score, train_1, train_2, dev_score, dev_1, dev_2, test_1, test_2


def create_model(layerss, dropouts, activation_func):
    input1 = tf.keras.layers.Input(shape=(1024,))
    input2 = tf.keras.layers.Input(shape=(1024,))
    conc = tf.keras.layers.Concatenate()([input1, input2])
    drop = tf.keras.layers.Dropout(dropouts[0])(conc)
    layer = tf.keras.layers.Dense(layerss[0], activation=activation_func)(drop)
    for l, d in zip(layerss[1:],dropouts[1:-1]):
        drop = tf.keras.layers.Dropout(d)(layer)
        layer = tf.keras.layers.Dense(l, activation=activation_func)(drop)
    drop = tf.keras.layers.Dropout(dropouts[-1])(layer)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(drop)
    model = tf.keras.Model((input1, input2), out)
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["mse"])
    return model


if __name__ == '__main__':
    layers = [1024]
    dropout  = [0.3, 0.3]
    batch_size = 256
    epochs = 200
    activation_func = "relu"
    model = create_model(layers, dropout, activation_func)
    train_score, train_1, train_2, dev_score, dev_1, dev_2, test_1, test_2 = load_all_data()
    checkpointer = tf.keras.callbacks.ModelCheckpoint("mlp_model/best.model", save_best_only=True, 
                                                      verbose=1,save_weights_only=True,
                                                      monitor='val_loss')
    model.fit((train_1, train_2), train_score, batch_size=batch_size, epochs=epochs,
              validation_data = ((dev_1, dev_2), dev_score), callbacks=[checkpointer])
    model.load_weights("mlp_model/best.model")
    pred = model.predict((test_1, test_2))
    with open("mlp_model/scores.txt","w") as f:
        f.write("\n".join([str(x[0]) for x in pred]))
    
