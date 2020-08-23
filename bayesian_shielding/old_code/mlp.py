import tensorflow as tf
import keras
import numpy as np
import os
import pickle as pkl
from util import read_labeled_data
from tqdm import trange
import random
from RoBERTa_handler import init_model_roberta, simple_sentence_embedder



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


def create_model(layerss, dropout, activation_func):
    input1 = tf.keras.layers.Input(shape=(1024,))
    input2 = tf.keras.layers.Input(shape=(1024,))
    conc = tf.keras.layers.Concatenate()([input1, input2])
    drop = tf.keras.layers.Dropout(dropout)(conc)
    layer = tf.keras.layers.Dense(layerss[0], activation=activation_func)(drop)
    for l in layerss[1:]:
        drop = tf.keras.layers.Dropout(dropout)(layer)
        layer = tf.keras.layers.Dense(l, activation=activation_func)(drop)
    drop = tf.keras.layers.Dropout(dropout)(layer)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(drop)
    model = tf.keras.Model((input1, input2), out)
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["mse"])
    return model


def random_model_testing(n, batch_size=256):
    train_score, train_1, train_2, dev_score, dev_1, dev_2, test_1, test_2 = load_all_data()
    for i in trange(n):
        num_layers = random.randint(1, 10)
        dropout = random.uniform(0.05, 0.5)
        layers = []
        for i in range(num_layers):
            layers.append(random.randint(50,2000))
        model = create_model(layers, dropout, 'relu')
        filepath = "mlp_model/models/model_" + str(layers) + str(dropout) + ".h5" 
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=10),
                     tf.keras.callbacks.ModelCheckpoint(filepath=filepath)]
        model.fit((train_1, train_2), train_score, batch_size=batch_size, epochs=10000,
                  validation_data = ((dev_1, dev_2), dev_score), callbacks=callbacks,
                  verbose=1)
        if i%5 == 0:
            model_clean_up(dev_1, dev_2, dev_score)


def model_clean_up(dev_1, dev_2, dev_score):
    all_models = os.listdir("mlp_model/models")
    best_model = None
    best_file = None
    for model in all_models:
        filepath = "mlp_model/models/" + model
        m = tf.keras.models.load_model(filepath)
        challenger = m.evaluate((dev_1, dev_2), dev_score, verbose=0)[1]
        if not(best_model is None):
            if best_model > challenger:
                os.remove(best_file)
                best_model = challenger
                best_file = filepath
            else:
                os.remove(filepath)
        else:
            best_model = challenger
            best_file = filepath
    print("Best Model dev mse :" + str(best_model))


if __name__ == '__main__':
    train_score, train_1, train_2, dev_score, dev_1, dev_2, test_1, test_2 = load_all_data()
    #random_model_testing(50)
    model_clean_up(dev_1, dev_2, dev_score)
    
    
