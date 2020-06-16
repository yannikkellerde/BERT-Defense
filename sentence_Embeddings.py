import numpy as np
import io
from sentence_transformers import SentenceTransformer
import logging
logger = logging.getLogger()


def init_model_roberta():
    return SentenceTransformer("roberta-large-nli-stsb-mean-tokens")


def get_most_likely_sentence(distribution,dic):
    sentence = ""
    for i,(p,c) in enumerate(distribution):
        pmaxin = np.argmax(p)
        if len(c) == 0:
            sentence += dic[pmaxin]
        else:
            csum = sum(x[1] for x in c)
            if p[pmaxin]/(1+csum)>c[0][1]:
                sentence += dic[pmaxin]
            else:
                for w in c[0][0]:
                    sentence += w.replace("##","")
        if i!=len(distribution)-1:
            sentence+=" "
    return sentence

def sentence_embedding_only_best_word(model, sentence):
    logger.debug("Encoding Sentence: "+sentence)
    sentences = [sentence]
    sentence_embeddings = model.encode(sentences,show_progress_bar=False)
    return sentence_embeddings[0]


def sentence_average_from_word_embeddings(posterior, dic, embeddings):
    sentence_embedding =[]
    word_vecs = []
    for word_distribution in posterior:
        word_vec = get_word_vec_from_distribution(word_distribution, dic, embeddings)
        word_vecs.append(word_vec)
    sentence_embedding.append(np.average(np.array(word_vecs), axis=0))
    return sentence_embedding


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


def get_word_vec_from_distribution(word_distribution, dic, embeddings):
    word_vecs = []
    weights = []
    for i in range(len(word_distribution)):
        if word_distribution[i] !=0:
            word_vecs.append(embeddings[dic[i]])
            weights.append(word_distribution[i])
    word_vecs = np.array(word_vecs)
    average_word = np.average(word_vecs, axis=0, weights=weights)
    return average_word
