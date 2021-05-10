from util.utility import get_full_word_dict, preprocess_sentence
import numpy as np

word_dict = get_full_word_dict()

def word_to_prob(word):
    for i,dic_word in enumerate(word_dict):
        if dic_word == word:
            probs = np.zeros(len(word_dict))
            probs[i] = 1
            break
    else:
        probs = np.ones(len(word_dict))/len(word_dict)
    return (probs,word_dict)

def get_stupid_prior(sentences):
    out = []
    for sentence in sentences:
        words = preprocess_sentence(sentence)
        words = [word_to_prob(word) for word in words]
        out.append([(1,words)])
    return out