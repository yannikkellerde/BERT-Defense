import sys
sys.path.append("..")
from context_bert.bert_posterior import bert_posterior,bert_posterior_probabilistic
from edit_distance.edit_distance import get_word_dic_distance
from util.util import get_full_word_dict,get_most_likely_sentence,preprocess_sentence

import numpy as np

dictionary = get_full_word_dict()

def get_tokens_prior(tokens,dictionary):
    prior = np.empty((len(tokens),len(dictionary)))
    for i,token in enumerate(tokens):
        prior[i] = get_word_dic_distance(token,dictionary,cheap_actions=True,progress=True)
    return prior

def clean_sentence(sentence):
    tokens = preprocess_sentence(sentence)
    prior = get_tokens_prior(tokens,dictionary)
    print(get_most_likely_sentence(prior,dictionary))
    posterior_old = bert_posterior(prior,dictionary,10)
    print(get_most_likely_sentence(posterior_old,dictionary))
    posterior_new = bert_posterior_probabilistic(prior,dictionary,3)
    print(get_most_likely_sentence(posterior_new,dictionary))

if __name__ == '__main__':
    print(clean_sentence(sys.argv[1]))