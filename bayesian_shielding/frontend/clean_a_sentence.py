import sys
sys.path.append("..")
from context_bert.bert_posterior import BertPosterior
from edit_distance.edit_distance import get_word_dic_distance
from util.util import get_full_word_dict,get_most_likely_sentence,preprocess_sentence,softmax

import numpy as np

dictionary = get_full_word_dict()

def get_tokens_prior(tokens,dictionary):
    prior = np.empty((len(tokens),len(dictionary)))
    for i,token in enumerate(tokens):
        prior[i,:] = get_word_dic_distance(token,dictionary,cheap_actions=True,progress=True,keep_order=True)
    return prior

def clean_sentence(sentence):
    tokens = preprocess_sentence(sentence)
    print(tokens)
    context_bert = BertPosterior()
    print("initialized bert")
    prior = get_tokens_prior(tokens,dictionary)
    print("Prior:",get_most_likely_sentence(prior,dictionary))
    posterior_old = context_bert.bert_posterior_old(prior,10)
    print("Old Posterior:",get_most_likely_sentence(posterior_old,dictionary))
    posterior_live = context_bert.bert_posterior_probabilistic_live(prior,5,10,orig_prior=prior.copy())
    print("New Posterior live:",get_most_likely_sentence(posterior_live,dictionary))
    posterior_rounds = context_bert.bert_posterior_probabilistic_rounds(prior,5,2,theta=0.1)
    print("New Posterior rounds:",get_most_likely_sentence(posterior_rounds,dictionary))

if __name__ == '__main__':
    print(clean_sentence(sys.argv[1]))