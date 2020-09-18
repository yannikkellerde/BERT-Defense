import sys
import numpy as np
sys.path.append("..")
from context_bert.bert_posterior import BertPosterior
from edit_distance.substring_distance import Sub_dist
from util.utility import get_full_word_dict,get_most_likely_sentence_multidics,preprocess_sentence,softmax

def clean_sentence(sentence, context_bert=None, dist_handler=None, progress=False):
    tokens = preprocess_sentence(sentence)
    print(tokens)
    if context_bert is None:
        context_bert = BertPosterior()
    if dist_handler is None:
        dist_handler = Sub_dist()
    if progress:
        print("initialized bert")
    all_hyps = []
    hypothesis = dist_handler.get_sentence_hypothesis(tokens,progress=progress)
    for i,(prob,content) in enumerate(hypothesis):
        prior = [x[0] for x in content]
        word_dics = [x[1] for x in content]
        if progress:
            print("Prior:",get_most_likely_sentence_multidics(prior,word_dics))
        posterior = context_bert.bert_posterior_for_hypothesis(prior,word_dics,5,5,orig_prior=prior.copy())
        post_sent = get_most_likely_sentence_multidics(prior,word_dics)
        all_hyps.append((prob,post_sent))
        if progress:
            print(f"Hypothesis {i+1}, Prior prob: {prob}, Posterior: {post_sent}")
    posterior_hyps = context_bert.gtp_hypothesis(all_hyps)
    if progress:
        print("All posterior hypothesis",posterior_hyps)
    print(f"\nFinal cleaned sentence: {posterior_hyps[0][1]}")
    return posterior_hyps[0][1]


def clean_sentence_init():
    context_bert = BertPosterior()
    dist_handler = Sub_dist()
    return context_bert, dist_handler

if __name__ == "__main__":
    clean_sentence("Abycìisƫınblзókanöωriճe1ińaspèedіnΜpdstspeϲțɜtoɦԍ.")
    clean_sentence("`,`Opi;ni;o;natedands)tu)b)b)o)rn'''becomestreatwi/thdrugs.")