import sys
import numpy as np
sys.path.append("..")
from context_bert.bert_posterior import BertPosterior
from edit_distance.substring_distance import Sub_dist
from util.utility import get_full_word_dict,get_most_likely_sentence_multidics,preprocess_sentence,softmax

class Sentence_cleaner():
    def __init__(self):
        self.context_bert = BertPosterior()
        self.dist_handler = Sub_dist()
        self.bert_iterations = 5
        self.hyperparams = self.context_bert.hyperparams+self.dist_handler.hyperparams+["bert_iterations"]

    def set_hyperparams(self, **kwargs):
        self.context_bert.set_hyperparams(**kwargs)
        self.dist_handler.set_hyperparams(**kwargs)

    def clean_sentence(self, sentence, progress=False):
        tokens = preprocess_sentence(sentence)
        print(tokens)
        if progress:
            print("initialized bert")
        all_hyps = []
        hypothesis = self.dist_handler.get_sentence_hypothesis(tokens,progress=progress)
        for i,(prob,content) in enumerate(hypothesis):
            prior = [x[0] for x in content]
            word_dics = [x[1] for x in content]
            if progress:
                print("Prior:",get_most_likely_sentence_multidics(prior,word_dics))
            posterior = self.context_bert.bert_posterior_for_hypothesis(prior,word_dics,self.bert_iterations,orig_prior=prior.copy(),verbose=progress)
            post_sent = get_most_likely_sentence_multidics(prior,word_dics)
            all_hyps.append((prob,post_sent))
            if progress:
                print(f"Hypothesis {i+1}, Prior prob: {prob}, Posterior: {post_sent}")
        posterior_hyps = self.context_bert.gtp_hypothesis(all_hyps)
        if progress:
            print("All posterior hypothesis",posterior_hyps)
        print(f"\nFinal cleaned sentence: {posterior_hyps[0][1]}")
        return posterior_hyps[0][1]

if __name__ == "__main__":
    cleaner = Sentence_cleaner()
    #cleaner.clean_sentence("Abycìisƫınblзókanöωriճe1ińaspèedіnΜpdstspeϲțɜtoɦԍ.")
    #cleaner.clean_sentence("`,`Opi;ni;o;natedands)tu)b)b)o)rn'''becomestreatwi/thdrugs.")
    cleaner.clean_sentence("well-done riidng the skateboard",progress=True)