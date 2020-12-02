import sys
import numpy as np
sys.path.append("..")
from tqdm import tqdm
from context_bert.bert_posterior import BertPosterior
from edit_distance.substring_distance import Sub_dist
from edit_distance.multiprocess_distances import multiprocess_prior
from util.utility import get_full_word_dict,get_most_likely_sentence_multidics,preprocess_sentence,softmax

class Sentence_cleaner():
    def __init__(self):
        self.context_bert = None
        self.dist_handler = Sub_dist()
        self.hyperparams = self.dist_handler.hyperparams

    def set_hyperparams(self, **kwargs):
        self.load_bert()
        self.context_bert.set_hyperparams(**kwargs)
        self.dist_handler.set_hyperparams(**kwargs)

    def load_bert(self):
        if self.context_bert is None:
            self.context_bert = BertPosterior()
            self.hyperparams = self.context_bert.hyperparams+self.dist_handler.hyperparams

    def clean_sentence(self, sentence, progress=False,verbose=False):
        self.load_bert()
        tokens = preprocess_sentence(sentence)
        if verbose:
            print(tokens)
        all_hyps = []
        hypothesis = self.dist_handler.get_sentence_hypothesis(tokens,progress=progress)
        for i,(prob,content) in enumerate(hypothesis):
            prior = [x[0] for x in content]
            word_dics = [x[1] for x in content]
            if verbose:
                print(f"Hypothesis {i+1}, Probability: {prob}, Prior:{get_most_likely_sentence_multidics(prior,word_dics)}")
            posterior = self.context_bert.bert_posterior_for_hypothesis(prior,word_dics,len(prior),orig_prior=prior.copy(),verbose=verbose)
            post_sent = get_most_likely_sentence_multidics(prior,word_dics)
            all_hyps.append((prob,post_sent))
            if verbose:
                print(f"Hypothesis {i+1}, Prior prob: {prob}, Posterior: {post_sent}")
        posterior_hyps = self.context_bert.gpt_hypothesis(all_hyps,verbose=verbose)
        if verbose:
            print("All posterior hypothesis",posterior_hyps)
        print(f"\nFinal cleaned sentence: {posterior_hyps[0][1]}")
        return posterior_hyps[0][1]

    def batched_clean_given_prior(self,priors,batch_size=64):
        self.load_bert()
        all_posterior = []
        all_posterior_hyps = self.context_bert.batch_bert_posterior(priors,batch_size=batch_size)
        for hyps in tqdm(all_posterior_hyps):
            posterior_hyps = self.context_bert.gpt_hypothesis(hyps)
            all_posterior.append(posterior_hyps[0][1])
        return all_posterior

    def clean_sentences_given_prior(self,prior,progress=False):
        self.load_bert()
        all_posterior = []
        for hypothesis in prior:
            all_hyps = []
            for i,(prob,content) in enumerate(hypothesis):
                prior = [x[0] for x in content]
                word_dics = [x[1] for x in content]
                if progress:
                    print("Prior:",get_most_likely_sentence_multidics(prior,word_dics))
                posterior = self.context_bert.bert_posterior_for_hypothesis(prior,word_dics,len(prior),orig_prior=prior.copy(),verbose=progress)
                post_sent = get_most_likely_sentence_multidics(prior,word_dics)
                all_hyps.append((prob,post_sent))
                if progress:
                    print(f"Hypothesis {i+1}, Prior prob: {prob}, Posterior: {post_sent}")
            posterior_hyps = self.context_bert.gpt_hypothesis(all_hyps)
            if progress:
                print("All posterior hypothesis",posterior_hyps)
            print(f"\nFinal cleaned sentence: {posterior_hyps[0][1]}")
            all_posterior.append(posterior_hyps[0][1])
        return all_posterior

    def get_priors(self,sentences,store_path=None):
        return multiprocess_prior(self.dist_handler, sentences,store_path=store_path)
    
    def clean_sentences(self,sentences,progress=False):
        return self.clean_sentences_given_prior(self.get_priors(sentences),progress)


if __name__ == "__main__":
    cleaner = Sentence_cleaner()
    cleaner.clean_sentence(sys.argv[1],progress=True,verbose=True)
    #cleaner.dist_handler.cheap_actions = {key:False for key in cleaner.dist_handler.cheap_actions}
    #cleaner.clean_sentence(r"Ȧgɍicǘɭture mini*ś*ters fɼoᵐ moɺe than őnē h,uᶮᶑȓed ᴎ^ͣ^tȉoᴺs aře éx/pe/cte/ᵭ t.o attɛnd tᶣ$e thr&ee-da&ȳ Ministɝrîal Ȼonᶠerençe and Ȩxpo on Ą\"gɾɨcul\"tȕřâl ∾cīenᶝe a\ŉ\d Ṱǝchnoᶩǭgy sponșoreḓ bỷ ẗₕe U.S. Deᵖaȓṫmênt ơf Agrį|cű|ᷞt|urê.",progress=True)
