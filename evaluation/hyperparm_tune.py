import os,sys
from tqdm import tqdm,trange
import pandas as pd
import random
import metrics
sys.path.append("../bayesian_shielding")
from edit_distance.clean_sentences import Sentence_cleaner
from util.utility import cosine_similarity
from benchmark_tasks.STSB.RoBERTa_handler import init_model_roberta,simple_sentence_embedder

class Tuner():
    def __init__(self):
        self.cleaner = Sentence_cleaner()
        self.roberta = init_model_roberta()
        self.likelihood_store = "../DATA/hyperparam_tune/likelihood.csv"
        os.makedirs(os.path.dirname(self.likelihood_store),exist_ok=True)

    def likelihood_random_search(self, ground_truth, prior, iterations=100, stsb=True):
        # When stsb is True, ground_truth/prior shoud be a concatenation of all first sentences
        # and all second sentences.
        assert len(ground_truth)==len(prior),"prior and ground_truth length must be equal"
        params_with_ranges = {
            "top_n":[1, 100, int],
            "bert_theta":[0.1, 1, float],
            "gtp_theta":[0.001, 0.02, float]
        }
        tune_entries = []
        for i in trange(iterations):
            params = {}
            for key, value in params_with_ranges.items():
                if value[2] == int:
                    params[key] = random.randint(value[0], value[1])
                elif value[2] == float:
                    params[key] = random.random()*(value[1]-value[0])+value[0]
            self.cleaner.set_hyperparams(**params)
            posterior = self.cleaner.clean_sentences_given_prior(prior)
            tune_entry = params
            tune_entry.update({"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[]})
            if stsb:
                self.posterior_embeddings = simple_sentence_embedder(self.roberta,posterior)
                l = len(self.posterior_embeddings)
                cosine_sims = [cosine_similarity(x,y) for x,y in zip(self.posterior_embeddings[:l/2],self.posterior_embeddings[l/2:])]
                tune_entry["stsb"] = sum(cosine_sims)/len(cosine_sims)
            for gt,post in zip(ground_truth,posterior):
                tune_entry["bleu"].append(metrics.bleu_score(gt,post))
                all_rouge = metrics.rouge_score(gt,post)
                tune_entry["rouge-1"] = all_rouge["rouge-1"]["p"]
                tune_entry["rouge-4"] = all_rouge["rouge-4"]["p"]
                tune_entry["rouge-l"] = all_rouge["rouge-l"]["p"]
                tune_entry["rouge-w"] = all_rouge["rouge-w"]["p"]
                tune_entry["mover"] = metrics.mover_score(gt,post)
            tune_entries.append(tune_entry)
            pd.DataFrame(tune_entries).to_csv(self.likelihood_store)

    def create_prior_file(self):

