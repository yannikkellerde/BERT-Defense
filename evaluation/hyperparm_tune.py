import os,sys
from tqdm import tqdm,trange
import pickle
import pandas as pd
import random
import metrics
sys.path.append("../bayesian_shielding")
sys.path.append("../adversarial_attacks")
from attack_api import Adversarial_attacker
from frontend.clean_sentences import Sentence_cleaner
from util.utility import cosine_similarity,read_labeled_data,get_most_likely_sentence_multidics
from benchmark_tasks.STSB.RoBERTa_handler import init_model_roberta,simple_sentence_embedder

class Tuner():
    def __init__(self):
        self.cleaner = Sentence_cleaner()
        self.attack_api = Adversarial_attacker()
        self.roberta = None
        self.tune_store = "../DATA/hyperparam_tune/"
        os.makedirs(self.tune_store,exist_ok=True)

    def likelihood_random_search(self, ground_truth, prior, iterations=100, stsb=True):
        # When stsb is True, ground_truth/prior shoud be a concatenation of all first sentences
        # and all second sentences.
        os.makedirs(os.path.join(self.tune_store,"cleaned"),exist_ok=True)
        if stsb and self.roberta is None:
            print("loading RoBERTa")
            self.roberta = init_model_roberta()
            print("done loading RoBERTa")
        assert len(ground_truth)==len(prior),"prior and ground_truth length must be equal"
        params_with_ranges = {
            "top_n":[1, 10, int],
            "bert_theta":[0.1, 0.6, float],
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
            print(params)
            self.cleaner.set_hyperparams(**params)
            posterior = self.cleaner.clean_sentences_given_prior(prior)
            tune_entry = params
            tune_entry.update({"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[]})
            with open(os.path.join(self.tune_store,"cleaned",f'{params["top_n"]}_{params["bert_theta"]}_{params["gtp_theta"]}'),"w") as f:
                f.write("\n".join(posterior))
            if stsb:
                self.posterior_embeddings = simple_sentence_embedder(self.roberta,posterior)
                l = len(self.posterior_embeddings)
                cosine_sims = [cosine_similarity(x,y) for x,y in zip(self.posterior_embeddings[:l/2],self.posterior_embeddings[l/2:])]
                tune_entry["stsb"] = sum(cosine_sims)/len(cosine_sims)
            for gt,post in zip(ground_truth,posterior):
                tune_entry["bleu"].append(metrics.bleu_score(gt,post))
                all_rouge = metrics.rouge_score(gt,post)
                tune_entry["rouge-1"].append(all_rouge["rouge-1"]["p"])
                tune_entry["rouge-4"].append(all_rouge["rouge-4"]["p"])
                tune_entry["rouge-l"].append(all_rouge["rouge-l"]["p"])
                tune_entry["rouge-w"].append(all_rouge["rouge-w"]["p"])
                tune_entry["mover"].append(metrics.mover_score(gt,post)[0])
            tune_entry["bleu"] = sum(tune_entry["bleu"])
            tune_entry["rouge-1"] = sum(tune_entry["rouge-1"])
            tune_entry["rouge-4"] = sum(tune_entry["rouge-4"])
            tune_entry["rouge-l"] = sum(tune_entry["rouge-l"])
            tune_entry["rouge-w"] = sum(tune_entry["rouge-w"])
            tune_entry["mover"] = sum(tune_entry["mover"])
            tune_entries.append(tune_entry)
            pd.DataFrame(tune_entries).to_csv(os.path.join(self.tune_store,"tune_info"))

    def create_prior_file(self,ground_truth):
        atk_sentences = []
        for sentence in ground_truth:
            random.shuffle(self.attack_api.methods)
            attacks_with_severity = [(x,0.2) for x in self.attack_api.methods[:2]]
            if attacks_with_severity[0][0]=="intrude":
                attacks_with_severity.reverse()
            atk_sentences.append(self.attack_api.multiattack(sentence,attacks_with_severity))
        with open("atk_sentences.txt", "w") as f:
            f.write("\n".join(atk_sentences))
        prior = self.cleaner.get_priors(atk_sentences)
        with open("prior.pkl", "wb") as f:
            pickle.dump(prior,f)

    def sentence_prior(self,prior,target="prior.txt"):
        sentences = []
        for i,p in enumerate(prior):
            print("\n",len(p),i)
            print(len(p[0]))
            print(p[0][0])
            print(len(p[0][1]),"\n")
            sentences.append(get_most_likely_sentence_multidics([b[0] for b in p[0][1]],[b[1] for b in p[0][1]]))
        with open(target, "w") as f:
            f.write("\n".join(sentences))

    def load_ground_truth(self,path="test_sentences.txt"):
        scores, first_sentences, second_sentences = read_labeled_data(path)
        return first_sentences+second_sentences

if __name__ == "__main__":
    tun = Tuner()
    ground_truth = tun.load_ground_truth()
    #tun.create_prior_file(ground_truth)
    with open("prior.pkl", "rb") as f:
        prior = pickle.load(f)
    tun.sentence_prior(prior)
    print(len(prior),len(prior[0]),len(ground_truth))
    tun.likelihood_random_search(ground_truth,prior,100,False)
