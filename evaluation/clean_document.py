import os,sys
from tqdm import tqdm,trange
import pickle
import pandas as pd
import random
sys.path.append("../bayesian_shielding")
from frontend.clean_sentences import Sentence_cleaner
from util.utility import read_labeled_data,get_most_likely_sentence_multidics

cleaner = Sentence_cleaner()
def clean_document(infile):
    basename = os.path.basename(infile).split(".")[0]
    scores,first_sentences, second_sentences = read_labeled_data(infile)
    sentences = first_sentences + second_sentences
    prior = cleaner.get_priors(sentences)
    with open(f"cleaned/priors/pkls/{basename}.pkl", "wb") as f:
        pickle.dump(prior, f)
    prior_cleaned = [get_most_likely_sentence_multidics([b[0] for b in p[0][1]],[b[1] for b in p[0][1]]) for p in prior]
    post_cleaned = cleaner.clean_sentences_given_prior(prior)
    
    prior_it = zip(scores,prior_cleaned[:len(first_sentences)],prior_cleaned[len(first_sentences):])
    post_it = zip(scores,post_cleaned[:len(first_sentences)],post_cleaned[len(first_sentences):])

    with open(f"cleaned/priors/txts/{basename}.txt", "w") as f:
        f.write("\n".join(f"{sc}\t{fs}\t{ss}" for sc, fs, ss in prior_it))
    with open(f"cleaned/bayesian_shielding/{basename}.txt", "w") as f:
        f.write("\n".join(f"{sc}\t{fs}\t{ss}" for sc, fs, ss in post_it))

if __name__ == "__main__":
    for fname in os.listdir("attacked_documents")[1:]:
        clean_document(os.path.join("attacked_documents",fname))
        del cleaner.context_bert
        cleaner.context_bert = None