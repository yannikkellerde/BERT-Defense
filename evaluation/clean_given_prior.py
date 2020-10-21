import pickle
import sys,os
sys.path.append("../bayesian_shielding")
from frontend.clean_sentences import Sentence_cleaner

with open("cleaned/priors/pkls/disemvowel.pkl","rb") as f:
    prior = pickle.load(f)

cleaner = Sentence_cleaner()
post_cleaned = cleaner.batched_clean_given_prior(prior)

with open("test.txt","w") as f:
    f.write("\n".join(post_cleaned))