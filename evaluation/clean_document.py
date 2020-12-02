import os,sys
from tqdm import tqdm,trange
import pickle
import pandas as pd
import shutil
import random
sys.path.append("../bayesian_shielding")
from frontend.clean_sentences import Sentence_cleaner
from util.utility import read_labeled_data,get_most_likely_sentence_multidics

cleaner = Sentence_cleaner()
def clean_document(infile,use_existing_priors=True,cheap_actions=True,batch_size=128):
    if "mnli" in infile:
        mnli = True
        prefix = "cleaned_mnli"
    else:
        mnli = False
        prefix = "cleaned"
    basename = os.path.basename(infile).split(".")[0]
    scores,first_sentences, second_sentences = read_labeled_data(infile,do_float=not mnli)
    sentences = first_sentences + second_sentences
    pkl_path = f"{prefix}/{'' if cheap_actions else 'nocheap_'}priors/pkls/{basename}"
    if not os.path.isdir(pkl_path) or not use_existing_priors:
        os.makedirs(pkl_path,exist_ok=True)
        if cheap_actions:
            cleaner.dist_handler.cheap_actions = {key:True for key in cleaner.dist_handler.cheap_actions}
        else:
            cleaner.dist_handler.cheap_actions = {key:False for key in cleaner.dist_handler.cheap_actions}
        cleaner.get_priors(sentences,store_path=pkl_path)
    prior = []
    for filename in sorted(os.listdir(pkl_path),key=lambda x:int(x.split(".")[0])):
        with open(os.path.join(pkl_path, filename),"rb") as f:
            prior += pickle.load(f)
    prior_cleaned = [get_most_likely_sentence_multidics([b[0] for b in p[0][1]],[b[1] for b in p[0][1]]) for p in prior]

    prior_it = zip(scores,prior_cleaned[:len(first_sentences)],prior_cleaned[len(first_sentences):])
    txt_path = f"{prefix}/{'' if cheap_actions else 'nocheap_'}priors/txts/{basename}.txt"
    with open(txt_path,"w") as f:
        f.write("\n".join(f"{sc}\t{fs}\t{ss}" for sc, fs, ss in prior_it))

    print("hyp_count",sum(len(x) for x in prior))
    prior_batches = []
    count = 0
    cur = []
    for p in prior:
        count += len(p)
        if count >= batch_size:
            prior_batches.append(cur)
            cur = []
            count = 0
        cur.append(p)
    if len(cur)>0:
        prior_batches.append(cur)
    print("prior refs",sys.getrefcount(prior),len(prior_batches))
    del prior
    os.makedirs("/tmp/adv_shield",exist_ok=True)
    for i,batch in enumerate(prior_batches):
        with open(f"/tmp/adv_shield/{i}.pkl","wb") as f:
            pickle.dump(batch,f)
    del prior_batches

    post_cleaned = []
    for fname in tqdm(sorted(os.listdir("/tmp/adv_shield"),key=lambda x:int(x.split(".")[0]))):
        with open(os.path.join("/tmp/adv_shield",fname),"rb") as f:
            prior_batch = pickle.load(f)
        post_cleaned.extend(cleaner.batched_clean_given_prior(prior_batch))
        del prior_batch

    post_it = zip(scores,post_cleaned[:len(first_sentences)],post_cleaned[len(first_sentences):])
    out_path = f"{prefix}/{'' if cheap_actions else 'nocheap_'}bayesian_shielding/{basename}.txt"
    os.makedirs(os.path.dirname(txt_path),exist_ok=True)
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(f"{sc}\t{fs}\t{ss}" for sc, fs, ss in post_it))
    shutil.rmtree("/tmp/adv_shield")

if __name__ == "__main__":
    #for fname in os.listdir("attacked_documents")[2:]:
    #    clean_document(os.path.join("attacked_documents",fname))
    #    del cleaner.context_bert
    #    cleaner.context_bert = None
    clean_document(sys.argv[1],use_existing_priors=True,cheap_actions=sys.argv[2] == "true")
    sys.exit()
