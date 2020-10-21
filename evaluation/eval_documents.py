import os,sys
from tqdm import tqdm,trange
import pickle
import pandas as pd
import random
import numpy as np
from scipy import stats
import metrics
sys.path.append("../bayesian_shielding")
from benchmark_tasks.STSB.RoBERTa_handler import init_model_roberta,simple_sentence_embedder
from util.utility import cosine_similarity,read_labeled_data

roberta = init_model_roberta()
def eval_document(ground_truth, cleaned):
    tscores,tfirst_sentences,tsecond_sentences = read_labeled_data(ground_truth)
    tall = tfirst_sentences + tsecond_sentences
    cscores,cfirst_sentences,csecond_sentences = read_labeled_data(cleaned)
    call = cfirst_sentences + csecond_sentences
    evals = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[]}
    posterior_embeddings = simple_sentence_embedder(roberta,call)
    l = len(posterior_embeddings)
    cosine_sims = [cosine_similarity(x,y) for x,y in zip(posterior_embeddings[:l/2],posterior_embeddings[l/2:])]
    evals["sts-b"] = stats.spearmanr(cosine_sims,cscores)
    for t,c in zip(tall,call):
        evals["bleu"].append(metrics.bleu_score(t,c))
        all_rouge = metrics.rouge_score(t,c)
        evals["rouge-1"].append(all_rouge["rouge-1"]["p"])
        evals["rouge-4"].append(all_rouge["rouge-4"]["p"])
        evals["rouge-l"].append(all_rouge["rouge-l"]["p"])
        evals["rouge-w"].append(all_rouge["rouge-w"]["p"])
        evals["mover"].append(metrics.mover_score(t,c)[0])
    evals["bleu"] = sum(evals["bleu"])
    evals["rouge-4"] = sum(evals["rouge-4"])
    evals["rouge-l"] = sum(evals["rouge-l"])
    evals["rouge-w"] = sum(evals["rouge-w"])
    evals["mover"] = sum(evals["mover"])
    return evals

def eval_many(ground_truth,documents,outfile):
    if os.path.isfile(outfile):
        df = pd.read_csv(outfile)
        data = df.to_dict("records")
    else:
        data = []
    for doc in documents:
        ev = eval_document(ground_truth,doc)
        ev["document"] = doc
        data.append(ev)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="document",keep="last")
    df.to_csv(outfile)