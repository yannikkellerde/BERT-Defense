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
from benchmark_tasks.MNLI.MNLI_handler import get_mnli_accuracy,get_mnli_model
from util.utility import cosine_similarity,read_labeled_data
import warnings 
warnings.filterwarnings("ignore")

roberta = init_model_roberta()
mnli_tokenizer,mnli_model = get_mnli_model()
def eval_document(ground_truth, cleaned):
    mnli = "mnli" in cleaned
    tscores,tfirst_sentences,tsecond_sentences = read_labeled_data(ground_truth,do_float=not mnli)
    tall = tfirst_sentences + tsecond_sentences
    cscores,cfirst_sentences,csecond_sentences = read_labeled_data(cleaned,do_float=not mnli)
    call = cfirst_sentences + csecond_sentences
    evals = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[],"editdistance":[]}
    if mnli:
        evals["mnli"] = get_mnli_accuracy(mnli_tokenizer,mnli_model,cfirst_sentences,csecond_sentences,cscores)
    else:
        posterior_embeddings = simple_sentence_embedder(roberta,call)
        l = len(posterior_embeddings)
        cosine_sims = [cosine_similarity(x,y) for x,y in zip(posterior_embeddings[:int(l/2)],posterior_embeddings[int(l/2):])]
        evals["sts-b"] = stats.spearmanr(cosine_sims,cscores)[0]
    for t,c in tqdm(zip(tall,call)):
        evals["bleu"].append(metrics.bleu_score(t,c))
        all_rouge = metrics.rouge_score(t,c)
        evals["editdistance"].append(metrics.edit_distance(t,c,lower=True))
        evals["rouge-1"].append(all_rouge["rouge-1"]["p"])
        evals["rouge-4"].append(all_rouge["rouge-4"]["p"])
        evals["rouge-l"].append(all_rouge["rouge-l"]["p"])
        evals["rouge-w"].append(all_rouge["rouge-w"]["p"])
        evals["mover"].append(metrics.mover_score(t,c)[0])
    evals["editdistance"] = sum(evals["editdistance"])/len(evals["editdistance"])
    evals["bleu"] = sum(evals["bleu"])/len(evals["bleu"])
    evals["rouge-1"] = sum(evals["rouge-1"])/len(evals["rouge-1"])
    evals["rouge-4"] = sum(evals["rouge-4"])/len(evals["rouge-4"])
    evals["rouge-l"] = sum(evals["rouge-l"])/len(evals["rouge-l"])
    evals["rouge-w"] = sum(evals["rouge-w"])/len(evals["rouge-w"])
    evals["mover"] = sum(evals["mover"])/len(evals["mover"])
    return evals

def eval_many(ground_truth,documents,outfile):
    if os.path.isfile(outfile):
        df = pd.read_csv(outfile)
        data = df.to_dict("records")
    else:
        data = []
    #print(df["document"])
    for doc in tqdm(documents):
        if doc in df["document"].to_numpy():
            print("skipping",doc)
            continue
        print("starting",doc)
        ev = eval_document(ground_truth,doc)
        ev["document"] = doc
        data.append(ev)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="document",keep="last")
    df = df[["document","bleu","mover","sts-b","rouge-1","rouge-4","rouge-l","rouge-w","editdistance","mnli"]]
    df.to_csv(outfile)

if __name__ == "__main__":
    ground_truth = "../bayesian_shielding/benchmark_tasks/STSB/400_sentences.csv"
    documents = []
    """documents = [os.path.join("attacked_documents",x) for x in os.listdir("attacked_documents")]
    documents.extend([os.path.join("cleaned/priors/txts",x) for x in os.listdir("cleaned/priors/txts")])
    documents.extend([os.path.join("cleaned/bayesian_shielding",x) for x in os.listdir("cleaned/bayesian_shielding")])
    documents.extend([os.path.join("cleaned/nocheap_bayesian_shielding",x) for x in os.listdir("cleaned/nocheap_bayesian_shielding")])
    documents.extend([os.path.join("cleaned/nocheap_priors/txts",x) for x in os.listdir("cleaned/nocheap_priors/txts")])
    documents.extend([os.path.join("cleaned/pyspellchecker",x) for x in os.listdir("cleaned/pyspellchecker")])
    documents.extend([os.path.join("cleaned/Adversarial_Misspellings",x) for x in os.listdir("cleaned/Adversarial_Misspellings")])"""
    documents.extend([os.path.join("cleaned_mnli/priors/txts",x) for x in os.listdir("cleaned_mnli/priors/txts")])
    documents.extend([os.path.join("cleaned_mnli/bayesian_shielding",x) for x in os.listdir("cleaned_mnli/bayesian_shielding")])
    documents.extend([os.path.join("cleaned_mnli/nocheap_bayesian_shielding",x) for x in os.listdir("cleaned_mnli/nocheap_bayesian_shielding")])
    documents.extend([os.path.join("cleaned_mnli/nocheap_priors/txts",x) for x in os.listdir("cleaned_mnli/nocheap_priors/txts")])
    documents.extend([os.path.join("cleaned_mnli/pyspellchecker",x) for x in os.listdir("cleaned_mnli/pyspellchecker")])
    documents.extend([os.path.join("cleaned_mnli/Adversarial_Misspellings",x) for x in os.listdir("cleaned_mnli/Adversarial_Misspellings")])
    documents.extend([os.path.join("attacked_mnli",x) for x in os.listdir("attacked_mnli")])
    ground_truth = "../bayesian_shielding/benchmark_tasks/MNLI/mnli_dataset.csv"
    eval_many(ground_truth,documents,"evaluation.csv")