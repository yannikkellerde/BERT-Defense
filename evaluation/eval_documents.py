import os,sys
from tqdm import tqdm,trange
import pickle
import csv
import pandas as pd
import random
import numpy as np
from scipy import stats
import metrics
sys.path.append("../bayesian_shielding")
#from benchmark_tasks.STSB.RoBERTa_handler import init_model_roberta,simple_sentence_embedder
#from benchmark_tasks.MNLI.MNLI_handler import get_mnli_accuracy,get_mnli_model
from util.utility import cosine_similarity,read_labeled_data
import warnings 
warnings.filterwarnings("ignore")

#roberta = init_model_roberta()
#mnli_tokenizer,mnli_model = get_mnli_model()

def ttest_documents(ground_truth,d1,d2):
    _,tfirst_sentences,tsecond_sentences = read_labeled_data(ground_truth,do_float=True)
    tall = tfirst_sentences + tsecond_sentences
    _,cfirst_sentences,csecond_sentences = read_labeled_data(d1,do_float=True)
    d1all = cfirst_sentences + csecond_sentences
    _,cfirst_sentences,csecond_sentences = read_labeled_data(d2,do_float=True)
    d2all = cfirst_sentences + csecond_sentences

    ev1 = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[],"editdistance":[]}
    make_easy_measures(ev1,tall,d1all,average=False)
    ev2 = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[],"editdistance":[]}
    make_easy_measures(ev2,tall,d2all,average=False)

    res = {}
    for key in ev1:
        res[key] = stats.ttest_ind(ev1[key],ev2[key])
    return res

def ttest_vs_human(ground_truth,document,condition):
    ev1 = eval_document_40(ground_truth,document,avg=False)
    ev2 = evaluate_human(condition,avg=False)
    res = {}
    for key in ev1:
        print(key,np.mean(ev1[key]),np.mean(ev2[key]),np.std(ev1[key]),np.std(ev2[key]))
        res[key] = stats.ttest_ind(ev1[key],ev2[key])
    return res

def make_easy_measures(evals,tall,call,average=True):
    for t,c in tqdm(zip(tall,call)):
        #if t.endswith("."):
        #    t = t[:-1]
        #if c.endswith("."):
        #    c = c[:-1]
        evals["bleu"].append(metrics.bleu_score(t,c))
        all_rouge = metrics.rouge_score(t,c)
        evals["editdistance"].append(metrics.edit_distance(t,c,lower=True))
        evals["rouge-1"].append(all_rouge["rouge-1"]["p"])
        evals["rouge-4"].append(all_rouge["rouge-4"]["p"])
        evals["rouge-l"].append(all_rouge["rouge-l"]["p"])
        evals["rouge-w"].append(all_rouge["rouge-w"]["p"])
        evals["mover"].append(metrics.mover_score(t,c)[0])
    evals["perfect_cleaned"] = (np.array(evals["bleu"])==1).astype(np.float)
    if average:
        evals["perfect_cleaned"] = sum(evals["perfect_cleaned"])/len(evals["perfect_cleaned"])
        evals["editdistance"] = sum(evals["editdistance"])/len(evals["editdistance"])
        evals["bleu"] = sum(evals["bleu"])/len(evals["bleu"])
        evals["rouge-1"] = sum(evals["rouge-1"])/len(evals["rouge-1"])
        evals["rouge-4"] = sum(evals["rouge-4"])/len(evals["rouge-4"])
        evals["rouge-l"] = sum(evals["rouge-l"])/len(evals["rouge-l"])
        evals["rouge-w"] = sum(evals["rouge-w"])/len(evals["rouge-w"])
        evals["mover"] = sum(evals["mover"])/len(evals["mover"])

def compare_with_human(my_folder,avg=True):
    ground_truth_path = "../bayesian_shielding/benchmark_tasks/STSB/400_sentences.csv"
    human_data = pd.read_csv("human_data/eval_data.csv",sep="\t",quoting=csv.QUOTE_NONE)
    gt_scores,gt_first,gt_second = read_labeled_data(ground_truth_path)
    attacks = human_data["attack"].unique().tolist()
    attacks.remove("correct")
    tall = []
    call = []
    machine_clean = {}
    for attack in attacks:
        truth_to_clean = {}
        at_scores,at_first,at_second = read_labeled_data(os.path.join(my_folder,attack+".txt"))
        for i in range(len(at_first)):
            truth_to_clean[gt_first[i][:-1].lower() if gt_first[i].endswith(".") else gt_first[i].lower()] = at_first[i]
            truth_to_clean[gt_second[i][:-1].lower() if gt_second[i].endswith(".") else gt_second[i].lower()] = at_second[i]
        machine_clean[attack] = truth_to_clean
    for index,row in human_data.iterrows():
        if row["attack"]=="correct":
            continue
        call.append(row["response"])
        tall.append(machine_clean[row["attack"]][row["cleanSent"][:-1] if row["cleanSent"].endswith(".") else row["cleanSent"]])
    evals = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[],"editdistance":[]}
    make_easy_measures(evals,tall,call,average=avg)
    return evals

def compare_all_with_human():
    folder_map = {"attacked_documents":"no cleaning","cleaned/Adversarial_Misspellings":"danishpruthi","cleaned/bayesian_shielding":"ours fp",
                  "cleaned/nocheap_bayesian_shielding":"ours bp","cleaned/pyspellchecker":"pyspellchecker"}
    all_evals = []
    for folder in folder_map:
        ev = compare_with_human(folder)
        ev["method"] = folder_map[folder]
        all_evals.append(ev)
    df = pd.DataFrame(all_evals)
    df.to_csv("compare_with_human.csv")

def evaluate_human(condition,avg=True):
    data = pd.read_csv("human_data/eval_data.csv",sep="\t")
    data = data[data["attack"]==condition]
    evals = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[],"editdistance":[]}
    make_easy_measures(evals,data["cleanSent"],data["response"],average=avg)
    return evals

def evaluate_full_human(outfile):
    if os.path.isfile(outfile):
        df = pd.read_csv(outfile)
        data = df.to_dict("records")
    else:
        data = []
    name_map = {"visual":"vi","phonetic":"ph","full-swap":"fs","inner-swap":"is","disemvowel":"dv","truncate":"tr",
                "keyboard-typo":"kt","natural-typo":"nt","intrude":"in","segmentation":"sg","rand":"rd"}
    cond_map = {"natural_typo":"nt:0.3","phonetic":"ph:0.3","full_swap":"fs:0.3","rand2":"rd:0.3;rd:0.3",
                "disemvowel":"dv:0.3","rand_hard":"rd:0.6;rd:0.6","visual":"vi:0.3","phonetic_70":"ph:0.7",
                "segment_typo":"sg:0.5;kt:0.3"}
    for cond in tqdm(cond_map):
        ev = evaluate_human(cond)
        ev["document"] = "human/"+cond_map[cond]
        data.append(ev)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="document",keep="last")
    df = df[["document","bleu","mover","sts-b","rouge-1","rouge-4","rouge-l","rouge-w","editdistance","perfect_cleaned"]]
    df.to_csv(outfile)

def eval_document_40(ground_truth, cleaned,avg=True):
    tscores,tfirst_sentences,tsecond_sentences = read_labeled_data(ground_truth,do_float=True)
    tall = tfirst_sentences + tsecond_sentences
    cscores,cfirst_sentences,csecond_sentences = read_labeled_data(cleaned,do_float=True)
    call = cfirst_sentences + csecond_sentences
    human_data = pd.read_csv("human_data/eval_data.csv",sep="\t",quoting=csv.QUOTE_NONE)
    delinds = []
    #print("russia, china veto un resolution on syria endorsing" in list(human_data["cleanSent"]))
    #exit()
    at_set = cleaned.split("/")[-1].split(".")[0]
    my_important_sent = list(human_data[human_data["attack"]==at_set]["cleanSent"])
    if len(my_important_sent) < 10:
        return None
    for i in range(len(tall)):
        if tall[i].endswith("."):
            tall[i] = tall[i][:-1]
        tall[i] = tall[i].lower()
        if tall[i] not in my_important_sent or tall[i] in tall[:i]:
            delinds.append(i)
    for i in reversed(delinds):
        del tall[i]
        del call[i]
    print(len(tall))
    #assert len(tall) == 40
    evals = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[],"editdistance":[]}
    make_easy_measures(evals,tall,call,average=avg)
    return evals

def eval_many_40(ground_truth,documents,outfile):
    if os.path.isfile(outfile):
        df = pd.read_csv(outfile)
        data = df.to_dict("records")
    else:
        data = []
    #print(df["document"])
    for doc in tqdm(documents):
        doc_store = doc+":40"
        #if doc_store in df["document"].to_numpy():
        #    print("skipping",doc_store)
        #    continue
        print("starting",doc_store)
        ev = eval_document_40(ground_truth,doc)
        if ev is None:
            continue
        ev["document"] = doc_store
        data.append(ev)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="document",keep="last")
    df = df[["document","bleu","mover","sts-b","rouge-1","rouge-4","rouge-l","rouge-w","editdistance","perfect_cleaned"]]
    df.to_csv(outfile)

def eval_document(ground_truth, cleaned):
    tscores,tfirst_sentences,tsecond_sentences = read_labeled_data(ground_truth,do_float=True)
    tall = tfirst_sentences + tsecond_sentences
    cscores,cfirst_sentences,csecond_sentences = read_labeled_data(cleaned,do_float=True)
    call = cfirst_sentences + csecond_sentences
    evals = {"bleu":[],"mover":[],"rouge-1":[],"rouge-4":[],"rouge-l":[],"rouge-w":[],"editdistance":[]}
    #posterior_embeddings = simple_sentence_embedder(roberta,call)
    #l = len(posterior_embeddings)
    #cosine_sims = [cosine_similarity(x,y) for x,y in zip(posterior_embeddings[:int(l/2)],posterior_embeddings[int(l/2):])]
    #evals["sts-b"] = stats.spearmanr(cosine_sims,cscores)[0]
    make_easy_measures(evals,tall,call)
    return evals

def eval_many(ground_truth,documents,outfile):
    if os.path.isfile(outfile):
        df = pd.read_csv(outfile)
        data = df.to_dict("records")
    else:
        data = []
    #print(df["document"])
    for doc in tqdm(documents):
        #if doc in df["document"].to_numpy():
        #    print("skipping",doc)
        #    continue
        print("starting",doc)
        ev = eval_document(ground_truth,doc)
        ev["document"] = doc
        data.append(ev)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset="document",keep="last")
    df = df[["document","bleu","mover","sts-b","rouge-1","rouge-4","rouge-l","rouge-w","editdistance","perfect_cleaned"]]
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
    documents.extend([os.path.join("cleaned/no_bert_bayesian_shielding",x) for x in os.listdir("cleaned/no_bert_bayesian_shielding")])
    documents.extend([os.path.join("cleaned/no_lev_bayesian_shielding",x) for x in os.listdir("cleaned/no_lev_bayesian_shielding")])
    documents.extend([os.path.join("cleaned/no_gpt_bayesian_shielding",x) for x in os.listdir("cleaned/no_gpt_bayesian_shielding")])
    """
    documents.extend([os.path.join("cleaned_mnli/priors/txts",x) for x in os.listdir("cleaned_mnli/priors/txts")])
    documents.extend([os.path.join("cleaned_mnli/bayesian_shielding",x) for x in os.listdir("cleaned_mnli/bayesian_shielding")])
    documents.extend([os.path.join("cleaned_mnli/nocheap_bayesian_shielding",x) for x in os.listdir("cleaned_mnli/nocheap_bayesian_shielding")])
    documents.extend([os.path.join("cleaned_mnli/nocheap_priors/txts",x) for x in os.listdir("cleaned_mnli/nocheap_priors/txts")])
    documents.extend([os.path.join("cleaned_mnli/pyspellchecker",x) for x in os.listdir("cleaned_mnli/pyspellchecker")])
    documents.extend([os.path.join("cleaned_mnli/Adversarial_Misspellings",x) for x in os.listdir("cleaned_mnli/Adversarial_Misspellings")])
    documents.extend([os.path.join("attacked_mnli",x) for x in os.listdir("attacked_mnli")])
    ground_truth = "../bayesian_shielding/benchmark_tasks/STSB/400_sentences.csv"""
    eval_many(ground_truth,documents,"evaluation.csv")
    #print(eval_document(ground_truth,"cleaned/nocheap_bayesian_shielding/rand2.txt"))
    #eval_many_40(ground_truth,documents,"evaluation.csv")
    #evaluate_full_human("evaluation.csv")
    #compare_all_with_human()
    #print(compare_with_human("cleaned/bayesian_shielding"))
    #print(ttest_documents(ground_truth,"cleaned/nocheap_bayesian_shielding/rand_hard.txt","cleaned/pyspellchecker/rand_hard.txt"))
    #print(ttest_vs_human(ground_truth,"cleaned/bayesian_shielding/rand2.txt","rand2"))
