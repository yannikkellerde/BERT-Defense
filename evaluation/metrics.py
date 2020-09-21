import sys
import os 
sys.path.append("..")
sys.path.append("../bayesian_shielding")
from benchmark_tasks.STSB.RoBERTa_handler import init_model_roberta,simple_sentence_embedder
from bayesian_shielding.util.utility import read_labeled_data,cosine_similarity, only_read_dataset
from adversarial_attacks.attack_api import Adversarial_attacker
from frontend.clean_sentences import clean_sentence, clean_sentence_init
import scipy.stats
from nltk import tokenize
from nltk.translate.bleu_score import sentence_bleu
from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict
import numpy as np
import rouge
import pandas as pd
from tqdm import tqdm

def bleu_score(reference, sentence, blue_num=4):
    blue_array = np.ones(blue_num) / blue_num
    reference = [tokenize.word_tokenize(reference)]
    sentence = tokenize.word_tokenize(sentence)
    return sentence_bleu(reference, sentence, weights=blue_array)


def rouge_score(reference, sentence, metrics=["rouge-n","rouge-l","rouge-w"], n=4, stemming=True):
    evaluator = rouge.Rouge(metrics=metrics, max_n=n, stemming=stemming)
    return evaluator.get_scores(sentence, reference)


def mover_score(reference, sentence):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    return word_mover_score([reference], [sentence], idf_dict_ref, idf_dict_hyp)


def direct_evaluation(datafile, attackfunc, cleanfunc):
    sentences = only_read_dataset(datafile)
    sentences = list(map(lambda x: x[0].lower(), sentences))
    sentences_atk, sentences_clean = clean_atk_sentences(sentences, attackfunc, cleanfunc)
    mover_score_atk = [mover_score(ref, sent) for ref,sent in zip(sentences, sentences_atk)]
    mover_score_clean = [mover_score(ref, sent) for ref,sent in zip(sentences, sentences_clean)]
    bleu_score_atk = [bleu_score(ref, sent) for ref,sent in zip(sentences, sentences_atk)]
    bleu_score_clean = [bleu_score(ref, sent) for ref,sent in zip(sentences, sentences_clean)]
    rouge_1_atk, rouge_4_atk, rouge_l_atk, rouge_w_atk = _get_rouge_scores(sentences, sentences_atk)
    rouge_1_clean, rouge_4_clean, rouge_l_clean, rouge_w_clean = _get_rouge_scores(sentences, sentences_atk)
    perfekt_sentences_atk = len(list(filter(lambda x: x == 1, bleu_score_atk)))
    perfekt_sentences_clean = len(list(filter(lambda x: x == 1, bleu_score_clean)))
    atk_means = [perfekt_sentences_atk]
    for scores in [mover_score_atk, bleu_score_atk, rouge_1_atk, rouge_4_atk, rouge_l_atk, rouge_w_atk]:
        atk_means.append(np.mean(scores))
    clean_means = [perfekt_sentences_clean]
    for scores in [mover_score_clean, bleu_score_clean, rouge_1_clean, rouge_4_clean, rouge_l_clean, rouge_w_clean]:
        clean_means.append(np.mean(scores))
    df = pd.DataFrame(zip(atk_means,clean_means),columns=["Attacked Sentence", "Cleaned Sentences"],
                      index=["right sentences",'mover_score','bleu_score', 'rouge_1', 'rouge_4', 'rouge_l', 'rouge_w'])
    return df
    


    
def _get_rouge_scores(references, sentences):
    rouge_4 = []
    rouge_1 = []
    rouge_l = []
    rouge_w = []
    for ref,sent in zip(references, sentences):
        all_scores = rouge_score(ref,sent)
        rouge_4.append(all_scores['rouge-4']['p'])
        rouge_1.append(all_scores['rouge-1']['p'])
        rouge_l.append(all_scores['rouge-l']['p'])
        rouge_w.append(all_scores['rouge-w']['p'])
    return rouge_1, rouge_4, rouge_l, rouge_w

def sts_b_spearmans_rank(datafile, attackfunc, cleanfunc):
    model = init_model_roberta()
    print("Read sentence in")
    scores, first_sentences, second_sentences = read_labeled_data(datafile)
    print("process sentences")
    first_sentences_atk, first_sentences_clean = clean_atk_sentences(first_sentences, attackfunc, cleanfunc)
    second_sentences_atk, second_sentences_clean = clean_atk_sentences(second_sentences, attackfunc, cleanfunc)
    print("Start embedding")
    embed_first_sentences_clean = simple_sentence_embedder(model,first_sentences_clean)
    embed_second_sentences_clean = simple_sentence_embedder(model,second_sentences_clean)

    embed_first_sentences_atk = simple_sentence_embedder(model, first_sentences_atk)
    embed_second_sentences_atk = simple_sentence_embedder(model, second_sentences_atk)
    
    embed_first_sentences = simple_sentence_embedder(model, first_sentences)
    embed_second_sentences = simple_sentence_embedder(model, second_sentences)
    print("calculate cosine")
    cosine_sims = [cosine_similarity(x,y) for x,y in zip(embed_first_sentences,embed_second_sentences)]
    cosine_sims_atk = [cosine_similarity(x,y) for x,y in zip(embed_first_sentences_atk,embed_second_sentences_atk)]
    cosine_sims_clean = [cosine_similarity(x,y) for x,y in zip(embed_first_sentences_clean,embed_second_sentences_clean)]
    print("Calculate spearmans rank")
    spearman = [scipy.stats.spearmanr(scores,np.clip(x, 0.0, 1.0)) for x in [cosine_sims, cosine_sims_atk, cosine_sims_clean]]
    print(f"Real rank: {spearman[0]}, Atk rank: {spearman[1]}, Clean rank {spearman[2]}")
    return spearman

def clean_atk_sentences(sentences, attackfunc, cleanfunc):
    sentences_atk = [attackfunc(x) for x in tqdm(sentences)]
    sentences_clean = [cleanfunc(x) for x in tqdm(sentences_atk)]
    return sentences_atk, sentences_clean
    


if __name__ == '__main__':
    
    attack = Adversarial_attacker()
    attacks_with_severity = [(x,0.1) for x in ['keyboard-typo','natural-typo']]
    context_bert, dist_handler= clean_sentence_init()
    
    spearman = sts_b_spearmans_rank("test_senteces.txt",lambda sentence: attack.multiattack(sentence, attacks_with_severity), 
                                    lambda sentence: clean_sentence(sentence, context_bert=context_bert, dist_handler=dist_handler))
    
    data= direct_evaluation("sts-b-sentences_short.txt",lambda sentence: attack.multiattack(sentence, attacks_with_severity), 
                            lambda sentence: clean_sentence(sentence, context_bert=context_bert, dist_handler=dist_handler))
    print(f"Real rank: {spearman[0]}, Atk rank: {spearman[1]}, Clean rank {spearman[2]}")
    print(data)
    #print(bleu_score('My name is Jan M','My name is Jan'))
