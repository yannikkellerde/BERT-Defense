import sys
import os 
sys.path.append("..")
sys.path.append("../bayesian_shielding")
from benchmark_tasks.STSB.RoBERTa_handler import init_model_roberta,simple_sentence_embedder
from bayesian_shielding.util.utility import read_labeled_data,cosine_similarity
from adversarial_attacks.attack_api import Adversarial_attacker
from frontend.clean_vs_segmentation import clean_sentence
import scipy.stats
from nltk import tokenize
from nltk.translate.bleu_score import sentence_bleu
from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict
import numpy as np
import rouge

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


def sts_b_spearmans_rank(datafile, attackfunc, cleanfunc):
    model = init_model_roberta()
    print("Read sentence in")
    scores, first_sentences, second_sentences = read_labeled_data(datafile)
    print("Start attack")
    first_sentences_atk = [attackfunc(x) for x in first_sentences]
    second_sentences_atk = [attackfunc(x) for x in second_sentences]
    print("Start clean")
    first_sentences_clean = [cleanfunc(x) for x in first_sentences_atk]
    second_sentences_clean = [cleanfunc(x) for x in second_sentences_atk]
    print("Start embedding")
    embed_first_sentences_clean = simple_sentence_embedder(model,first_sentences_clean)
    embed_second_sentences_clean = simple_sentence_embedder(model,second_sentences_clean)

    embed_first_sentences_atk = simple_sentence_embedder(model, first_sentences_atk)
    embed_secound_sentences_atk = simple_sentence_embedder(model, second_sentences_atk)
    
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


    


if __name__ == '__main__':
    attack = Adversarial_attacker()
    attacks_with_severity = [(x,0.1) for x in attack.methods]
    sts_b_spearmans_rank("test_senteces.txt",lambda sentence: attack.multiattack(sentence, attacks_with_severity), clean_sentence)
    # print(mover_score('they are now equipped with air conditioning and nice toilets.','they have air conditioning and new toilets.'))
