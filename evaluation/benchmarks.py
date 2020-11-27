import sys
import os
import multiprocessing
from tqdm import tqdm
import csv
sys.path.append(os.path.realpath(os.path.abspath("../..")))
sys.path.append(os.path.realpath(os.path.abspath("../../Adversarial_Misspellings/defenses/scRNN")))
from Adversarial_Misspellings.defenses.scRNN.corrector import ScRNNChecker
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
sys.path.append("../bayesian_shielding")
from util.utility import read_labeled_data
import warnings 
warnings.filterwarnings("ignore") 

def clean_all_documents(attacked_docs, clean_func):
    docs = os.listdir(attacked_docs)
    for doc in tqdm(docs):
        basename = os.path.basename(os.path.join(attacked_docs,doc)).split(".")[0]
        func_name = clean_func("", name=True)
        doc_path = f"cleaned/{func_name}/{basename}.txt"
        if os.path.isfile(doc_path):
            continue
        scores, first_sentence, secound_sentence = read_labeled_data(os.path.join(attacked_docs,doc))
        sentences = first_sentence + secound_sentence
        clean_sentences = list(map(clean_func, tqdm(sentences)))
        clean_data = zip(scores, clean_sentences[:len(first_sentence)], clean_sentences[len(first_sentence):])
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        with open(doc_path,"w", encoding="utf-8") as f:
            f.write("\n".join(f"{sc}\t{fs}\t{ss}" for sc, fs, ss in clean_data))
        

def clean_with_Adversarial_Misspellings(sentence, name=False):
    if name:
        return "Adversarial_Misspellings"
    checker = ScRNNChecker()
    clean_string = checker.correct_string(sentence)
    return clean_string

def clean_with_pyspellchecker(sentence, name=False):
    if name:
        return "pyspellchecker"
    checker = SpellChecker()
    sentence = sentence.lower()
    tokens = sentence.split(" ")
    misspelled = checker.unknown(tokens)
    for word in misspelled:
        index = tokens.index(word)
        right_word = checker.correction(word)
        tokens[index] = right_word
    out_sentence = " ".join(tokens)
    return out_sentence 


def make_attacked_data_to_tsv(filename, output):
    score, f_s, s_s = read_labeled_data(filename)
    sentences = f_s + s_s
    with open(f'{output}.tsv', 'w', encoding="utf-8") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['index', 'sentence'])
        for sent, i in zip(sentences, range(len(sentences))):
            tsv_writer.writerow([str(i), sent])

    


if __name__ == "__main__":
    clean_all_documents("attacked_documents", clean_with_Adversarial_Misspellings)
    #make_attacked_data_to_tsv("attacked_documents/all_attacks.txt","all_attacks")
    clean_all_documents("attacked_documents", clean_with_pyspellchecker)
