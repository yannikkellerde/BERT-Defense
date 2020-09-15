import logging
import pickle as pkl
# create logger with 'root'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('shared_task.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(file_formatter)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(console_formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
logger.debug("\n\n")

import numpy as np
from bert_posterior import bert_posterior, format_dict
from edit_distance import get_word_dic_distance
from util import load_dictionary,load_pickle,load_and_preprocess_dataset,cosine_similarity,write_dataset,only_read_dataset, combine_known_transpos, get_most_likely_sentence
from letter_stuff import sentence_ends
from multiprocess_tasks import multiprocess_word_distances
from RoBERTa_handler import create_pre_mapping,load_vectors, encode_one_sentence,init_model_roberta, sentence_average_from_word_embeddings
import time
import os,sys
from tqdm import tqdm,trange
from second_roberta import get_similarity


if __name__ == '__main__':
    logger.info("loading dictionary, word embedding and dataset")
    letter_dic = load_dictionary("DATA/dictionaries/bert_letter_begin.txt")
    number_dic = load_dictionary("DATA/dictionaries/bert_number_begin.txt")
    punc_dic = load_dictionary("DATA/dictionaries/bert_punctuations.txt")
    dictionary = letter_dic+number_dic+punc_dic
    word_embedding = load_pickle("visual_embeddings.pkl")
    bert_dict = format_dict(dictionary)
    dataset = load_and_preprocess_dataset("DATA/test-final-dataset.txt")
    logger.info("Init Sentence Embedding Model")
    model = init_model_roberta()
    #if os.path.exists("storage_first_version/word_distances.pkl"):
    #    with open("storage_first_version/word_distances.pkl","rb") as f:
    #        first_version_transpo = pkl.load(f)
    if os.path.exists("storage/word_distances.pkl"):
        with open("storage/word_distances.pkl","rb") as f:
            transpo_dict = pkl.load(f)
    else:
        transpo_dict = {}
    """if os.path.exists("storage/embeddings_robert.pkl"):
        with open("storage/embeddings_robert.pkl","rb") as f:
            all_embed_robert=pkl.load(f)def convert_prior_to_weights_tensor(prior,dictionary):
    weights_tensor = np.zeros(len(prior),len(tokenizer.vocab))
    for i,p in enumerate(prior):
        for j,weight in enumerate(p):
            weights_tensor[i][dictionary[j]] = weight
    return weights_tensor

    else:
        all_embed_robert = []"""
    if os.path.exists("storage/sim_scores_robert.txt"):
        with open("storage/sim_scores_robert.txt","r") as f:
            sim_scores_robert = f.read().splitlines()
    else:
        sim_scores_robert = []
    if os.path.exists("storage/cleaned_dataset.txt"):
        all_cleaned_dataset = only_read_dataset("storage/cleaned_dataset.txt")
    else:
        all_cleaned_dataset = []
    dataset = dataset[:1800]
    for whereami in trange(len(sim_scores_robert),len(dataset),20):
        in_data = dataset[whereami:whereami+20]
        logger.debug("input data:")
        logger.debug(in_data)
        start = time.perf_counter()
        logger.info("calculating distances")
        priors = multiprocess_word_distances(in_data,word_embedding,transpo_dict)
        #priors = combine_known_transpos(in_data,transpo_dict,first_version_transpo)
        logger.info(f"time distance calculation: {time.perf_counter()-start}")
        start = time.perf_counter()
        posterior_sentences = [[[] for _sentence in line] for line in priors]
        embedded_sentences_robert =  [[[] for _sentence in line] for line in priors]
        cleaned_dataset =  [[[] for _sentence in line] for line in priors]
        for l,line in enumerate(priors):
            for s,sentence_prior in enumerate(line):
                prior_sentence = get_most_likely_sentence(sentence_prior,dictionary)
                logger.debug("Only distance normalized sentence: "+str(prior_sentence))
                if prior_sentence[-1] not in sentence_ends:
                    tmp = [np.array([1/len(sentence_ends) if x in sentence_ends else 0 for x in dictionary]),tuple()]
                    sentence_prior.append(tmp)
                if s>0:
                    for i,word in enumerate(in_data[l][s]):
                        if word in pre_map:
                            sentence_prior[i] = pre_map[word]
                logger.debug("calculating posterior")
                posterior = bert_posterior(sentence_prior,bert_dict,int(len(sentence_prior)*1.5))
                if s == 0:
                    pre_map = create_pre_mapping(posterior,in_data[l][s],dictionary)
                out_sentence = get_most_likely_sentence(posterior,dictionary)
                sent_emb_robert = encode_one_sentence(model, out_sentence)
                embedded_sentences_robert[l][s] = sent_emb_robert
                cleaned_dataset[l][s] = out_sentence
                logger.debug("Posterior sentence"+str(out_sentence))
                posterior_sentences[l][s] = out_sentence
            sim_scores_robert.append(cosine_similarity(embedded_sentences_robert[l][0], embedded_sentences_robert[l][1]))
        #sim_scores_robert.extend(get_similarity(cleaned_dataset))
        logger.info(f"time posterior + similarity: {time.perf_counter()-start}")
        start = time.perf_counter()
        #all_embed_robert.extend(embedded_sentences_robert)
        all_cleaned_dataset.extend(cleaned_dataset)

        write_dataset("storage/cleaned_dataset.txt",all_cleaned_dataset,as_sentences=True)
        #with open("storage/embeddings_robert.pkl","wb") as f:
        #    pkl.dump(all_embed_robert,f)
        with open("storage/sim_scores_robert.txt","w") as f:
            f.write("\n".join([str(x) for x in sim_scores_robert]))
        with open("storage/word_distances.pkl","wb") as f:
            pkl.dump(transpo_dict,f)
        logger.debug("full posterior:")
        logger.debug(posterior_sentences)
        logger.debug("cosine_similarity")
        logger.debug(sim_scores_robert)
        logger.info(f"time writing all the stuff: {time.perf_counter()-start}")