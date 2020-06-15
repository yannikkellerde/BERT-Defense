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
from util import load_dictionary,load_pickle,load_and_preprocess_dataset,cosine_similarity
from letter_stuff import sentence_ends
from multiprocess_tasks import multiprocess_word_distances
from sentence_Embeddings import load_vectors, sentence_embedding_only_best_word,init_model_roberta, sentence_average_from_word_embeddings
import time
import os,sys
from tqdm import tqdm,trange


if __name__ == '__main__':
    logger.info("loading dictionary, word embedding and dataset")
    letter_dic = load_dictionary("DATA/dictionaries/bert_letter_begin.txt")
    number_dic = load_dictionary("DATA/dictionaries/bert_number_begin.txt")
    punc_dic = load_dictionary("DATA/dictionaries/bert_punctuations.txt")
    #word_vecs = load_vectors("DATA/embeddings/wiki-news-300d-1M-subword.txt")
    dictionary = letter_dic+number_dic+punc_dic
    word_embedding = load_pickle("visual_embeddings.pkl")
    bert_dict = format_dict(dictionary)
    dataset = load_and_preprocess_dataset("DATA/test-scoreboard-dataset.txt")
    logger.info("Init Sentence Embedding Model")
    model = init_model_roberta()
    if os.path.exists("storage/word_distances.pkl"):
        with open("storage/word_distances.pkl","rb") as f:
            transpo_dict = pkl.load(f)
    else:
        transpo_dict = {}
    if os.path.exists("storage/embeddings_robert.pkl"):
        with open("storage/embeddings_robert.pkl","rb") as f:
            all_embed_robert=pkl.load(f)
    else:
        all_embed_robert = []
    if os.path.exists("storage/sim_scores_robert.txt"):
        with open("storage/sim_scores_robert.txt","r") as f:
            sim_scores_robert = f.read().splitlines()
    else:
        sim_scores_robert = []
    #sim_scores_word = []
    #all_embed_word = []
    for whereami in trange(len(sim_scores_robert),len(dataset),20):
        in_data = dataset[whereami:whereami+20]
        logger.debug("input data:")
        logger.debug(in_data)
        start = time.perf_counter()
        logger.info("calculating distances")
        priors = multiprocess_word_distances(in_data,dictionary,word_embedding,transpo_dict)
        logger.info(f"time distance calculation: {time.perf_counter()-start}")
        start = time.perf_counter()
        posterior_sentences = [[[] for _sentence in line] for line in priors]
        embedded_sentences_robert =  [[[] for _sentence in line] for line in priors]
        #embedded_sentences_word = [[[] for _sentence in line] for line in priors]
        for l,line in enumerate(priors):
            for s,sentence_prior in enumerate(line):
                prior_sentence = [dictionary[np.argmax(p)] for p in sentence_prior]
                logger.debug("Only distance normalized sentence: "+str(prior_sentence))
                if prior_sentence[-1] not in sentence_ends:
                    tmp = np.array([1/len(sentence_ends) if x in sentence_ends else 0 for x in dictionary])
                    sentence_prior.append(tmp)
                logger.debug("calculating posterior")
                posterior = bert_posterior(sentence_prior,bert_dict,int(len(sentence_prior)*1.5))
                sent_emb_robert = sentence_embedding_only_best_word(model, posterior, dictionary)
                #sent_emb_word = sentence_average_from_word_embeddings(posterior, dictionary, word_vecs)
                embedded_sentences_robert[l][s] = sent_emb_robert
                #embedded_sentences_word[l][s] = sent_emb_word
                out_sentence = [dictionary[np.argmax(p)] for p in posterior]
                logger.debug("Posterior sentence"+str(out_sentence))
                posterior_sentences[l][s] = out_sentence
            sim_scores_robert.append(cosine_similarity(embedded_sentences_robert[l][0], embedded_sentences_robert[l][1]))
            #sim_scores_word.append(cosine_similarity(embedded_sentences_word[l][0], embedded_sentences_word[l][1]))
        logger.info(f"time posterior + similarity: {time.perf_counter()-start}")
        start = time.perf_counter()
        all_embed_robert.extend(embedded_sentences_robert)
        #all_embed_word.extend(embedded_sentences_word)

        with open("storage/embeddings_robert.pkl","wb") as f:
            pkl.dump(all_embed_robert,f)
            #f.write("\n".join(["\t".join([" ".join([str(vecnum) for vecnum in sentence]) for sentence in line]) for line in all_embed_robert]))
        #with open("storage/embeddings_word_vec.txt","w") as f:
        #    f.write("\n".join(["\t".join([" ".join(sentence) for sentence in line]) for line in all_embed_word]))
        with open("storage/sim_scores_robert.txt","w") as f:
            f.write("\n".join([str(x) for x in sim_scores_robert]))
        with open("storage/word_distances.pkl","wb") as f:
            pkl.dump(transpo_dict,f)
        #with open("storage/sim_scores_word_vec.txt","w") as f:
        #    f.write("\n".join(sim_scores_word))

        logger.debug("full posterior:")
        logger.debug(posterior_sentences)
        logger.debug("cosine_similarity")
        logger.debug(sim_scores_robert)
        logger.info(f"time writing all the stuff: {time.perf_counter()-start}")