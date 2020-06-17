import logging
# create logger with 'root'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('__main__.log')
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

import numpy as np
from bert_posterior import bert_posterior, format_dict
from edit_distance import get_word_dic_distance
from util import load_dictionary,load_pickle,load_and_preprocess_dataset,cosine_similarity
from letter_stuff import sentence_ends
from multiprocess_tasks import multiprocess_word_distances
from sentence_Embeddings import load_vectors, sentence_embedding_only_best_word,init_model_roberta, sentence_average_from_word_embeddings
import time


if __name__ == '__main__':
    logger.info("loading dictionary, word embedding and dataset")
    letter_dic = load_dictionary("DATA/dictionaries/bert_letter_begin.txt")
    number_dic = load_dictionary("DATA/dictionaries/bert_number_begin.txt")
    punc_dic = load_dictionary("DATA/dictionaries/bert_punctuations.txt")
    # word_vecs = load_vectors("DATA/embeddings/wiki-news-300d-1M-subword.vec")
    dictionary = letter_dic+number_dic+punc_dic
    word_embedding = load_pickle("visual_embeddings.pkl")
    bert_dict = format_dict(dictionary)
    dataset = load_and_preprocess_dataset("DATA/test-scoreboard-dataset.txt")
    in_data = dataset[0:5]
    logger.info(in_data)
    logger.info("Init Sentence Embedding Model")
    model = init_model_roberta()
    logger.info("calculating distances")
    start = time.perf_counter()
    priors = multiprocess_word_distances(in_data,dictionary,word_embedding)
    posterior_sentences = [[[] for _sentence in line] for line in priors]
    embedded_sentences =  [[[] for _sentence in line] for line in priors]
    for l,line in enumerate(priors):
        for s,sentence_prior in enumerate(line):
            prior_sentence = [list(sorted([(y[1],y[0]) for y in x],reverse=True))[0][1] for x in sentence_prior]
            logger.info("Only distance normalized sentence: "+str(prior_sentence))
            if prior_sentence[-1] not in sentence_ends:
                tmp = [(x,1/len(sentence_ends) if x in sentence_ends else 0) for x in dictionary]
                sentence_prior.append(tmp)
            bert_prior = np.array([[x[1] for x in p] for p in sentence_prior])
            logger.info("calculating posterior")
            posterior = bert_posterior(bert_prior,bert_dict,10)
            sent_emb = sentence_embedding_only_best_word(model, posterior, dictionary)
            # sent_emb2 = sentence_average_from_word_embeddings(posterior, dictionary, word_vecs)
            embedded_sentences[l][s] = sent_emb
            out_sentence = [dictionary[np.argmax(p)] for p in posterior]
            logger.info("Posterior sentence"+str(out_sentence))
            posterior_sentences[l][s] = out_sentence
    sim_scores = []
    for line in embedded_sentences:
        sim_scores.append(cosine_similarity(line[0], line[1]))
    logger.info(f"time taken: {time.perf_counter()-start}")

    logger.info("full posterior:")
    logger.info(posterior_sentences)
    logger.info("cosine_similarity")
    logger.info(sim_scores)