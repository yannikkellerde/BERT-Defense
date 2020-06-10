import numpy as np
from bert_posterior import bert_posterior, format_dict
from edit_distance import get_word_dic_distance
from util import load_dictionary,load_pickle,load_and_preprocess_dataset
from letter_stuff import sentence_ends
from util import multiprocess_word_distances

if __name__ == '__main__':
    print("loading dictionary, word embedding and dataset")
    dictionary = load_dictionary("DATA/bert_wiki_full_words.txt")
    word_embedding = load_pickle("visual_embeddings.pkl")
    bert_dict = format_dict(dictionary)
    dataset = load_and_preprocess_dataset("DATA/test-scoreboard-dataset.txt")
    in_data = dataset[2:4]
    print(in_data)
    print("calculating distances")
    priors = multiprocess_word_distances(in_data,dictionary,word_embedding)
    posterior_sentences = [[[] for _sentence in line] for line in priors]
    for l,line in enumerate(priors):
        for s,sentence_prior in enumerate(line):
            prior_sentence = [list(sorted([(y[1],y[0]) for y in x],reverse=True))[0][1] for x in sentence_prior]
            print("Only distance normalized sentence: ",prior_sentence)
            if prior_sentence[-1] not in sentence_ends:
                tmp = [(x,1/len(sentence_ends) if x in sentence_ends else 0) for x in dictionary]
                sentence_prior.append(tmp)
            bert_prior = np.array([[x[1] for x in p] for p in sentence_prior])
            print("calculating posterior")
            posterior = bert_posterior(bert_prior,bert_dict,10)
            out_sentence = [dictionary[np.argmax(p)] for p in posterior]
            print("Posterior sentence",out_sentence)
            posterior_sentences[l][s] = out_sentence

    print("full posterior:\n")
    print(posterior_sentences)