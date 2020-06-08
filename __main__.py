import numpy as np
from bert_posterior import bert_posterior, format_prior_and_dict
from edit_distance_Jan import get_word_dic_distance
from util import load_dictionary,load_pickle,load_and_preprocess_dataset

if __name__ == '__main__':
    print("loading dictionary, word embedding and dataset")
    dictionary = load_dictionary("DATA/bert_wiki_full_words.txt")
    word_embedding = load_pickle("visual_embeddings.pkl")
    dataset = load_and_preprocess_dataset("DATA/test-scoreboard-dataset.txt")
    in_sentence = dataset[3][1]
    print(in_sentence)
    print("calculating distances")
    prior = []
    for word in in_sentence:
        if word.lower() in dictionary:
            tmp = [(x,0) for x in dictionary]
            tmp[dictionary.index(word.lower())] = (word.lower(),1)
            prior.append(tmp)
        else:
            prior.append(get_word_dic_distance(word, dictionary, word_embedding))
    print("Only distance normalized sentence: ",[list(sorted([(y[1],y[0]) for y in x],reverse=True))[0][1] for x in prior])
    print("formating prior and dict")
    bert_prior,bert_dict = format_prior_and_dict(prior,dictionary)
    print("calculating posterior")
    posterior = bert_posterior(bert_prior,bert_dict,10)
    out_sentence = [dictionary[np.argmax(p)] for p in posterior]
    print(out_sentence)