import torch
import numpy as np
from util import softmax
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

print("loading bert model ...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
print("done loading bert model")

def format_prior_and_dict(prior,dictionary):
    """prior is a WxD 2d numpy float array with W the amount of words in sentence and D the amount of words in the dictionary.
    dictionary is a list of words
    """
    new_prior=[]
    for p in prior:
        d = [(dictionary.index(x[0]), x[1]) for x in p]
        new_prior.append([x[1] for x in sorted(d)])
    new_prior = np.array(new_prior)
    new_dictionary = [tokenizer.vocab[x] for x in dictionary]
    return new_prior, new_dictionary

def bert_posterior(prior,dictionary,maxdepth):
    return bert_posterior_recur(prior.copy(),prior,np.zeros(len(prior)),dictionary,maxdepth)

def bert_posterior_recur(orig_priors,prior,alreadys,dictionary,maxdepth):
    """Function partly based on https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model
    Prior and dictionary should be formated according to format_prior_and_dict
    """
    if maxdepth<=0:
        return prior
    sent_ray = [dictionary[np.argmax(p)] for p in prior]
    print(tokenizer.convert_ids_to_tokens(sent_ray))
    my_min = np.inf
    for i,p in enumerate(prior):
        s = np.sort(p)
        diff = s[-1]-s[-2]
        if diff==1:
            diff = 100
        if diff+alreadys[i]<my_min:
            lowest = i
            my_min = diff
    alreadys[lowest] += 1
    old_word = sent_ray[lowest]
    sent_ray[lowest] = tokenizer.vocab["[MASK]"]
    indexed_tokens = [tokenizer.vocab["[CLS]"]]+sent_ray+[tokenizer.vocab["[SEP]"]]
    print(tokenizer.convert_ids_to_tokens(indexed_tokens))
    # Create the segments tensors.
    segments_ids = [0] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    masked_index = indexed_tokens.index(tokenizer.vocab["[MASK]"])
    preds = predictions[0, masked_index].numpy()
    likelihood = softmax(np.array([preds[dictionary[i]] for i in range(len(dictionary))]))

    best_indexied = list(reversed(np.argsort(preds)))
    best_scores = [likelihood[dictionary.index(index)] for index in best_indexied[:10] if index in dictionary]
    predicted_tokens = list(zip(tokenizer.convert_ids_to_tokens(best_indexied[:5]),best_scores[:5]))
    print(predicted_tokens)
    print(tokenizer.convert_ids_to_tokens([old_word])[0],likelihood[dictionary.index(old_word)])

    numerator = orig_priors[lowest] * likelihood
    posterior = prior.copy()
    posterior[lowest] = numerator/np.sum(numerator)
    #if dictionary[np.argmax(posterior[lowest])] != old_word:
    return bert_posterior_recur(orig_priors,posterior,alreadys, dictionary, maxdepth-1)
    #return posterior