import sys
sys.path.append("..")
import torch
import numpy as np
import math
from util.utility import softmax,get_most_likely_sentence,get_full_word_dict,get_most_likely_sentence_multidics
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from context_bert.probabilistic_bert import my_BertForMaskedLM
import logging
from copy import deepcopy

class BertPosterior():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        print("Loading BERT")
        self.model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.model.eval()
        self.probmodel = my_BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.probmodel.eval()
        print("Done Loading BERT, loading GTP now")
        self.gtp = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.gtp.eval()
        # Load pre-trained model tokenizer (vocabulary)
        self.gtp_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        print("Done loading GTP")

        self.word_dic = get_full_word_dict()
        self.bert_dic = [self.tokenizer.vocab[x] for x in self.word_dic]

        self.mask_tensor = torch.zeros(len(self.tokenizer.vocab))
        self.mask_tensor[self.tokenizer.vocab["[MASK]"]]=1

        self.top_n = 5
        self.bert_theta = 0.5
        self.gtp_theta = 0.005
        self.hyperparams = ["top_n","bert_theta","gtp_theta"]

    def set_hyperparams(self,**kwargs):
        if "top_n" in kwargs:
            self.top_n = kwargs["top_n"]
        if "bert_theta" in kwargs:
            self.bert_theta = kwargs["bert_theta"]
        if "gtp_theta" in kwargs:
            self.gtp_theta = kwargs["gtp_theta"]

    def convert_prior_to_weights_tensor_hypothesis(self,prior,dicts):
        weights_tensor = torch.zeros((len(prior)+2,len(self.tokenizer.vocab)))
        weights_tensor[0][self.tokenizer.vocab["[CLS]"]]=1
        weights_tensor[len(prior)+1][self.tokenizer.vocab["[SEP]"]]=1
        for i,p in enumerate(prior):
            top_indices = np.flip(np.argsort(p))[:self.top_n]
            mysum = np.sum(p[top_indices])
            for j in top_indices:
                weights_tensor[i+1][dicts[i][j]] = p[j]/mysum
        return weights_tensor

    def calc_probabilistic_likelihood_hypothesis(self,weights_tensor,mask_id,dictionary):
        inner_tensor = weights_tensor.clone().reshape((1,weights_tensor.size(0),weights_tensor.size(1)))
        inner_tensor[0][mask_id] = self.mask_tensor
        predictions = self.probmodel(inner_tensor)
        preds = predictions[0, mask_id].numpy()
        return softmax(np.array([preds[dictionary[i]] for i in range(len(dictionary))]),theta=self.bert_theta)

    def showprobs_hypothesis(self,probs,word_dic,amount=20):
        inds = np.flip(np.argsort(probs))
        print(inds,amount)
        return [[word_dic[inds[i]],probs[inds[i]]] if len(inds)>i else [] for i in range(amount)]

    def bert_posterior_for_hypothesis(self,prior,word_dics,iterations_left,alreadys=None,orig_prior=None,bert_dics=None,verbose=False):
        if iterations_left <= 0:
            return prior
        if alreadys is None:
            alreadys = np.zeros(len(prior))
        if bert_dics is None:
            bert_dics = [[self.tokenizer.vocab[word] for word in word_dic] for word_dic in word_dics]
        mask_id = self.get_most_uncertain_index(prior,alreadys)+1
        if mask_id is None:
            return prior
        alreadys[mask_id-1] += 1
        with torch.no_grad():
            weights_tensor = self.convert_prior_to_weights_tensor_hypothesis(prior,bert_dics)
            likelihood = self.calc_probabilistic_likelihood_hypothesis(weights_tensor,mask_id,bert_dics[mask_id-1])
        if orig_prior is None:
            numerator = prior[mask_id-1] * likelihood
        else:
            numerator = orig_prior[mask_id-1] * likelihood
        prior[mask_id-1] = numerator/np.sum(numerator)
        if verbose:
            print("masked",mask_id-1)
            print("likelihood",self.showprobs_hypothesis(likelihood,word_dics[mask_id-1],10))
            print("posterior",get_most_likely_sentence_multidics(prior,word_dics))
        return self.bert_posterior_for_hypothesis(prior,word_dics,iterations_left-1,alreadys,orig_prior,bert_dics,verbose)

    def gtp_score_sentence(self,sentence):
        with torch.no_grad():
            tokenize_input = self.gtp_tokenizer.tokenize(sentence)
            tensor_input = torch.tensor([self.gtp_tokenizer.convert_tokens_to_ids(tokenize_input)])
            loss=self.gtp(tensor_input, lm_labels=tensor_input)
        return -math.exp(loss)

    def gtp_hypothesis(self,hypothesis):
        probs,sentences = zip(*hypothesis)
        likelihood = softmax(np.array([self.gtp_score_sentence(sentence) for sentence in sentences]),theta=self.gtp_theta)
        priors = np.array(probs)
        posterior = priors * likelihood
        posterior /= sum(posterior)
        hyps = list(zip(posterior,sentences))
        hyps.sort(reverse=True)
        return hyps

    def get_most_uncertain_index(self,prior,alreadys):
        my_min = np.inf
        lowest = None
        for i,p in enumerate(prior):
            if len(p)<2:
                continue
            s = np.partition(-p,1)
            diff = s[1]-s[0]
            if diff == 1 or s[0]==0:
                diff = 100
            if diff+alreadys[i]<my_min:
                lowest = i
                my_min = diff+alreadys[i]
        return lowest