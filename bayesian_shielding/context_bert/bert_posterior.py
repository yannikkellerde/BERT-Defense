import sys,os
sys.path.append("..")
import torch
import numpy as np
from typing import List,Tuple
import math
from util.utility import softmax,get_most_likely_sentence,get_full_word_dict,get_most_likely_sentence_multidics
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from context_bert.probabilistic_bert import my_BertForMaskedLM
import logging
from tqdm import tqdm,trange
from copy import deepcopy

class BertPosterior():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        print("Loading BERT")
        self.probmodel = my_BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.probmodel.eval()
        print("Done Loading BERT")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.probmodel = self.probmodel.to(self.device)
        self.probmodel.bert.device = self.device
        self.probmodel.bert.embeddings.device = self.device
        # Load pre-trained model tokenizer (vocabulary)
        self.gpt = None
        self.gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

        self.word_dic = get_full_word_dict()
        self.bert_dic = [self.tokenizer.vocab[x] for x in self.word_dic]

        self.mask_tensor = torch.zeros(len(self.tokenizer.vocab))
        self.mask_tensor[self.tokenizer.vocab["[MASK]"]]=1

        self.top_n = 4
        self.bert_theta = 0.25
        self.gpt_theta = 0.005
        self.hyperparams = ["top_n","bert_theta","gpt_theta"]

    def load_gtp(self):
        if self.gpt is None:
            print("loading gpt")
            self.gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
            self.gpt.eval()
            print("Done loading gpt")

    def set_hyperparams(self,**kwargs):
        if "top_n" in kwargs:
            self.top_n = kwargs["top_n"]
        if "bert_theta" in kwargs:
            self.bert_theta = kwargs["bert_theta"]
        if "gpt_theta" in kwargs:
            self.gpt_theta = kwargs["gpt_theta"]

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
        inner_tensor = inner_tensor.to(self.device)
        predictions = self.probmodel(inner_tensor)
        preds = predictions[0, mask_id].cpu().numpy()
        return softmax(np.array([preds[dictionary[i]] for i in range(len(dictionary))]),theta=self.bert_theta)

    def showprobs_hypothesis(self,probs,word_dic,amount=20):
        inds = np.flip(np.argsort(probs))
        print(inds,amount)
        return [[word_dic[inds[i]],probs[inds[i]]] if len(inds)>i else [] for i in range(amount)]

    def bert_posterior_for_hypothesis(self,prior,word_dics,iterations_left,alreadys=None,orig_prior=None,bert_dics=None,verbose=False):
        """ Improve sentence prediction via BERT.
        :param prior: list of numpy arrays that represent probability distributions over words for the respective word_dic/bert_dic
        :param word_dics: list of lists that each contain word strings. Aligned with prior.
        :param iterations_left: number of how many iterations are left
        :param alreadys: how many time was each word aready improved. List of integers.
        :param orig_prior: A copy of the original prior.
        :param bert_dics: list of lists, that each contain the bert-index of words. Alligned with prior and word_dics
        :param verbose: Print out info about each replacement and masking.
        :returns: A probability distibution in the same shape and form of the prior parameter.
        """
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
            if len(word_dics[-1][np.argmax(prior[-1])])>1:
                weights_tensor = self.convert_prior_to_weights_tensor_hypothesis(prior+[np.array([1])],bert_dics+[[self.tokenizer.vocab["."]]])
            else:
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

    def gpt_score_sentence(self,sentence):
        with torch.no_grad():
            tokenize_input = self.gpt_tokenizer.tokenize(sentence)
            tensor_input = torch.tensor([self.gpt_tokenizer.convert_tokens_to_ids(tokenize_input)])
            loss=self.gpt(tensor_input, lm_labels=tensor_input)
        return -math.exp(loss)

    def gpt_hypothesis(self,hypothesis,verbose=False):
        self.load_gtp()
        probs,sentences = zip(*hypothesis)
        likelihood = softmax(np.array([self.gpt_score_sentence(sentence) for sentence in sentences]),theta=self.gpt_theta)
        priors = np.array(probs)
        posterior = priors * likelihood
        posterior /= sum(posterior)
        if verbose:
            print(f"hypothesis prior probabilities: {probs}. Likelihood: {likelihood}. Posterior: {posterior}")
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

    def calc_likelihood_batch(self,attention_masks,weights_tensors,mask_ids,dictionaries):
        attention_mask = torch.stack(attention_masks)
        inner_tensor = torch.stack(weights_tensors)
        for i in range(inner_tensor.size(0)):
            inner_tensor[i][mask_ids[i]] = self.mask_tensor
        inner_tensor = inner_tensor.to(self.device)
        attention_mask = attention_mask.to(self.device)
        predictions = self.probmodel(inner_tensor,attention_mask=attention_mask)
        out = []
        for i in range(predictions.size(0)):
            pred = predictions[i,mask_ids[i]].cpu().numpy()
            out.append(softmax(np.array([pred[dictionaries[i][mask_ids[i]-1][j]] for j in range(len(dictionaries[i][mask_ids[i]-1]))]),theta=self.bert_theta))
        return out

    def get_weights_tensor(self,prior:List[np.ndarray],word_dics:List[List[str]],alreadys:List[int],bert_dics:List[List[int]],max_prior_len:int) -> Tuple[torch.Tensor,int]:
        """ Create the weights tensor to then put into BERT.
        :param prior: list of numpy arrays that represent probability distributions over words for the respective word_dic/bert_dic
        :param word_dics: list of lists that each contain word strings. Aligned with prior.
        :param alreadys: how many time was each word aready improved. List of integers.
        :param bert_dics: list of lists, that each contain the bert-index of words. Alligned with prior and word_dics
        :returns: A attention tensor signaling the parts of the resulting tensor that are actuall tokens,
                  a tensor containing the weighted word embeddings (by prior) and the id of the masked word.
        """
        mask_id = self.get_most_uncertain_index(prior,alreadys)+1
        if mask_id is None:
            return prior
        alreadys[mask_id-1] += 1
        if len(word_dics[-1][np.argmax(prior[-1])])>1:
            attention_mask,weights_tensor = self.convert_prior_to_weights_tensor_pad(prior+[np.array([1])],bert_dics+[[self.tokenizer.vocab["."]]],max_prior_len)
        else:
            attention_mask,weights_tensor = self.convert_prior_to_weights_tensor_pad(prior,bert_dics,max_prior_len)
        return attention_mask,weights_tensor,mask_id

    def convert_prior_to_weights_tensor_pad(self,prior,dicts,max_prior_len):
        weights_tensor = torch.zeros((max_prior_len+2,len(self.tokenizer.vocab)))
        attention_mask = torch.cat((torch.ones(len(prior)+2),torch.zeros(max_prior_len-len(prior))))
        weights_tensor[0][self.tokenizer.vocab["[CLS]"]]=1
        weights_tensor[len(prior)+1][self.tokenizer.vocab["[SEP]"]]=1
        for i,p in enumerate(prior):
            top_indices = np.flip(np.argsort(p))[:self.top_n]
            mysum = np.sum(p[top_indices])
            for j in top_indices:
                weights_tensor[i+1][dicts[i][j]] = p[j]/mysum
        return attention_mask,weights_tensor

    def batch_bert_posterior(self,priors_hypform:List[List[Tuple[float,List[Tuple[np.ndarray,List[str]]]]]],batch_size:int=128) -> List[List[Tuple[float,List[Tuple[np.ndarray,List[str]]]]]]:
        priors_nd_word_dics = sum([[content for prob,content in hyps] for hyps in priors_hypform],[])
        priors:List[List[np.ndarray]] = [[x[0] for x in hyp] for hyp in priors_nd_word_dics]
        all_word_dics:List[List[List[str]]] = [[x[1] for x in hyp] for hyp in priors_nd_word_dics]
        #priors,all_word_dics = zip(*priors_nd_word_dics)
        orig_priors = [x.copy() for x in priors]
        max_prior_len = max(len(x) for x in priors)
        all_alreadys = [np.zeros(len(prior)) for prior in priors]
        all_bert_dics:List[List[List[int]]] = [[[self.tokenizer.vocab[word] for word in word_dic] for word_dic in word_dics] for word_dics in all_word_dics]
        with torch.no_grad():
            for i in trange(max_prior_len):
                likelihoods = []
                all_mask_ids = []
                for i in trange(0,len(priors),batch_size):
                    weights_tensors = []
                    attention_masks = []
                    mask_ids = []
                    for prior,alreadys,word_dics,bert_dics in zip(priors[i:i+batch_size],all_alreadys[i:i+batch_size],all_word_dics[i:i+batch_size],all_bert_dics[i:i+batch_size]):
                        attention_mask,weights_tensor,mask_id = self.get_weights_tensor(prior,word_dics,alreadys=alreadys,bert_dics=bert_dics,max_prior_len=max_prior_len)
                        weights_tensors.append(weights_tensor)
                        attention_masks.append(attention_mask)
                        mask_ids.append(mask_id)
                    all_mask_ids.extend(mask_ids)
                    likelihoods.extend(self.calc_likelihood_batch(attention_masks,weights_tensors,mask_ids,all_bert_dics[i:i+batch_size]))
                for prior,orig_prior,likelihood,mask_id in zip(priors,orig_priors,likelihoods,all_mask_ids):
                    numerator = orig_prior[mask_id-1] * likelihood
                    prior[mask_id-1] = numerator/np.sum(numerator)
        hyped_posterior = []
        count = 0
        for hyp in priors_hypform:
            hyp_block = []
            for prob,content in hyp:
                hyp_block.append((prob,get_most_likely_sentence_multidics(priors[count],all_word_dics[count])))
                count+=1
            hyped_posterior.append(hyp_block)
        return hyped_posterior