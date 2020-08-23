import sys
sys.path.append("..")
from context_bert.bert_posterior import bert_posterior,bert_posterior_probabilistic
from edit_distance.edit_distance import levenshteinDistance
from util.util import get_full_word_dict

dictionary = get_full_word_dict()
def clean_sentence(sentence):
    