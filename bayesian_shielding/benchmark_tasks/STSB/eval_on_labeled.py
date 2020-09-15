import sys
sys.path.append("../..")
from STSB.RoBERTa_handler import init_model_roberta,simple_sentence_embedder
from util.utility import read_labeled_data,cosine_similarity
import scipy.stats

scores,first_sentences,second_sentences = read_labeled_data("../../../DATA/training-dataset.txt")

model = init_model_roberta()

embed_first_sentences = simple_sentence_embedder(model,first_sentences)
embed_second_sentences = simple_sentence_embedder(model,second_sentences)

cosine_sims = [cosine_similarity(x,y) for x,y in zip(embed_first_sentences,embed_second_sentences)]

print(scipy.stats.spearmanr(scores,cosine_sims))