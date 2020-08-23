import sys
sys.path.append("../..")
import numpy as np
import io
from sentence_transformers import SentenceTransformer
import logging
logger = logging.getLogger()


def init_model_roberta():
    return SentenceTransformer("roberta-large-nli-stsb-mean-tokens")

def simple_sentence_embedder(model, sentences):
    return model.encode(sentences, show_progress_bar=True)

def encode_one_sentence(model, sentence):
    logger.debug("Encoding Sentence: "+sentence)
    sentences = [sentence]
    sentence_embeddings = model.encode(sentences,show_progress_bar=False)
    return sentence_embeddings[0]