import sys
sys.path.append("..")
import numpy as np
import pickle as pkl
import sys
import progressbar
from PIL import Image
import multiprocessing
from itertools import chain
from functools import reduce
from util.letter_stuff import small_letters,big_letters

def get_most_likely_sentence(distribution,dic):
    sentence = ""
    for i,(p,c) in enumerate(distribution):
        pmaxin = np.argmax(p)
        if len(c) == 0:
            sentence += dic[pmaxin]
        else:
            csum = sum(x[1] for x in c)
            if p[pmaxin]/(1+csum)>c[0][1]:
            #if p[pmaxin]>c[0][1]:
                sentence += dic[pmaxin]
            else:
                for w in c[0][0]:
                    sentence += w.replace("##","")
        if i!=len(distribution)-1:
            sentence+=" "
    return sentence

def load_and_preprocess_dataset(filename):
    """
    Load a dataset in 3d list of dimensions LxSxW with
    L = number of lines
    S = number of sentences in line
    W = number of words in sentence

    The preprocessing includes
    1. Replacing multiple spaces with a single one
    2. Replacing spaces in the end of sentences
    3. Making periods, commas, quotation marks, semicolons and colons
       their own word, if they are in the beginning or end of a word.
    Then the words are seperated by splitting at spaces.
    """
    out_dataset = []
    with open(filename, 'r',  encoding="utf8") as f:
        for line in f.read().splitlines():
            one_line = []
            out_dataset.append(one_line)
            sentences = line.split("\t")
            for sentence in sentences:
                sentence=sentence.strip()
                for _ in range(3):
                    sentence.replace("  "," ")
                words = sentence.split(" ")
                newwords = []
                for i,word in enumerate(words):
                    lateradd = []
                    if len(word)>0 and (word[0]=='"' or word[0]=="'"):
                        newwords.append(word[0])
                        word = word[1:]
                    for _ in range(3):
                        if len(word)>0 and (word[-1]=="," or word[-1]=='"' or word[-1]==";" or word[-1]==":" or word[-1]=="'"):
                            lateradd.append(word[-1])
                            word = word[:-1]
                        if i==len(words)-1 and len(word)>1 and ((word[-1]=="." and word[-2]!=".") or word[-1]=="?" or word[-1]=="!"):
                            lateradd.append(word[-1])
                            word = word[:-1]
                    if len(word)>0:
                        newwords.append(word)
                    newwords.extend(list(reversed(lateradd)))
                one_line.append(newwords)
    return out_dataset
        
def load_dictionary(filename):
    with open(filename,'r', encoding="utf8") as f:
        return list(filter(lambda x:not x.startswith("#!comment:"),f.read().splitlines()))

def each_char_in(haystack,needle):
    for char in needle:
        if not char in haystack:
            return False
    return True

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)

def write_dataset(filename,dataset,as_sentences=False):
    """Write a dataset of dimensions LxSxW to a file. S is the amount of sentences and W the number of words"""
    with open(filename, 'w') as f:
        if as_sentences:
            f.write("\n".join(["\t".join(x) for x in dataset]))
        else:
            f.write("\n".join(["\t".join([" ".join(y) for y in x]) for x in dataset]))
def only_read_dataset(filename):
    with open(filename, 'r') as f:
        return [x.split("\t") for x in f.read().splitlines()]
def cosine_similarity(a,b):
    return (a@b)/(np.linalg.norm(a)*np.linalg.norm(b))

def redraw_vec(vec,image_dim = (23,15)):
    """Reconstruct the letter from a visual embedding vector."""
    vec = vec.reshape(image_dim)
    vec = (vec+np.min(vec))*(255/(np.max(vec)-np.min(vec)))
    image = Image.fromarray(vec.astype('uint8'), 'L')
    image.show()

def softmax(x, theta=1):
    ps = np.exp(x * theta)
    ps /= np.sum(ps)
    return ps

def calc_mean(means, size):
    mean = 0
    for i in range(len(means)):
        mean = mean + means[i] * size[i]
    norm = np.sum(np.array(size))
    return mean/norm

def fast_argmin(a):
    return min(range(len(a)), key=lambda x: a[x])

lower_dict = {x:x.lower() for x in big_letters}
def mylower(word):
    out = ""
    for letter in word:
        if letter in lower_dict:
            out += lower_dict[letter]
        else:
            out += letter
    return out


def read_labeled_data(link):
    text_to_read = open(link, "r", encoding="utf8")
    scores = []
    first_sentences = []
    second_sentences = []
    for line in text_to_read:
        point = line.split("\t")
        scores.append(float(point[0]))
        first_sentences.append(point[1].strip())
        second_sentences.append(point[2].strip())
    return scores, first_sentences, second_sentences


if __name__ == '__main__':
    word_embedding = load_pickle("../binaries/visual_embeddings.pkl")
    redraw_vec(word_embedding[ord("t")])