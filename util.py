import numpy as np
import pickle as pkl
import sys
import progressbar
from PIL import Image

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

def load_unlabeled_dataset(filename):
    with open(filename, 'r') as f:
        return [a.split("\t") for a in filter(lambda x:x!="",f.read().splitlines())]

def write_dataset(filename,dataset):
    """Write a dataset of dimensions NxM to a file. M should usually be 2"""
    with open(filename, 'w') as f:
        f.write("\n".join(["\t".join(x) for x in dataset]))

def cosine_similarity(a,b):
    return (a@b)/(np.linalg.norm(a)*np.linalg.norm(b))

def redraw_vec(vec,image_dim = (23,15)):
    """Reconstruct the letter from a visual embedding vector."""
    vec = vec.reshape(image_dim)
    image = Image.fromarray(vec.astype('uint8'), 'L')
    image.show()


def eval_cosine_similarity(embedding):
    similarity_scores = []
    sim_mean = []
    sim_max = []
    sim_min = []
    size = []
    bar = progressbar.ProgressBar(max_value=len(embedding)**2)
    i = 0
    j = 0
    for key1 in embedding:
        for key2 in embedding:
            similarity_scores.append(cosine_similarity(embedding[key1], embedding[key2]))
            j +=1
            i +=1
            if j == 100000:
                similarity_scores = np.array(similarity_scores)
                sim_max.append(np.max(similarity_scores))
                sim_mean.append(np.mean(similarity_scores))
                sim_min.append(np.min(similarity_scores))
                size.append(j)
                j = 0
                similarity_scores = []
            bar.update(i)
    if j !=0:
        similarity_scores = np.array(similarity_scores)
        sim_max.append(np.max(similarity_scores))
        sim_mean.append(np.mean(similarity_scores))
        sim_min.append(np.min(similarity_scores))
        size.append(j)
    sim_mean = calc_mean(sim_mean, size)
    sim_max = np.max(sim_max)
    sim_min = np.min(sim_min)
    with open("sim_eval.txt", "w") as f:
        f.write(str(sim_mean) + "\n")
        f.write(str(sim_max) + "\n")
        f.write(str(sim_min) + "\n")


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

if __name__ == '__main__':
    vec_dict = load_pickle("visual_embeddings.pkl")
    eval_cosine_similarity(vec_dict)
    vec1 = vec_dict[ord("ì")].astype(int)
    vec2 = vec_dict[ord("i")].astype(int)
    redraw_vec(vec1)
    redraw_vec(vec2)
    print(cosine_similarity(vec1, vec2))
    print(np.dot(vec1, vec2))
    """for i in range(len(vec1)):
        if vec1[i]!=0 and vec2[i]!=0:
            print(vec1[i],vec2[i])"""