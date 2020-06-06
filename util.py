import numpy as np
import pickle as pkl
import sys
from PIL import Image

def load_dictionary(filename):
    with open(filename,'r') as f:
        return list(filter(lambda x:not x.startswith("#!comment:"),f.read().splitlines()))

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

if __name__ == '__main__':
    vec_dict = load_embedding_file("visual_embeddings.pkl")
    vec1 = vec_dict[ord("ì")].astype(int)
    vec2 = vec_dict[ord("i")].astype(int)
    redraw_vec(vec1)
    redraw_vec(vec2)
    print(cosine_similarity(vec1, vec2))
    print(np.dot(vec1, vec2))
    """for i in range(len(vec1)):
        if vec1[i]!=0 and vec2[i]!=0:
            print(vec1[i],vec2[i])"""