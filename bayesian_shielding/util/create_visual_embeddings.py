"""Create visual embeddings for a range of Unicode characters. Requires
an appropiate font that includes a wide range of glyphs for Unicode characters"""
import sys
sys.path.append("..")
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle as pkl
from tqdm import tqdm,trange
from sklearn.decomposition import PCA

fnt = ImageFont.truetype('../binaries/unifont-13.0.02.ttf', 25)
vec_dict = {}

for i in trange(0,30000,1):
    txt = Image.new('L', (15,23), (255,))
    d = ImageDraw.Draw(txt)
    d.text((0,0), chr(i), font=fnt, fill=(0,))
    vec_dict[i] = np.array(txt).flatten().astype(int)
for i in range(20):
    for key in tqdm(vec_dict):
        vec_dict[key] = vec_dict[key]/np.linalg.norm(vec_dict[key])
    all_vecs = np.array(list(vec_dict.values()))
    mean = np.mean(all_vecs)
    for key in tqdm(vec_dict):
        vec_dict[key] = (vec_dict[key]-mean)
    print(mean,np.linalg.norm(vec_dict[ord("O")]))
with open("../binaries/visual_embeddings.pkl","wb") as f:
    pkl.dump(vec_dict,f)

vecs = []
keys = []
for key in tqdm(vec_dict):
    vecs.append(vec_dict[key])
    keys.append(key)
pca = PCA(n_components=150)
pca.fit(vecs)
print(np.sum(pca.explained_variance_ratio_))
vecs = pca.transform(vecs)
vec_dict = {}
for key, vec in zip(keys, vecs):
    vec_dict[key] = vec
for i in range(20):
    for key in tqdm(vec_dict):
        vec_dict[key] = vec_dict[key]/np.linalg.norm(vec_dict[key])
    all_vecs = np.array(list(vec_dict.values()))
    mean = np.mean(all_vecs)
    for key in tqdm(vec_dict):
        vec_dict[key] = (vec_dict[key]-mean)
    print(mean,np.linalg.norm(vec_dict[ord("O")]))
with open("../binaries/visual_embeddings_pca.pkl","wb") as f:
    pkl.dump(vec_dict,f)    
