"""Create visual embeddings for a range of Unicode characters. Requires
an appropiate font that includes a wide range of glyphs for Unicode characters"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle as pkl
from tqdm import tqdm,trange

fnt = ImageFont.truetype('unifont-13.0.02.ttf', 25)
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
with open("visual_embeddings.pkl","wb") as f:
    pkl.dump(vec_dict,f)