from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle as pkl
from tqdm import tqdm,trange

fnt = ImageFont.truetype('unifont-13.0.02.ttf', 25)
vec_dict = {}

for i in trange(33,1000,1):
    txt = Image.new('L', (15,23), (255,))
    d = ImageDraw.Draw(txt)
    d.text((0,0), chr(i), font=fnt, fill=(0,))
    vec_dict[i] = [int(x) for x in np.array(txt).flatten()]

with open("visual_embeddings.pkl","wb") as f:
    pkl.dump(vec_dict,f)