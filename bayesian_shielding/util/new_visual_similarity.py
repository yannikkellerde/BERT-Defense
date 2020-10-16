import sys,os
sys.path.append("..")
from util.letter_stuff import letters,numbers,big_letters,small_letters
from util.utility import cosine_similarity
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import pickle
from tqdm import tqdm,trange

letnum = big_letters+numbers
back_map = {x:letnum[x] for x in range(len(letnum))}

def black_match_similarity(ray1,ray2,bl_sum):
    return np.sum(np.logical_and(ray1,ray2))/bl_sum

def get_img_in_forms(char_img,out_size):
    forms = []
    black_pix = np.sum(char_img)
    for rot in range(8):
        if rot<4:
            im_use = np.rot90(char_img,rot)
        else:
            if rot==4:
                im_use = np.flip(char_img,0)
            elif rot==5:
                im_use = np.flip(char_img,1)
            elif rot==6:
                im_use = np.flip(np.rot90(char_img,1),0)
            elif rot==7:
                im_use = np.flip(np.rot90(char_img,1),1)
        #for y_start in range(0,out_size[0]-im_use.shape[0]+1):
        #    for x_start in range(0,out_size[1]-im_use.shape[1]+1):
        x_start = 0
        for y_start in [0,out_size[0]-im_use.shape[0]]:
            form_im = np.pad(im_use,((y_start,out_size[0]-y_start-im_use.shape[0]),(x_start,out_size[1]-x_start-im_use.shape[1])),constant_values=(False,False))
            forms.append((form_im,black_pix,rot))
    return forms

def get_resized_font_img(char,font,out_size):
    txt = Image.new('L', out_size, (255,))
    try:
        d = ImageDraw.Draw(txt)
        d.text((0,0), char, font=font, fill=(0,))
    except OSError as e:
        print(e)
        return None
    np_img = np.array(txt)
    left = np_img.shape[0]
    right = 0
    top = np_img.shape[1]
    bottom = 0
    for y in range(np_img.shape[0]):
        for x in range(np_img.shape[1]):
            if np_img[y,x] < 160:
                left = min(left,x)
                right = max(right,x+1)
                top = min(top,y)
                bottom = max(bottom,y+1)
    if top>=bottom or left>=right:
        return None
    np_img = np_img[top:bottom,left:right]
    if out_size[0]/out_size[1] < np_img.shape[0]/np_img.shape[1]:
        new_size = (out_size[0],np_img.shape[1]*(out_size[0]/np_img.shape[0]))
    else:
        new_size = (np_img.shape[0]*(out_size[1]/np_img.shape[1]),out_size[1])
    im = Image.fromarray(np.uint8(np_img))
    im = im.resize(np.uint8(new_size)[::-1])
    np_img = np.array(im)<160
    return np_img

def get_in_multiple_sizes(char):
    ims = []
    combos = [((30,30),20),((27,27),18),((24,24),16),((21,21),14),((18,18),12),((15,15),10)]
    for out_size,fontsize in combos:
        font = char_to_font[char][fontsize]
        ims.append(get_resized_font_img(char,font,out_size))
    return ims

def get_in_all_forms(char,out_size):
    forms = []
    ims = get_in_multiple_sizes(char)
    for i,im in enumerate(ims):
        b = get_img_in_forms(im,out_size)
        forms.extend(b)
    return forms

def get_letter_forms(out_size):
    form_list = []
    for sl,bl in tqdm(zip(small_letters,big_letters)):
        form_list.append(get_in_all_forms(sl,out_size)+get_in_all_forms(bl,out_size))
    for num in tqdm(numbers):
        form_list.append(get_in_all_forms(num,out_size))
    return form_list

def get_sim(forms, img, img_black):
    maxsim = 0
    for form,black_form,rot in forms:
        sim = black_match_similarity(form,img,(img_black+black_form)/2)
        if rot!=0:
            sim*=0.9
        if sim > maxsim:
            maxsim = sim
    return maxsim

def display_boolray(img):
    im = Image.fromarray(np.uint8(~img)*255)
    im.show()


font_paths = [os.path.join("../binaries/noto-regular",x) for x in os.listdir("../binaries/noto-regular")]
pil_fonts = {x:{y:ImageFont.truetype(x, y) for y in range(10,21,2)} for x in font_paths}
with open("../binaries/char_font_map.pkl","rb") as f:
    char_to_path = pickle.load(f)
char_to_font = {key:pil_fonts[value] for key,value in char_to_path.items()}
char_count = 30000

out_size=(30,30)
fontsize = 20
sim_matrix = np.zeros((char_count,len(big_letters+numbers)))
forms = get_letter_forms(out_size)
#with open("../binaries/letter_forms.pkl","wb") as f:
#    pickle.dump(forms,f)
#with open("../binaries/letter_forms.pkl","rb") as f:
#    forms = pickle.load(f)
s = time.perf_counter()
for i in trange(char_count):
    char = chr(i)
    if char not in char_to_font:
        continue
    img = get_resized_font_img(char,char_to_font[char][fontsize],out_size)
    if img is None:
        continue
    img = np.pad(img,((0,out_size[0]-img.shape[0]),(0,out_size[1]-img.shape[1])),constant_values=(False,False))
    black_pix = np.sum(img)
    for j,form in enumerate(forms):
        sim = get_sim(form,img,black_pix)
        sim_matrix[i,j] = sim
    if i%1000==999:
        np.save("../binaries/vis_sim.npy",sim_matrix)
print(time.perf_counter()-s)
np.save("../binaries/vis_sim.npy",sim_matrix)