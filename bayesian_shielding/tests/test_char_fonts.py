import sys
sys.path.append("..")
from PIL import Image, ImageDraw, ImageFont

with open("../../DATA/test-scoreboard-dataset.txt","r") as f:
    uniques = set()
    for char in f.read():
        if char!="\n" and char!="\t" and char!=" ":
            uniques.add(char)
uniques = "".join(sorted(list(uniques)))


txt = Image.new('L', (1500,300), (255,))

# get a font
fnt = ImageFont.truetype('../binaries/unifont-13.0.02.ttf', 20)
# get a drawing context
d = ImageDraw.Draw(txt)

# draw text, half opacity
for i in range(0,len(uniques),60):
    d.text((10,10+i/2), uniques[i:i+60], font=fnt, fill=(0,))

txt.show()
