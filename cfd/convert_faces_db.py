import numpy as np
# Convert face patterns from 2444x1718 RGB - to 152x107 8bit greyscale resolution
#  
# First you mast import all neutral faces from the Chicago Face Database to the folder
# ../faces.
#   
#   mkdir -p faces labels
#   
#

from PIL import Image

import glob
import re


figures = glob.glob("../faces/*.jpg")
w = 2444
h = 1718
ww = w//16
hh = h//16

pfigs = np.zeros([len(figures), ww*hh])
plabels = []

for k, fig in enumerate(figures):
    with Image.open(fig).convert("L") as jfig:
        jjfig = jfig.resize((ww,hh))
        pixels = np.hstack([jjfig.getpixel((i,j)) 
            for i in range(ww) for j in range(hh)])
        pfigs[k] = pixels
        lab_re = re.match(r".*CFD-(\w+)-.*", fig)
        plabels.append(lab_re.group(1))
        print(plabels[k])
np.savetxt("figs", pfigs.astype(int), fmt='%4d' )
np.savetxt("labels", plabels, fmt='%s' )

