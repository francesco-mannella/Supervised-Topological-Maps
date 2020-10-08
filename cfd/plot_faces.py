import numpy as np
import matplotlib.pyplot as plt

faces = np.loadtxt("figs")
labels = np.loadtxt("labels", dtype="str")
faces = faces[labels == "AF"]

p = plt.imshow(np.zeros([107, 152]), cmap=plt.cm.gray, vmin=0, vmax=255)
for i,f in enumerate(faces):
    p.set_array(f.reshape(152, 107).T)
    plt.axis("off")
    plt.savefig("/tmp/af%05d.png" % i)
