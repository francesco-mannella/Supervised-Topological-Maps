import PIL.Image as Image
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.ioff()


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig, shape=None):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    if shape is not None:
        w, h = shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


# %%


class BarrColors:

    def __init__(self, n):

        self.n = n
        colors = np.random.uniform(0, 1, [n, 3])
        orientations = np.random.uniform(0, 2*np.pi, n)

        fig = plt.figure(figsize=(1,1), dpi=30)

        self.imgs = []
        for i in range(n):
            fig.clear()
            ax = fig.add_subplot(111, aspect="equal")
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            c = colors[i]
            a = orientations[i]
            na = (a + np.pi)
            coords = np.array([
                [np.cos(a), np.sin(a)],
                [np.cos(na), np.sin(na)]])

            ax.plot(*coords.T, lw=10, c=c)
            ax.set_axis_off()
            img = fig2img(fig)
            self.imgs.append(np.array(img))

bc = BarrColors(20000)
np.save("rodsdb", bc.imgs)
