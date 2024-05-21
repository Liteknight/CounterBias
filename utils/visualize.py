import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import make_grid

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap='jet')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def grid_show(imgs, nrows=32):
    show(make_grid(imgs, nrows))


def img_grid(img_list, rows=1, cols=12, cmap='gray', clim=None, titles=[]):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.1)

    for i, (ax, im) in enumerate(zip(grid, img_list)):
        ax.imshow(im, clim=clim, cmap=cmap)
        if len(titles):
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title=titles[i])
        else:
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return fig
