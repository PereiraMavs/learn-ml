import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    """Display an image with unnormalization."""
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
