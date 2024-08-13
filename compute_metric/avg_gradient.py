import numpy as np


def __avg_gradient__(img_vi, img_ir, fused):
    assert isinstance(img_vi, (np.ndarray, list)) and isinstance(img_ir, (np.ndarray, list)) \
           and isinstance(fused, (np.ndarray, list))

    r, c, b = fused.shape

    g = []
    for k in range(b):
        band = fused[:, :, k]
        # Calculate the gradient of the band
        dzdx = np.gradient(band, axis=1)  # Gradient in x-direction
        dzdy = np.gradient(band, axis=0)  # Gradient in y-direction
        s = np.sqrt((dzdx ** 2 + dzdy ** 2) / 2)  # Gradient magnitude
        g.append(np.sum(s) / ((r - 1) * (c - 1)))  # Sum of gradients divided by the number of pixels

    # Calculate the mean of the gradients over all bands
    res = np.mean(g)
    return res
