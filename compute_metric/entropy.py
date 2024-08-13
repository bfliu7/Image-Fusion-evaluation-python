import cv2
import numpy as np



def __entropy__(img_vi, img_ir, fused):
    assert isinstance(img_vi, (np.ndarray, list)) and isinstance(img_ir, (np.ndarray, list)) \
           and isinstance(fused, (np.ndarray, list))
    fused = np.double(fused)
    m, n, b = fused.shape
    m1, n1, b1 = img_ir.shape

    if b == 1:
        g = compute_entropy(img_vi, img_ir, fused)
        res = g
    elif b1 == 1:
        g = [compute_entropy(img_vi[:, :, k], img_ir, fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [compute_entropy(img_vi[:, :, k], img_ir[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)

    return res


def compute_entropy(img_vi, img_ir, fused):
    if len(fused.shape)==2:
        fused = fused[:,:,np.newaxis]
    s = fused.shape[2]
    if s == 3:
        h1 = cv2.cvtColor(fused, cv2.COLOR_RGB2GRAY)
        h1 = h1[:, :, np.newaxis]
    else:
        h1 = fused

    m, n, _ = h1.shape
    X, _ = np.histogram(h1.flatten(), bins=256, range=(0, 255))

    result = 0.0
    for P in X / (m * n):
        if P > 0:
            result -= P * np.log2(P)

    return result
