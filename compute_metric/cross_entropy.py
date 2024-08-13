import numpy as np
import cv2


def __cross_entropy__(img_vi, img_ir, fused):
    assert isinstance(img_vi, (np.ndarray, list)) and isinstance(img_ir, (np.ndarray, list)) \
           and isinstance(fused, (np.ndarray, list))

    m, n, b = fused.shape
    m1, n1, b1 = img_ir.shape

    if b == 1:  # fused single channel
        g = cross_entropy_single(img_vi, img_ir, fused)
        res = g
    elif b1 == 1:  #
        g = [cross_entropy_single(img_vi[:, :, k], img_ir, fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [cross_entropy_single(img_vi[:, :, k], img_ir[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)

    return res


def cross_entropy_single(img1, img2, fused):
    if len(img1.shape)==2:
        img1 = img1[:,:,np.newaxis]
    if len(img2.shape)==2:
        img2 = img2[:,:,np.newaxis]
    if len(fused.shape)==2:
        fused = fused[:,:,np.newaxis]

    cross_entropy_vi = compute_cross_entropy(img1, fused)
    cross_entropy_ir = compute_cross_entropy(img2, fused)
    return (cross_entropy_vi + cross_entropy_ir) / 2.0


def compute_cross_entropy(img1, fused):
    if img1.shape[2] == 3:
        f1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        f1 = f1[:, :, np.newaxis]
    else:
        f1 = img1

    if fused.shape[2] == 3:
        f2 = cv2.cvtColor(fused, cv2.COLOR_RGB2GRAY)
        f2 = f2[:, :, np.newaxis]
    else:
        f2 = fused

    g1 = f1.astype(np.float64)
    g2 = f2.astype(np.float64)


    x1, _ = np.histogram(g1.flatten(), bins=256, range=(0, 255))
    x2, _ = np.histogram(g2.flatten(), bins=256, range=(0, 255))

    p1 = x1 / len(g1.flatten())
    p2 = x2 / len(g2.flatten())

    result = 0.0
    for i in range(256):
        if p1[i] > 0 and p2[i] > 0:
            result += p1[i] * np.log2(p1[i] / p2[i])

    return result

