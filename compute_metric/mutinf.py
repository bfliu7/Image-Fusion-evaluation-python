import cv2
import numpy as np


def __mutinf__(img_vi, img_ir, fused):
    assert isinstance(img_vi, (np.ndarray, list)) and isinstance(img_ir, (np.ndarray, list)) \
           and isinstance(fused, (np.ndarray, list))
    fused = np.double(fused)
    m, n, b = fused.shape
    m1, n1, b1 = img_ir.shape

    if b == 1:
        g = mutinf_single(img_vi, img_ir, fused)
        res = g
    elif b1 == 1:
        g = [mutinf_single(img_vi[:, :, k], img_ir, fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [mutinf_single(img_vi[:, :, k], img_ir[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.ean(g)

    return res


def mutinf_single(img1, img2, fused):
    if len(img1.shape) == 2:
        img1 = img1[:, :, np.newaxis]
    if len(img2.shape) == 2:
        img2 = img2[:, :, np.newaxis]
    if len(fused.shape) == 2:
        fused = fused[:, :, np.newaxis]
    mi_vi = compute_mutinf(img1, fused)
    mi_ir = compute_mutinf(img2, fused)
    mi = mi_vi + mi_ir
    return mi

def compute_mutinf(img, fused):
    if img.shape[2] == 3:
        f1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f1 = f1[:, :, np.newaxis]
    else:
        f1 = img

    if fused.shape[2] == 3:
        f2 = cv2.cvtColor(fused, cv2.COLOR_RGB2GRAY)
        f2 = f2[:, :, np.newaxis]
    else:
        f2 = fused

    G1 = f1.squeeze().astype(np.float64)
    G2 = f2.squeeze().astype(np.float64)

    m1, n1 = G1.shape
    X1, _ = np.histogram(G1.flatten(), bins=256, range=(0, 255))
    X2, _ = np.histogram(G2.flatten(), bins=256, range=(0, 255))

    P1 = X1 / len(G1.flatten())
    P2 = X2 / len(G2.flatten())

    result = 0.0
    for i in range(256):
        if P1[i] > 0 and P2[i] > 0:
            result += P1[i] * np.log2(P1[i] / P2[i])

    return result
