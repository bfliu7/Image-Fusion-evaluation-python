import cv2
import numpy as np


def __Qabf__(img_vi, img_ir, fused):
    assert isinstance(img_vi, (np.ndarray, list)) and isinstance(img_ir, (np.ndarray, list)) \
           and isinstance(fused, (np.ndarray, list))

    fused = fused.astype(np.float64)
    m, n, b = fused.shape
    m1, n1, b1 = img_ir.shape

    if b == 1:
        g = compute_Qabf(img_vi[:, :, 0], img_ir[:, :, 0], fused[:, :, 0])
        res = g
    elif b1 == 1:
        g = [compute_Qabf(img_vi[:, :, k], img_ir[:, :, 0], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [compute_Qabf(img_vi[:, :, k], img_ir[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)

    return res

def compute_angles(Sx, Sy):
    angles = np.arctan2(Sy, Sx)
    angles[Sx == 0] = np.pi / 2
    return angles


def compute_Qabf(strA, strB, strF):
    # model parameters
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    pA = strA.astype(np.float64)
    pB = strB.astype(np.float64)
    pF = strF.astype(np.float64)

    if len(pA.shape) > 2:
        pA = cv2.cvtColor(pA, cv2.COLOR_RGB2GRAY)
    if len(pB.shape) > 2:
        pB = cv2.cvtColor(pB, cv2.COLOR_RGB2GRAY)
    if len(pF.shape) > 2:
        pF = cv2.cvtColor(pF, cv2.COLOR_RGB2GRAY)

    pA = 255 * cv2.normalize(pA, None, 0, 1, cv2.NORM_MINMAX)
    pB = 255 * cv2.normalize(pB, None, 0, 1, cv2.NORM_MINMAX)
    pF = 255 * cv2.normalize(pF, None, 0, 1, cv2.NORM_MINMAX)

    SAx = cv2.Sobel(pA, cv2.CV_64F, 1, 0, ksize=3)
    SAy = cv2.Sobel(pA, cv2.CV_64F, 0, 1, ksize=3)
    gA = np.sqrt(SAx ** 2 + SAy ** 2)
    aA = compute_angles(SAx, SAy)

    SBx = cv2.Sobel(pB, cv2.CV_64F, 1, 0, ksize=3)
    SBy = cv2.Sobel(pB, cv2.CV_64F, 0, 1, ksize=3)
    gB = np.sqrt(SBx ** 2 + SBy ** 2)
    aB = compute_angles(SBx, SBy)

    SFx = cv2.Sobel(pF, cv2.CV_64F, 1, 0, ksize=3)
    SFy = cv2.Sobel(pF, cv2.CV_64F, 0, 1, ksize=3)
    gF = np.sqrt(SFx ** 2 + SFy ** 2)
    aF = compute_angles(SFx, SFy)

    h, w = gA.shape
    GAF = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if gA[i, j] > gF[i, j]:
                GAF[i, j] = gF[i, j] / gA[i, j]
            elif gA[i, j] == gF[i, j]:
                GAF[i, j] = gF[i, j]
            else:
                GAF[i, j] = gA[i, j] / gF[i, j]

    AAF = 1 - np.abs(aA - aF) / (np.pi / 2)
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF

    GBF = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if gB[i, j] > gF[i, j]:
                GBF[i, j] = gF[i, j] / gB[i, j]
            elif gB[i, j] == gF[i, j]:
                GBF[i, j] = gF[i, j]
            else:
                GBF[i, j] = gB[i, j] / gF[i, j]

    ABF = 1 - np.abs(aB - aF) / (np.pi / 2)
    QgBF = Tg / (1 + np.exp(kg * (GBF - Dg)))
    QaBF = Ta / (1 + np.exp(ka * (ABF - Da)))
    QBF = QgBF * QaBF

    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)
    return nume / deno