import numpy as np


def __psnr__(img_vi, img_ir, fused):
    assert isinstance(img_vi, (np.ndarray, list)) and isinstance(img_ir, (np.ndarray, list)) \
           and isinstance(fused, (np.ndarray, list))

    m, n, b = fused.shape
    m1, n1, b1 = img_ir.shape
    img1 = img_vi.astype(np.float64)
    img2 = img_ir.astype(np.float64)

    if b == 1:
        g = compute_Psnr(img1, img2, fused)
        res = g
    elif b1 == 1:
        g = [compute_Psnr(img1[:, :, k], img2, fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [compute_Psnr(img1[:, :, k], img2[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)

    return res


def compute_Psnr(img1, img2, fused):
    B = 8
    MAX = 2 ** B - 1
    MES = (mse(img1, fused) + mse(img2, fused)) / 2.0
    PSNR = 20 * np.log10(MAX / np.sqrt(MES))
    return PSNR


def mse(a, b):
    if a.ndim > 2:
        a = np.mean(a, axis=2)
    if b.ndim > 2:
        b = np.mean(b, axis=2)

    m, n = a.shape
    temp = np.sqrt(np.sum((a - b) ** 2))
    res0 = temp / (m * n)
    return res0