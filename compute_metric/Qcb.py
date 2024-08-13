import numpy as np
import cv2


def __Qcb__(img1, img2, fused):
    fused = np.double(fused)
    img1 = np.double(img1)
    img2 = np.double(img2)

    # Get the size of img
    m, n, b = fused.shape
    m1, n1, b1 = img2.shape

    if b == 1:
        g = Qcb(img1[:, :, 0], img2[:, :, 0], fused[:, :, 0], )
        res = g
    elif b1 == 1:
        g = [Qcb(img1[:, :, k], img2[:, :, 0], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [Qcb(img1[:, :, k], img2[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)

    return res


def Qcb(im1, im2, fused):
    im1 = np.double(im1)
    im2 = np.double(im2)
    fused = np.double(fused)

    im1 = normalize1(im1)
    im2 = normalize1(im2)
    fused = normalize1(fused)

    # set up some constant values for experiment
    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622

    # parameters for local contrast computation
    k = 1
    h = 1
    p = 3
    q = 2
    Z = 0.0001
    sigma = 2

    # calculate the quality Q
    hang, lie = im1.shape

    # DoG filter
    HH = hang // 30
    LL = lie // 30

    u, v = np.meshgrid(np.linspace(-LL, LL, lie), np.linspace(-HH, HH, hang))
    r = np.sqrt(u ** 2 + v ** 2)

    Sd = np.exp(-(r / f0) ** 2) - a * np.exp(-(r / f1) ** 2)

    # contrast sensitivity filtering
    fused1 = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im1)) * Sd)))
    fused2 = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im2)) * Sd)))
    ffused = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(fused)) * Sd)))

    # local contrast computation
    # one level of contrast
    G1 = gaussian2d(hang, lie, 2)
    G2 = gaussian2d(hang, lie, 4)

    # filtering in frequency domain
    C1 = contrast(G1, G2, fused1)
    C1 = np.abs(C1)
    C1P = (k * (C1 ** p)) / (h * (C1 ** q) + Z)

    C2 = contrast(G1, G2, fused2)
    C2 = np.abs(C2)
    C2P = (k * (C2 ** p)) / (h * (C2 ** q) + Z)

    Cf = contrast(G1, G2, ffused)
    Cf = np.abs(Cf)
    CfP = (k * (Cf ** p)) / (h * (Cf ** q) + Z)

    # contrast preservation calculation
    mask = (C1P < CfP)
    mask = mask.astype(float)
    Q1F = (C1P / CfP) * mask + (CfP / C1P) * (1 - mask)

    mask = (C2P < CfP)
    mask = mask.astype(float)
    Q2F = (C2P / CfP) * mask + (CfP / C2P) * (1 - mask)

    # Saliency map generation
    ramda1 = (C1P * C1P) / (C1P * C1P + C2P * C2P)
    ramda2 = (C2P * C2P) / (C1P * C1P + C2P * C2P)

    # global quality map
    Q = ramda1 * Q1F + ramda2 * Q2F

    return np.mean(Q)


def gaussian2d(n1, n2, sigma):
    """Create a 2D Gaussian filter in the spatial domain."""
    H = (n1 - 1) // 2
    L = (n2 - 1) // 2

    x, y = np.meshgrid(np.arange(-15, 16), np.arange(-15, 16))
    G = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    return G


def contrast(G1, G2, im):
    buff = cv2.filter2D(im, -1, G1, borderType=cv2.BORDER_REPLICATE)
    buff1 = cv2.filter2D(im, -1, G2, borderType=cv2.BORDER_REPLICATE)

    return buff / buff1 - 1


def normalize1(data):
    """Normalize the data to the range of 0-255 (gray level)."""
    data = np.double(data)
    da = np.max(data)
    xiao = np.min(data)
    if da == 0 and xiao == 0:
        return data
    else:
        newdata = (data - xiao) / (da - xiao)
        return np.round(newdata * 255)
