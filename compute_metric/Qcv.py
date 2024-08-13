import numpy as np
import cv2


def __Qcv__(img1, img2, fused):
    fused = np.double(fused)

    # Get the size of img
    m, n, b = fused.shape
    m1, n1, b1 = img2.shape

    if b == 1:
        g = Qcv(img1[:, :, 0], img2[:, :, 0], fused[:, :, 0])
        res = g
    elif b1 == 1:
        g = [Qcv(img1[:, :, k], img2[:, :, 0], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)
    else:
        g = [Qcv(img1[:, :, k], img2[:, :, k], fused[:, :, k]) for k in range(b)]
        res = np.mean(g)

    return res


def Qcv(im1, im2, fused):
    # set up the constant values
    alpha_c = 1
    alpha_s = 0.685
    f_c = 97.3227
    f_s = 12.1653

    # local window size 16 x 16
    windowSize = 16

    # alpha = 1, 2, 3, 4, 5, 10, 15. This value is adjustable.
    alpha = 5

    # pre-processing
    im1 = np.double(im1)
    im2 = np.double(im2)
    fused = np.double(fused)

    im1 = normalize1(im1)
    im2 = normalize1(im2)
    fused = normalize1(fused)

    # Step 1: extract edge information
    flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    fuseX = cv2.filter2D(fused, -1, flt1, borderType=cv2.BORDER_REPLICATE)
    fuseY = cv2.filter2D(fused, -1, flt2, borderType=cv2.BORDER_REPLICATE)
    fuseG = np.sqrt(fuseX ** 2 + fuseY ** 2)

    buffer = (fuseX == 0)
    buffer = buffer * 0.00001
    fuseX = fuseX + buffer
    fuseA = np.arctan(fuseY / fuseX)

    img1X = cv2.filter2D(im1, -1, flt1, borderType=cv2.BORDER_REPLICATE)
    img1Y = cv2.filter2D(im1, -1, flt2, borderType=cv2.BORDER_REPLICATE)
    im1G = np.sqrt(img1X ** 2 + img1Y ** 2)

    buffer = (img1X == 0)
    buffer = buffer * 0.00001
    img1X = img1X + buffer
    im1A = np.arctan(img1Y / img1X)

    img2X = cv2.filter2D(im2, -1, flt1, borderType=cv2.BORDER_REPLICATE)
    img2Y = cv2.filter2D(im2, -1, flt2, borderType=cv2.BORDER_REPLICATE)
    im2G = np.sqrt(img2X ** 2 + img2Y ** 2)

    buffer = (img2X == 0)
    buffer = buffer * 0.00001
    img2X = img2X + buffer
    im2A = np.arctan(img2Y / img2X)

    # calculate the local region saliency
    hang, lie = im1.shape
    H = hang // windowSize
    L = lie // windowSize

    fun = lambda x: np.sum(np.sum(x ** alpha))

    ramda1 = cv2.copyMakeBorder(im1G, 0, windowSize - hang % windowSize, 0, windowSize - lie % windowSize,
                                cv2.BORDER_CONSTANT, value=0)
    ramda1 = cv2.resize(ramda1, (L * windowSize, H * windowSize))
    ramda1 = cv2.blur(ramda1, (windowSize, windowSize))
    ramda1 = cv2.filter2D(ramda1, -1, np.ones((windowSize, windowSize)) / (windowSize * windowSize))

    ramda2 = cv2.copyMakeBorder(im2G, 0, windowSize - hang % windowSize, 0, windowSize - lie % windowSize,
                                cv2.BORDER_CONSTANT, value=0)
    ramda2 = cv2.resize(ramda2, (L * windowSize, H * windowSize))
    ramda2 = cv2.blur(ramda2, (windowSize, windowSize))
    ramda2 = cv2.filter2D(ramda2, -1, np.ones((windowSize, windowSize)) / (windowSize * windowSize))

    # similarity measurement
    f1 = im1 - fused
    f2 = im2 - fused

    # filtering with CSF filters
    hang, lie = im1.shape
    u, v = np.meshgrid(np.linspace(-lie // 8, lie // 8, lie), np.linspace(-hang // 8, hang // 8, hang))
    r = np.sqrt(u ** 2 + v ** 2)

    # Mannos-Skarison's filter
    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-(0.144 * r) ** 1.1)

    # Daly's filter
    index = (r == 0)
    r[index] = 1
    buff = 0.008 / r ** 3 + 1
    buff[index] = 0
    buff = buff ** (-0.2)
    buff1 = -0.3 * r * np.sqrt(1 + 0.06 * np.exp(0.3 * r))
    theta_d = buff ** (-0.2) * (1.42 * r * np.exp(buff1))
    theta_d[index] = 0

    # Ahumada filter
    theta_a = alpha_c * np.exp(-(r / f_c) ** 2) - alpha_s * np.exp(-(r / f_s) ** 2)

    # filter the image in frequency domain
    ff1 = np.fft.fft2(f1)
    ff2 = np.fft.fft2(f2)

    Df1 = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(ff1) * theta_m)))
    Df2 = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(ff2) * theta_m)))

    fun2 = lambda x: np.mean(x ** 2)
    D1 = cv2.filter2D(Df1, -1, np.ones((windowSize, windowSize)) / (windowSize * windowSize),
                      borderType=cv2.BORDER_REPLICATE)
    D2 = cv2.filter2D(Df2, -1, np.ones((windowSize, windowSize)) / (windowSize * windowSize),
                      borderType=cv2.BORDER_REPLICATE)

    # global quality
    Q = np.sum(ramda1 * D1 + ramda2 * D2) / np.sum(ramda1 + ramda2)

    return Q


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
