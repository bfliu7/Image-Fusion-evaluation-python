import numpy as np
import cv2

def __edge_intensity__(img_vi, img_ir, fused):
    assert isinstance(img_vi, (np.ndarray, list)) and isinstance(img_ir, (np.ndarray, list)) \
           and isinstance(fused, (np.ndarray, list))

    # sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    # sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    # grad_x = cv2.filter2D(fused, -1, sobel_x)
    # grad_y = cv2.filter2D(fused, -1, sobel_y)
    # grad_magnitude = cv2.magnitude(grad_x, grad_y)


    sobel_x = cv2.Sobel(fused, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(fused, cv2.CV_64F, 0, 1, ksize=3)
    # Compute the gradient magnitude
    grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Calculate the mean of the gradient magnitude
    res = np.mean(grad_magnitude)

    return res

