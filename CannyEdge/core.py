""" Canny Edge Detection is based on the following five steps:

    1. Gaussian filter
    2. Gradient Intensity
    3. Non-maximum suppression
    4. Double threshold
    5. Edge tracking

    This module contains these five steps as five separate Python functions.
"""

# Module imports
from CannyEdge.utils import round_angle

# Third party imports
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import numpy as np
import cv2


def gs_filter(img, sigma):
    """ Step 1: Gaussian filter

    Args:
        img: Numpy ndarray of image
        sigma: Smoothing parameter

    Returns:
        Numpy ndarray of smoothed image
    """
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        # return np.array(cv2.GaussianBlur(np.array(img,dtype=np.uint8), (3,3),sigma),dtype=np.float)
        return np.array(cv2.bilateralFilter(np.array(img,dtype=np.uint8), 9, 75, 75),dtype=np.float)
        # return gaussian_filter(img, sigma)


def gradient_intensity(img):
    """ Step 2: Find gradients

    Args:
        img: Numpy ndarray of image to be processed (denoised image)

    Returns:
        G: gradient-intensed image
        D: gradient directions
    """

    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kx = np.array(
    #     [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.int32
    # )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Ky = np.array(
    #     [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.int32
    # )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Iy)  # 计算斜边 out=sqrt(x^2+y^2)
    D = np.arctan2(Iy, Ix)  # 计算角度
    return (G, D)
    

def suppression(img, D):
    """ Step 3: Non-maximum suppression

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img

    Returns:
        ...
    """

    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            # 从梯度方向寻找临近像素
            where = round_angle(D[i, j])
            try:
                if where == 0:  # dy=0 dx++ or dx--
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):  # x方向凸型变化
                        Z[i, j] = img[i, j]
                elif where == 90:  # dx=0 dy++ or dy--
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):  # y方向凸型变化
                        Z[i, j] = img[i, j]
                elif where == 135:  # x++ y-- or x-- y++
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):  # 135度方向凸型变化
                        Z[i, j] = img[i, j]
                elif where == 45:  # x++ y++ or x-- y--
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):  # 135度方向凸型变化
                        Z[i, j] = img[i, j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries.处理图像边界处的像素 """
                pass
    return Z


def threshold(img, t, T):
    """ Step 4: Thresholding
    Iterates through image pixels and marks them as WEAK and STRONG edge
    pixels based on the threshold values.

    Args:
        img: Numpy ndarray of image to be processed (suppressed image)
        t: lower threshold
        T: upper threshold

    Return:
        img: Thresholdes image

    """
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(50),
        'STRONG': np.int32(255),
    }

    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)

    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)

    return (img, cf.get('WEAK'))

def tracking(img, weak, strong=255):
    """ Step 5:
    将与弱边相邻的强边转化为强边
    Checks if edges marked as weak are connected to strong edges.

    Note that there are better methods (blob analysis) to do this,
    but they are more difficult to understand. This just checks neighbour
    edges.

    Also note that for perfomance reasons you wouldn't do this kind of tracking
    in a seperate loop, you would do it in the loop of the tresholding process.
    Since this is an **educational** implementation ment to generate plots
    to help people understand the major steps of the Canny Edge algorithm,
    we exceptionally don't care about perfomance here.

    Args:
        img: Numpy ndarray of image to be processed (thresholded image)
        weak: Value that was used to mark a weak edge in Step 4

    Returns:
        final Canny Edge image.
    """

    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)检查其中一个邻居是否强（默认为= 255）
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                         or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                         or (img[i+1, j + 1] == strong) or (img[i-1, j - 1] == strong)
                         or (img[i+1, j-1] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
