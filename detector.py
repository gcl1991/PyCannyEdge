from CannyEdge.utils import to_ndarray
from CannyEdge.core import (gs_filter, gradient_intensity, suppression,
                            threshold, tracking)

import matplotlib.pyplot as plt
from copy import copy
import argparse
import logging

# Argparse
parser = argparse.ArgumentParser(description='Educational Canny Edge Detector')
parser.add_argument('source', metavar='src', help='image source (jpg, png)')
parser.add_argument('sigma', type=float, metavar='sigma', help='Gaussian smoothing parameter')
parser.add_argument('t', type=int, metavar='t', help='lower threshold')
parser.add_argument('T', type=int, metavar='T', help='upper threshold')
parser.add_argument("--all", help="Plot all in-between steps")
args = parser.parse_args()
logging.basicConfig(filename='run.log', filemode='w',level=logging.DEBUG)


def ced(img_file, sigma, t, T, all=False):
    logging.debug('img_file {}'.format(img_file))
    # 0 图像灰度化
    img = to_ndarray(img_file)
    logging.debug('img shape {}'.format(img.shape))
    if not all:
        # avoid copies, just do all steps:＃避免复制，只需执行所有步骤：
        # 1 高斯滤波，去除噪声
        img = gs_filter(img, sigma)
        # 2 计算梯度强度(x^2+y^2)和夹角(arctan(y/x))
        img, D = gradient_intensity(img)
        # 3 捕获梯度变化
        img = suppression(img, D)
        # 4 根据阈值对像素分类
        img, weak = threshold(img, t, T)
        img = tracking(img, weak)
        return [img]
    else:
        # make copies, step by step一步一步地制作副本
        img1 = gs_filter(img, sigma)
        logging.debug('filter {} {}'.format(type(img1),img1.shape))
        img2, D = gradient_intensity(img1)
        img3 = suppression(copy(img2), D)
        img4, weak = threshold(copy(img3), t, T)
        img5 = tracking(copy(img4), weak)
        return [to_ndarray(img_file), img1, img2, img3, img4, img5]


def plot(img_list):
    for d, img in enumerate(img_list):
        plt.subplot(1, len(img_list), d+1), plt.imshow(img, cmap='gray'),
        plt.xticks([]), plt.yticks([])
    plt.show()


img_list = ced(args.source, args.sigma, args.t, args.T, all=args.all)
plot(img_list)
