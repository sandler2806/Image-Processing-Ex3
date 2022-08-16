import math
import sys
from typing import List

import numpy as np
import cv2
import pygame
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 319097036


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        plt.gray()
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        plt.gray()

    xy = []
    duv = []
    win = win_size // 2
    kernel = np.array([[-1, 0, 1]])
    x = cv2.filter2D(im2, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    y = cv2.filter2D(im2, -1, kernel.T, borderType=cv2.BORDER_REPLICATE)
    t = im2 - im1
    for i in range(step_size, im1.shape[0] - win + 1, step_size):
        for j in range(step_size, im1.shape[1] - win + 1, step_size):
            it = t[i - win:i + win + 1, j - win:j + win + 1].flatten().T
            ix = x[i - win:i + win + 1, j - win:j + win + 1].flatten()
            iy = y[i - win:i + win + 1, j - win:j + win + 1].flatten()

            a = np.array([ix, iy]).T
            ata = a.T @ a
            b = a.T @ (-1 * it)
            b = b.reshape(2, 1)
            eigenvalues = np.linalg.eigvals(ata)

            if eigenvalues.max() / eigenvalues.min() < 100 and eigenvalues.min() > 1:  # and (uv[0]!=0 or uv[1]!=0):
                atainv = np.linalg.inv(ata)
                uv = atainv @ b

                duv.append([uv[0, 0], uv[1, 0]])
            else:
                duv.append(np.array([0, 0]))
            xy.append([j, i])

    return np.array(xy), np.array(duv)


def opticalFlow2(im1: np.ndarray, im2: np.ndarray, step_size=10,
                 win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        plt.gray()
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        plt.gray()

    xy = []
    duv = []
    win = win_size // 2
    kernel = np.array([[-1, 0, 1]])
    x = cv2.filter2D(im2, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    y = cv2.filter2D(im2, -1, kernel.T, borderType=cv2.BORDER_REPLICATE)
    t = im2 - im1
    for i in range(step_size, im1.shape[0] - win + 1, step_size):
        for j in range(step_size, im1.shape[1] - win + 1, step_size):
            it = t[i - win:i + win + 1, j - win:j + win + 1].flatten().T
            ix = x[i - win:i + win + 1, j - win:j + win + 1].flatten()
            iy = y[i - win:i + win + 1, j - win:j + win + 1].flatten()

            a = np.array([ix, iy]).T
            ata = a.T @ a
            b = a.T @ (-1 * it)
            b = b.reshape(2, 1)
            eigenvalues = np.linalg.eigvals(ata)

            if eigenvalues.max() / eigenvalues.min() < 100 and eigenvalues.min() > 1:  # and (uv[0]!=0 or uv[1]!=0):
                atainv = np.linalg.inv(ata)
                uv = atainv @ b

                duv.append([uv[0, 0], uv[1, 0]])
            else:
                duv.append(np.array([0, 0]))
            xy.append([j, i])

    return xy, duv


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int):
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    img1pyr = gaussianPyr(img1, k)
    img1pyr.reverse()
    img2pyr = gaussianPyr(img2, k)
    img2pyr.reverse()
    xy, uv = opticalFlow(img1pyr[0], img2pyr[0], stepSize, winSize)
    xy, uv = list(xy), list(uv)
    for i in range(1, k):
        xyi, uvi = opticalFlow2(img1pyr[i], img2pyr[i], stepSize, winSize)
        for j in range(len(xy)):
            xy[j] = [element * 2 for element in xy[j]]
            uv[j] = [element * 2 for element in uv[j]]

        for pixel, uv_current in zip(xyi, uvi):
            if pixel not in xy:
                xy.append(pixel)
                uv.append(uv_current)
            else:
                uv[xy.index(pixel)][0] += uv_current[0]
                uv[xy.index(pixel)][1] += uv_current[1]

    mat = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if [j, i] in xy:
                mat[i, j] = uv[xy.index([j, i])]
            else:
                mat[i, j] = [0, 0]
    return mat


def getUv(mat):
    pts_pyr = []
    uv_pyr = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            x = mat[i, j]
            if mat[i, j][0] != 0 or mat[i, j][1] != 0:
                pts_pyr.append([j, i])
                uv_pyr.append(mat[i, j])
    return np.array(pts_pyr), np.array(uv_pyr)


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    xy, uv = opticalFlow(im1, im2, 20, 5)
    u = []
    v = []
    for a in uv:
        u.append(a[0] * 2)
        v.append(a[1] * 2)

    return np.array([[1, 0, np.median(u)], [0, 1, np.median(v)], [0, 0, 1]])


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """

    matrix = opticalFlowPyrLK(im1, im2, stepSize=20, winSize=5, k=5)
    xy, uv = getUv(matrix)
    degrees = []

    for point, uvi in zip(xy, uv):
        point = np.flip(point)
        point2 = (point[0] + uvi[1], point[1] + uvi[0])
        ang = find_angle(point, point2)
        degrees.append(abs(ang))
    degrees = [x for x in degrees if x >= 0.1]

    angle = np.mean(degrees) * 2 if np.mean(degrees) * 2 > 0.61 else np.median(degrees) * 2

    matrix = opticalFlowPyrLK(im1, im2, stepSize=20, winSize=5, k=5)
    xy, uv = getUv(matrix)
    u = []
    v = []
    for a in uv:
        u.append(a[0])
        v.append(a[1])
    u = [x * 2 for x in u if abs(x) >= 0.1]
    v = [x for x in v if abs(x) >= 0.1]

    return np.array([[math.cos(math.radians(angle)), -math.sin(math.radians(angle)), np.median(u)],
                     [math.sin(math.radians(angle)), math.cos(math.radians(angle)), np.median(v)],
                     [0, 0, 1]])


def find_angle(p1, p2):
    v1 = pygame.math.Vector2(p1[0] - 0, p1[1] - 0)
    v2 = pygame.math.Vector2(p2[0] - 0, p2[1] - 0)
    ang = v1.angle_to(v2)
    return ang


def correlation(im1, im2):
    shape = np.max(im1.shape) // 2
    fft1 = np.fft.fft2(np.pad(im1, shape))
    fft2 = np.fft.fft2(np.pad(im2, shape))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + shape:-shape + 1, 1 + shape:-shape + 1]
    y1, x1 = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(im2.shape) // 2
    return x1, x2, y1, y2


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    x1, x2, y1, y2 = correlation(im1, im2)
    x = x2 - x1 - 1
    y = y2 - y1 - 1
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float)


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """

    x1, x2, y1, y2 = correlation(im1, im2)
    angle = find_angle((x1, y1), (x2, y2))

    matrix = np.float32([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
        [0, 0, 1]
    ])
    matrix = np.linalg.inv(matrix)
    reverse = cv2.warpPerspective(im2, matrix, im2.shape[::-1])

    x1, x2, y1, y2 = correlation(im1, reverse)

    x = x2 - x1 - 1
    y = y2 - y1 - 1

    return np.float32([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), x],
        [math.sin(math.radians(angle)), math.cos(math.radians(angle)), y],
        [0, 0, 1]
    ])


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    img = np.zeros_like(im2)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            indexes = [[i], [j], [1]]
            new_indexes = T @ indexes
            new_indexes[0] = new_indexes[0] / new_indexes[2]
            new_indexes[1] = new_indexes[1] / new_indexes[2]

            x = math.floor(new_indexes[0])
            y = math.floor(new_indexes[1])
            a = new_indexes[0] - x
            b = new_indexes[1] - y
            if a[0] != 0 or b[0] != 0:
                img[i, j] = (1 - a[0]) * (1 - b[0]) * im2[x, y] + a[0] * (1 - b[0]) * im2[x + 1, y] + a[0] * b[0] * im2[
                    x + 1, y + 1] + (1 - a[0]) * b[0] * im2[x, y + 1]
            else:
                img[i, j] = im2[x, y]

    return img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    n = 2 ** levels
    width, height = n * int(img.shape[1] / n), n * int(img.shape[0] / n)
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float64)
    pyramids = [img]
    gauss_kernel = cv2.getGaussianKernel(5, sigma=0.3 * (4 * 0.5 - 1) + 0.8)
    for level in range(1, levels):
        img = cv2.filter2D(img, -1, kernel=gauss_kernel, borderType=cv2.BORDER_REPLICATE)
        img = img[::2, ::2]
        pyramids.append(img)
    return pyramids


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gauss_kernel = cv2.getGaussianKernel(5, sigma=0.3 * (4 * 0.5 - 1) + 0.8)
    gauss_kernel = np.dot(gauss_kernel, gauss_kernel.T)
    pyr = gaussianPyr(img, levels)
    pyr.reverse()
    laplacian_pyr = [pyr[0]]
    expand_pyr = []
    for i in range(len(pyr) - 1):
        current_pyr = pyr[i]
        x, y = current_pyr.shape[0], current_pyr.shape[1]
        if len(current_pyr.shape) > 2:
            shape = (x * 2, y * 2, 3)
        else:
            shape = (x * 2, y * 2)
        preImg = np.zeros(shape)
        preImg[::2, ::2] = current_pyr
        expand_pyr.append(cv2.filter2D(preImg, -1, gauss_kernel * 4, borderType=cv2.BORDER_REPLICATE))
    for i in range(len(expand_pyr)):
        laplacian_i = cv2.subtract(pyr[i + 1], expand_pyr[i])
        laplacian_pyr.append(laplacian_i)

    laplacian_pyr.reverse()
    return laplacian_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gauss_kernel = cv2.getGaussianKernel(5, sigma=0.3 * (4 * 0.5 - 1) + 0.8)
    gauss_kernel = np.dot(gauss_kernel, gauss_kernel.T)
    lap_pyr.reverse()
    img = lap_pyr[0]
    for i in range(1, len(lap_pyr)):
        x, y = img.shape[0], img.shape[1]
        if len(img.shape) == 3:  # RGB
            shape = (x * 2, y * 2, 3)
        else:  # GRAY
            shape = (x * 2, y * 2)
        preImg = np.zeros(shape)
        preImg[::2, ::2] = img
        img = cv2.filter2D(preImg, -1, gauss_kernel * 4, borderType=cv2.BORDER_REPLICATE)
        img = cv2.add(img, lap_pyr[i])
    lap_pyr.reverse()
    return img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    n = 2 ** levels
    width, height = n * int(img_1.shape[1] / n), n * int(img_1.shape[0] / n)
    img_1 = cv2.resize(img_1, (width, height))
    img_1 = img_1.astype(np.float64)
    img_2 = cv2.resize(img_2, (width, height))
    img_2 = img_2.astype(np.float64)
    mask = cv2.resize(mask, (width, height))
    mask = mask.astype(np.float64)

    pyr_m = gaussianPyr(mask, levels)
    pyr_1 = laplaceianReduce(img_1, levels)
    pyr_2 = laplaceianReduce(img_2, levels)
    pyr_c = []
    naive_blend = np.zeros_like(img_1)
    for i in range(len(pyr_m)):
        pyr = np.zeros_like(pyr_1[i])
        for x in range(pyr_m[i].shape[0]):
            for y in range(pyr_m[i].shape[1]):
                pyr[x, y] = pyr_m[i][x, y] * pyr_1[i][x, y] + (1 - pyr_m[i][x, y]) * pyr_2[i][x, y]
                if i == 0:
                    naive_blend[x, y] = mask[x, y] * img_1[x, y] + (1 - mask[x, y]) * img_2[x, y]
        pyr_c.append(pyr)
    img = laplaceianExpand(pyr_c)

    return img, naive_blend
