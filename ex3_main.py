import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("\nLK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    mat = opticalFlowPyrLK(img_1.astype(np.float), img_2.astype(np.float), stepSize=20, winSize=5, k=5)
    pts_pyr, uv_pyr = getUv(mat)
    et = time.time()
    print("\nHierarchical LK Demo")
    print("Time: {:.4f}".format(et - st))
    print(np.median(uv_pyr, 0))
    print(np.mean(uv_pyr, 0))
    displayOpticalFlow(img_2, pts_pyr, uv_pyr)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("\ncompareLK Demo")
    img_1 = cv2.cvtColor(cv2.imread("input/lucas_kanade.png"), cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(cv2.imread("input/lucas_kanade_pyr.png"), cv2.COLOR_BGR2RGB)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('lucas_kande')
    ax[1].set_title('lucas_kande_pyr')
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def TranslationLK(img_path):
    print("\nTranslationLK Demo")
    img_1 = cv2.cvtColor(cv2.imread("input/imTransA1.jpg"), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)

    t = np.array([[1, 0, .3],
                  [0, 1, .9],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    cv2.imwrite("input/imTransB1.jpg", cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR))
    matrix = findTranslationLK(img_1.astype(np.float), img_2.astype(np.float))
    img_3 = cv2.warpPerspective(img_1, matrix, img_1.shape[::-1])
    print("lk matrix")
    print(matrix)
    print("correct matrix")
    print(t)
    f, ax = plt.subplots(1, 3)
    plt.gray()
    ax[0].set_title('original image')
    ax[1].set_title('correct image')
    ax[2].set_title('lk image')
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[2].imshow(img_3)
    plt.show()


def RigidLK(img_path):
    print("\nRigidLK Demo")
    img_1 = cv2.cvtColor(cv2.imread("input/imRigidA1.jpg"), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    alpha = .5
    t = np.array([
        [math.cos(math.radians(alpha)), -math.sin(math.radians(alpha)), -.9],
        [math.sin(math.radians(alpha)), math.cos(math.radians(alpha)), .6],
        [0, 0, 1]], dtype=np.float)

    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    cv2.imwrite("input/imRigidB1.jpg", cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR))
    matrix = findRigidLK(img_1, img_2)
    img_3 = cv2.warpPerspective(img_1, matrix, img_1.shape[::-1])
    print("lk matrix")
    print(matrix)
    print("correct matrix")
    print(t)

    f, ax = plt.subplots(1, 3)
    plt.gray()
    ax[0].set_title('original image')
    ax[1].set_title('correct image')
    ax[2].set_title('lk image')
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[2].imshow(img_3)
    plt.show()


def TranslationCorr(img_path):
    print("\nTranslationCorr Demo")
    img_1 = cv2.cvtColor(cv2.imread("input/imTransA2.jpg"), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)

    t = np.array([[1, 0, 20],
                  [0, 1, 40],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    matrix = findTranslationCorr(img_1.astype(np.float), img_2.astype(np.float))
    img_3 = cv2.warpPerspective(img_1, matrix, img_1.shape[::-1])
    cv2.imwrite("input/imTransB2.jpg", cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR))

    print("corr matrix")
    print(matrix)
    print("correct matrix")
    print(t)
    f, ax = plt.subplots(1, 3)
    plt.gray()
    ax[0].set_title('original image')
    ax[1].set_title('correct image')
    ax[2].set_title('corr image')
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[2].imshow(img_3)
    plt.show()


def RigidCorr(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("\nRigidCorr Demo")
    img_1 = cv2.cvtColor(cv2.imread("input/imRigidA2.jpg"), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)

    alpha = 20
    t = np.array([
        [math.cos(math.radians(alpha)), -math.sin(math.radians(alpha)), 5],
        [math.sin(math.radians(alpha)), math.cos(math.radians(alpha)), 7],
        [0, 0, 1]])

    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    cv2.imwrite("input/imRigidB2.jpg", cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR))
    matrix = findRigidCorr(img_1, img_2)
    img_3 = cv2.warpPerspective(img_1, matrix, img_1.shape[::-1])
    print("corr matrix")
    print(matrix)
    print("correct matrix")
    print(t)

    f, ax = plt.subplots(1, 3)
    plt.gray()
    ax[0].set_title('original image')
    ax[1].set_title('correct image')
    ax[2].set_title('corr image')
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[2].imshow(img_3)
    plt.show()


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("\nimageWarpingDemo Demo")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)

    t = np.array([
        [0.8, 0, 0],
        [0, 0.8, 0],
        [0, 0, 1]], dtype=np.float)

    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])

    img_3 = warpImages(img_1, img_2, t)
    f, ax = plt.subplots(1, 3)
    plt.gray()
    ax[0].set_title('original image')
    ax[1].set_title('warped image')
    ax[2].set_title('backward image')
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[2].imshow(img_3)
    plt.show()


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("\nGaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("\nLaplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    print("\nblend Demo")
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())
    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)
    TranslationLK(img_path)
    RigidLK(img_path)
    TranslationCorr(img_path)
    RigidCorr(img_path)
    imageWarpingDemo(img_path)
    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
