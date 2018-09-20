import cv2 as cv
import numpy as np
import os
import copy


def extractFeature(imgR, imgL, detector="sift", pointsNum=None):
    if detector == "sift":
        feature = cv.xfeatures2d.SIFT_create(pointsNum)
    elif detector == "surf":
        feature = cv.xfeatures2d.SURF_create(pointsNum)
    else:
        feature = cv.ORB_create(pointsNum)
        # kpR, desR = feature.detectAndCompute(imgR, None)
        # kpL, desL = feature.detectAndCompute(imgL, None)
        # bf = cv.BFMatcher(cv.NORM_HAMMING)
        # matches = bf.knnMatch(desR, desL, k=2)
    kpR, desR = feature.detectAndCompute(imgR, None)
    kpL, desL = feature.detectAndCompute(imgL, None)
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(desR, desL, k=2)
    return [matches, kpR, kpL]


def Optimize(imgL, warpR, imgTmp):
    imgRes = imgTmp
    alpha = 1
    rows, cols = np.where(warpR[:, :, 0] != 0)
    start = min(cols)
    width = imgL.shape[1] - start
    for i in range(imgL.shape[0]):
        for j in range(start, imgL.shape[1]):
            if warpR[i, j, :].all() == 0:
                alpha = 1
                # alpha = (width - (j - start)) / width
            else:
                alpha = (width - (j - start)) / width
                # print([j,alpha])
            imgRes[i, j, :] = imgL[i, j, :] * alpha + wrapR[i, j, :] * (1 - alpha)
    return imgRes


if __name__ == '__main__':
    detector = "sift"
    cwd = os.getcwd()
    imgR = cv.imread("./originImg/05.jpg")
    imgL = cv.imread("./originImg/06.jpg")
    imgL = imgL[:,0:235,:]
    for pNum in range(500, 10000, 500):
        # 提取初步特征点
        [matches, kpR, kpL] = extractFeature(imgR, imgL, detector=detector, pointsNum=pNum)
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]  # 获取关键点的坐标
        src_pts = np.float32([kpR[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpL[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # cv.warpPerspective(img,h,(cols,rows))
        wrapR = cv.warpPerspective(imgR, H, (imgR.shape[1] + imgL.shape[1], imgL.shape[0]))
        imgTmp = copy.deepcopy(wrapR)
        imgTmp[0:imgL.shape[0], 0:imgL.shape[1]] = imgL
        # cv.imshow('wrap2.jpg', imgTmp)
        # cv.imshow('wrap1.jpg', wrapR)

        rows, cols = np.where(imgTmp[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        imgTmp = imgTmp[min_row:max_row, min_col:max_col, :]  # 去除黑色无用部分
        # 图像融合
        imgRes = Optimize(imgL, wrapR, imgTmp)

        # # 形态学处理,开运算
        # g = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
        # imgOpen = cv.morphologyEx(imgRes, cv.MORPH_OPEN, g)
        # res = abs(imgRes - imgTmp)
        resPath = cwd + "\\" + detector
        if not os.path.exists(resPath):
            os.mkdir(resPath)
        cv.imwrite(resPath + "\\" + "res" + str(pNum) + ".jpg", imgRes)
# cv.imshow('imgR.jpg', imgR)
# cv.imshow('imgL.jpg', imgL)
# cv.imshow('result.jpg', imgRes)
# # cv.imshow('res.jpg', res)
# cv.waitKey(0)
# cv.destroyAllWindows()
