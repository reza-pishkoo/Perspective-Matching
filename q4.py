import random

import cv2
import numpy as np


def my_find_homography(srcPoints, dstPoints, ransacReprojThreshold=10000000, maxIters=1000):
    points_count = srcPoints.shape[0]
    real_mask = np.zeros((points_count, 1), dtype=int)
    real_support_num = 0
    for itr in range(maxIters):
        mask = np.zeros((points_count, 1), dtype=int)
        support_num = 0
        rnd_list = np.random.permutation(points_count)[:4]

        res_matrix = np.zeros((8, 9))

        for i in range(4):
            src_point = srcPoints[rnd_list[i]]
            dst_point = dstPoints[rnd_list[i]]
            x = src_point[0]
            y = src_point[1]
            x_prim = dst_point[0]
            y_prim = dst_point[1]

            a_matrix = np.array([[-x, -y, -1, 0, 0, 0, x*x_prim, y*x_prim, x_prim],
                                 [0, 0, 0, -x, -y, -1, x*y_prim, y*y_prim, y_prim]])
            res_matrix[2*i:2*i+2, ] = a_matrix

        U, S, V = np.linalg.svd(res_matrix)

        homography = V[-1, :]
        homography = homography.reshape((3, 3))

        try:
            homography_inv = np.linalg.inv(homography)
        except:
            continue

        for i in range(points_count):
            X = np.array([[srcPoints[i, 0]], [srcPoints[i, 1]], [1]])
            X_prim = np.array([[dstPoints[i, 0]], [dstPoints[i, 1]], [1]])
            tmp_mat = np.matmul(homography, X)
            tmp_mat = tmp_mat / tmp_mat[-1]
            err1 = tmp_mat - X_prim
            tmp_mat2 = np.matmul(homography_inv, X_prim)
            tmp_mat2 = tmp_mat2 / tmp_mat2[-1]
            err2 = tmp_mat2 - X
            error = np.linalg.norm(err1) ** 2 + np.linalg.norm(err2) ** 2
            if error < ransacReprojThreshold:
                mask[i] = 1
                support_num = support_num + 1

        if support_num > 0.95 * points_count:
            real_mask = mask.copy()
            real_support_num = support_num
            break

        if support_num > real_support_num:
            real_mask = mask.copy()
            real_support_num = support_num

    counter = 0
    b_matrix = np.zeros((2 * real_support_num, 9))
    for k in range(points_count):
        if real_mask[k] == 1:
            src_point = srcPoints[k]
            dst_point = dstPoints[k]
            x = src_point[0]
            y = src_point[1]
            x_prim = dst_point[0]
            y_prim = dst_point[1]

            a_matrix = np.array([[-x, -y, -1, 0, 0, 0, x * x_prim, y * x_prim, x_prim],
                                 [0, 0, 0, -x, -y, -1, x * y_prim, y * y_prim, y_prim]])
            b_matrix[2 * counter:2 * counter + 2, ] = a_matrix
            counter = counter + 1

    U, S, V = np.linalg.svd(b_matrix)

    homography = V[-1, :]
    homography = homography / homography[-1]
    homography = homography.reshape((3, 3))
    return homography, real_mask

image1 = cv2.imread("im03.jpg")
image1_H = image1.shape[0]
image1_W = image1.shape[1]
image2 = cv2.imread("im04.jpg")
image2_H = image2.shape[0]
image2_W = image2.shape[1]

images_together = np.zeros((image1_H, image2_W + image1_W, 3))
images_together[:image1_H, :image1_W, ] = image1
images_together[:image2_H, image1_W:image1_W + image2_W, ] = image2

sift = cv2.SIFT_create()
key_points1, des1 = sift.detectAndCompute(image1, None)

key_points2, des2 = sift.detectAndCompute(image2, None)

res1 = cv2.drawMatches(image1, key_points1, image2, key_points2, None, None, None, singlePointColor=(0, 255, 0))


# BFMatcher
# drawMatches


cv2.imwrite("res22.jpg", res1)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

best_matches = []
best_matches2 = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        best_matches.append(m)
        best_matches2.append([m])

# res2 = cv2.drawMatches(image1, key_points1[best_matches.queryIdx])
x = images_together.copy()
for i in range(len(best_matches)):
    cv2.circle(x, ((int)(key_points1[best_matches[i].queryIdx].pt[0]), ((int)(key_points1[best_matches[i].queryIdx].pt[1]))), 15, (255, 0, 0), -1)
    cv2.circle(x, ((int)(key_points2[best_matches[i].trainIdx].pt[0]) + image1_W, ((int)(key_points2[best_matches[i].trainIdx].pt[1]))), 15, (255, 0, 0), -1)


cv2.imwrite("res23.jpg", x)

res3 = cv2.drawMatchesKnn(image1, key_points1, image2, key_points2, best_matches2, None, matchColor=(255, 0, 0), matchesMask=None, singlePointColor=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("res24.jpg", res3)

chosen_matches = random.sample(best_matches2, 20)

res4 = cv2.drawMatchesKnn(image1, key_points1, image2, key_points2, chosen_matches, None, matchColor=(255, 0, 0), matchesMask=None, singlePointColor=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("res25.jpg", res4)

homography, mask = my_find_homography(np.array([key_points2[m.trainIdx].pt for m in best_matches]), np.array([key_points1[m.queryIdx].pt for m in best_matches]), ransacReprojThreshold=1000, maxIters=1000)

x2 = images_together.copy()
for i in range(mask.shape[0]):
    if mask[i] == 1:
        cv2.circle(x2, ((int)(key_points1[best_matches[i].queryIdx].pt[0]), ((int)(key_points1[best_matches[i].queryIdx].pt[1]))), 7,(255, 0, 0), -1)
        cv2.circle(x2, ((int)(key_points2[best_matches[i].trainIdx].pt[0]) + image1_W,((int)(key_points2[best_matches[i].trainIdx].pt[1]))), 7, (255, 0, 0), -1)
        cv2.line(x2,
                 ((int)(key_points1[best_matches[i].queryIdx].pt[0]),
                  ((int)(key_points1[best_matches[i].queryIdx].pt[1]))),
                 ((int)(key_points2[best_matches[i].trainIdx].pt[0]) + image1_W,
                  ((int)(key_points2[best_matches[i].trainIdx].pt[1]))), (255, 0, 0))
    else:
        cv2.circle(x2, ((int)(key_points1[best_matches[i].queryIdx].pt[0]), ((int)(key_points1[best_matches[i].queryIdx].pt[1]))), 7, (0, 0, 255), -1)
        cv2.circle(x2, ((int)(key_points2[best_matches[i].trainIdx].pt[0]) + image1_W,((int)(key_points2[best_matches[i].trainIdx].pt[1]))), 7, (0, 0, 255), -1)


cv2.imwrite("res26.jpg", x2)

point1 = (0, 0, 1)
point2 = (image1_W, 0, 1)
point3 = (image1_W, image1_H, 1)
point4 = (0, image1_H, 1)

homography_inv = np.linalg.inv(homography)

point1_res = np.matmul(homography_inv, point1)
point1_res = point1_res / point1_res[2]

point2_res = np.matmul(homography_inv, point2)
point2_res = point2_res / point2_res[2]

point3_res = np.matmul(homography_inv, point3)
point3_res = point3_res / point3_res[2]

point4_res = np.matmul(homography_inv, point4)
point4_res = point4_res / point4_res[2]

x3 = images_together.copy()

cv2.line(x3, ((int)(point2_res[0]) + image1_W, (int)(point2_res[1])), ((int)(point1_res[0]) + image1_W, (int)(point1_res[1])), (0, 0, 255), thickness=5)
cv2.line(x3, ((int)(point3_res[0]) + image1_W, (int)(point3_res[1])), ((int)(point2_res[0]) + image1_W, (int)(point2_res[1])), (0, 0, 255), thickness=5)
cv2.line(x3, ((int)(point4_res[0]) + image1_W, (int)(point4_res[1])), ((int)(point3_res[0]) + image1_W, (int)(point3_res[1])), (0, 0, 255), thickness=5)
cv2.line(x3, ((int)(point1_res[0]) + image1_W, (int)(point1_res[1])), ((int)(point4_res[0]) + image1_W, (int)(point4_res[1])), (0, 0, 255), thickness=5)

cv2.imwrite("res28.jpg", x3)

transmission_matrix = np.array([[1, 0, 3100], [0, 1, 1350], [0, 0, 1]])
mat = np.matmul(transmission_matrix, homography)
res20 = cv2.warpPerspective(image2, mat, (9000, 3800))

cv2.imwrite("res29.jpg", res20)

image2_copy = image2.copy()
cv2.line(image2_copy, ((int)(point2_res[0]), (int)(point2_res[1])), ((int)(point1_res[0]), (int)(point1_res[1])), (0, 0, 255), thickness=5)
cv2.line(image2_copy, ((int)(point3_res[0]), (int)(point3_res[1])), ((int)(point2_res[0]), (int)(point2_res[1])), (0, 0, 255), thickness=5)
cv2.line(image2_copy, ((int)(point4_res[0]), (int)(point4_res[1])), ((int)(point3_res[0]), (int)(point3_res[1])), (0, 0, 255), thickness=5)
cv2.line(image2_copy, ((int)(point1_res[0]), (int)(point1_res[1])), ((int)(point4_res[0]), (int)(point4_res[1])), (0, 0, 255), thickness=5)

x4 = cv2.warpPerspective(image2_copy, mat, (9000, 3800))

x4_H = x4.shape[0]
x4_W = x4.shape[1]

images_together2 = np.zeros((x4_H, x4_W + image1_W, 3))
images_together2[:image1_H, :image1_W, ] = image1
images_together2[:x4_H, image1_W:image1_W + x4_W, ] = x4

cv2.imwrite("res30.jpg", images_together2)




