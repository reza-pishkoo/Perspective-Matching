import random

import cv2
import numpy as np

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


cv2.imwrite("res13_corners.jpg", res1)

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


cv2.imwrite("res14_correspondences.jpg", x)

res3 = cv2.drawMatchesKnn(image1, key_points1, image2, key_points2, best_matches2, None, matchColor=(255, 0, 0), matchesMask=None, singlePointColor=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("res15_matches.jpg", res3)

chosen_matches = random.sample(best_matches2, 20)

res4 = cv2.drawMatchesKnn(image1, key_points1, image2, key_points2, chosen_matches, None, matchColor=(255, 0, 0), matchesMask=None, singlePointColor=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("res16.jpg", res4)


homography, mask = cv2.findHomography(np.array([key_points2[m.trainIdx].pt for m in best_matches]), np.array([key_points1[m.queryIdx].pt for m in best_matches]), method=cv2.RANSAC, ransacReprojThreshold=7, maxIters=1000)

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


cv2.imwrite("res17.jpg", x2)

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

cv2.imwrite("res19.jpg", x3)

transmission_matrix = np.array([[1, 0, 2700], [0, 1, 1200], [0, 0, 1]])
mat = np.matmul(transmission_matrix, homography)
res20 = cv2.warpPerspective(image2, mat, (9000, 3800))

cv2.imwrite("res20.jpg", res20)

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

cv2.imwrite("res21.jpg", images_together2)
