#!/usr/bin/env python3

import numpy as np
import cv2
from scipy.spatial import Delaunay
from obj_utils import Face, write_obj
import sys
import scipy.stats as st

n_colors = 16

def kmeans_color_quantization(image, clusters, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h * w, 3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y][0:3]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              clusters,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS +
                                               cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                              rounds,
                                              cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((h, w, 3))

print("reading image")
image = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
image = cv2.GaussianBlur(image, (3, 3), 0)

if (image.shape[2] == 4):
    alpha_mask = image[:,:,3] > 128
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)
    image *= alpha_mask

print("quantizing image")
result = kmeans_color_quantization(image, clusters=n_colors)

print("detecting image hotpoints")
src = gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

ddepth = cv2.CV_16S
kernel_size = 3
dst = cv2.Laplacian(src, ddepth, ksize=kernel_size)
abs_dst = cv2.convertScaleAbs(dst)
# edges = cv2.Canny(abs_dst,100,200)
edges = cv2.Canny(result,100,200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

approx_contours = np.ndarray((0, 2))
for cnt in contours:
    epsilon = 0.0 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    approx_contours = np.concatenate((
        approx_contours,
        approx.reshape((approx.shape[0], 2))
    ), axis=0)

contours = np.array(approx_contours)
# contours = np.concatenate(contours)
points = np.matrix.astype(contours.reshape((contours.shape[0], 2)), dtype=int)
points = np.unique(points, axis=0)
points = points.astype(int)
# contours = contours.reshape((contours.shape[0] * contours.shape[1], 2))
print(points.shape)

# for pt in points:
    # print(tuple(pt))
    # cv2.circle(result, tuple(pt), 1, (255,0,0), 1)
# cv2.imshow('result', result)
# cv2.waitKey()

# -- Draw keypoints
# img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
print("triangulating image")
# kpts = np.array([kp.pt for kp in keypoints])
delaunay_tri_inds = Delaunay(points, qhull_options='').simplices
triangles = points[delaunay_tri_inds]
# for tri in triangles:
    # tri = tri.astype(int)
    # cv2.line(result, tuple(tri[0]), tuple(tri[1]), (0, 0, 0), 1)
    # cv2.line(result, tuple(tri[1]), tuple(tri[2]), (0, 0, 0), 1)
    # cv2.line(result, tuple(tri[2]), tuple(tri[0]), (0, 0, 0), 1)

# cv2.imshow('result', result)
# cv2.waitKey()
print("getting colors")
# image, result, kpts, delaunay_tri_inds, triangles
materials = []
for triangle, inds in zip(triangles, delaunay_tri_inds):
    cent = np.average(triangle, axis=0)
    # print(triangle, cent)
    # print(cent)
    materials.append(result[int(cent[1]), int(cent[0])][::-1])

materials = np.unique(np.array(materials, dtype=int), axis=0)
# materials_index = {}
# for i, mat in enumerate(materials):
    # materials_index[(((mat[0] << 8) + mat[1]) << 8) + mat[2]] = i

print("setting colors")
faces = []
for triangle, inds in zip(triangles, delaunay_tri_inds):
    cent = np.average(triangle, axis=0)
    color = np.array(result[int(cent[1]), int(cent[0])][::-1])
    # print(np.where((materials == color).all(axis=1))[0][0])
    # face = Face(inds, materials_index[(((color[0] << 8) + color[1]) << 8) + color[2]])
    face = Face(inds, np.where((materials == color).all(axis=1))[0][0])
    faces.append(face)

materials = materials.astype(float) / 255

# kpts3d = np.array([(kp[0], -kp[1], 0) for kp in points])

# 3D-ify

print("3d-ifying")

x = points[:, 0]
y = -points[:, 1]
kpts3d = np.array([x, y, np.zeros(x.shape[0])]).T
# values = np.vstack([x, y])
# kernel = st.gaussian_kde(values)
# z = kernel(values)
# z *= 40 / z.max()
# kpts3d = np.array([x, y, z]).T

print('saving')
# Save the files
write_obj(kpts3d, faces, materials)

# cv2.drawKeypoints(result, keypoints, img_keypoints)
# cv2.imshow('SURF Keypoints', img_keypoints)

# cv2.imshow('result', result)
# cv2.waitKey()

# -i /Users/val/Documents/workspace/ObjTextureCreator/out.obj -t 0 -a 3 -mx 180 -z 3 -f 40 -bkg 0 -ks 0 -kd .6 -mtl
