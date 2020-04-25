#!/usr/bin/env python3

import numpy as np
from obj_utils import write_obj, Face
import random
from scipy.stats import multivariate_normal

r = 0.0
w = 40
h = 24

cloud = np.zeros((h, w, 3))
randomizer = np.vectorize(lambda x : random.SystemRandom().uniform(-r, r))
cloud = randomizer(cloud)
# print(cloud)

rv = multivariate_normal([-3, -2], [[1, .5], [.5,5]])

verts = []
i = 0
for line in cloud:
    j = 0
    for pt in line:
        verts.append((
            j + pt[0],
            i + pt[1],
            rv.pdf([(j + pt[0] - w / 2) / 2, (i + pt[1] - h / 2) / 2]) * 50
        ))
        j += 1
    i += 1

materials = np.array([[0xFF, 0xB0, 0x00]]) / 0xFF

faces = []
for i in range(1, h):
    for j in range(1, w):
        faces.append(Face((
            w * i + j - 1,
            w * i + j,
            w * (i - 1) + j
        ), 0))
        faces.append(Face((
            w * (i - 1) + j - 1,
            w * (i - 1) + j,
            w * i + j - 1
        ), 0))

write_obj(verts, faces, materials)
