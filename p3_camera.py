import numpy as np
import cv2
import math
import scipy
import sys
import os
from matplotlib import pyplot as plt

if __name__ == "__main__":
    """
    Deal with command line arguments assuming they are correct.
    Then drive the rest of part 3. Should be able to be done in main.
    """
    rot_vec1 = np.loadtxt(sys.argv[1], max_rows=1)
    rot_vec1 = (math.pi / 180) * rot_vec1
    cosines = np.cos(rot_vec1)
    sines = np.sin(rot_vec1)
    rot1_x = np.array([[1, 0, 0], [0, cosines[0], -sines[0]], [0, sines[0], cosines[0]]])
    rot1_y = np.array([[cosines[1], 0, sines[1]], [0, 1, 0], [-sines[1], 0, cosines[1]]])
    rot1_z = np.array([[cosines[2], -sines[2], 0], [sines[2], cosines[2], 0], [0, 0, 1]])
    rot1 = np.matmul(rot1_y, rot1_z)
    rot1 = np.matmul(rot1_x, rot1)
    rot_vec2 = np.loadtxt(sys.argv[1], skiprows=2, max_rows=1)
    rot_vec2 = (math.pi / 180) * rot_vec2
    cosines = np.cos(rot_vec2)
    sines = np.sin(rot_vec2)
    rot2_x = np.array([[1, 0, 0], [0, cosines[0], -sines[0]], [0, sines[0], cosines[0]]])
    rot2_y = np.array([[cosines[1], 0, sines[1]], [0, 1, 0], [-sines[1], 0, cosines[1]]])
    rot2_z = np.array([[cosines[2], -sines[2], 0], [sines[2], cosines[2], 0], [0, 0, 1]])
    rot2 = np.matmul(rot2_y, rot2_z)
    rot2 = np.matmul(rot2_x, rot2)
    k1_arr = np.loadtxt(sys.argv[1], skiprows=1, max_rows=1)
    k2_arr = np.loadtxt(sys.argv[1], skiprows=3, max_rows=1)
    s1 = k1_arr[0]
    uc1 = k1_arr[1]
    vc1 = k1_arr[2]
    s2 = k2_arr[0]
    uc2 = k2_arr[1]
    vc2 = k2_arr[2]
    k1 = np.array([[s1, 0, vc1], [0, s1, uc1], [0, 0, 1]])
    k2 = np.array([[s2, 0, vc2], [0, s2, uc2], [0, 0, 1]])
    H_21 = k2@rot2@rot1.T@np.linalg.inv(k1)
    if H_21[2, 2] < 0:
        H_21=H_21*-1
    H_21 = H_21/np.linalg.norm(H_21)
    H_21 = H_21*1000
    print("Matrix: H_21")
    np.set_printoptions(precision=3)
    print('{0:.3f}, {1:.3f}, {2:.3f}'.format(H_21[0, 0], H_21[0, 1], H_21[0, 2]))
    print('{0:.3f}, {1:.3f}, {2:.3f}'.format(H_21[1, 0], H_21[1, 1], H_21[1, 2]))
    print('{0:.3f}, {1:.3f}, {2:.3f}'.format(H_21[2, 0], H_21[2, 1], H_21[2, 2]))
    c1 = np.array([0, 0, 1])
    c2 = np.array([0, 6000, 1])
    c3 = np.array([4000, 0, 1])
    c4 = np.array([4000, 6000, 1])
    H_21=H_21/1000
    c1_i2 = H_21@c1
    c2_i2 = H_21 @ c2
    c3_i2 = H_21 @ c3
    c4_i2 = H_21 @ c4
    c1_i2=c1_i2/c1_i2[-1]
    c2_i2 = c2_i2 / c2_i2[-1]
    c3_i2 = c3_i2 / c3_i2[-1]
    c4_i2 = c4_i2 / c4_i2[-1]
    rows = np.array([c1[0], c2[0], c3[0], c4[0], c1_i2[0], c2_i2[0], c3_i2[0], c4_i2[0]])
    cols = np.array([c1[1], c2[1], c3[1], c4[1], c1_i2[1], c2_i2[1], c3_i2[1], c4_i2[1]])
    '''USe the mins and maxes of image 1 points in image 2 space to find UL and LR bounds'''