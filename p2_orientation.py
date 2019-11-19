import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math



if __name__ == "__main__":
    """
        Deal with command line arguments assuming they are correct.
        Then drive the rest of part 1.
        Get the keypoints then calculate the distances and output.
    """
    sigma = float(sys.argv[1])
    img_name = sys.argv[2]
    points = np.loadtxt(sys.argv[3])
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = round(4*sigma)
    im_s = cv2.GaussianBlur(img.astype(np.float32), (2*width+1, 2*width+1), sigma)
    '''  Derivative kernels '''
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2

    '''  Derivatives '''
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)
    ''' Convert to Polar Values (phase between 0 and 360 better for making histogram)'''
    mag, phase = cv2.cartToPolar(im_dx, im_dy, angleInDegrees=True)
    phase = phase/10
    gauss = cv2.getGaussianKernel(2*width+1, 2*sigma)
    gauss = gauss * gauss.T
    orientation = np.zeros((points.shape[0], 36))
    nms_kernel = np.ones((3, 3), np.uint8)
    peak_test = cv2.dilate(mag.astype(np.float32), nms_kernel)
    peak_test[np.where(mag < peak_test)] = 0
    max_angles = []
    max_responses = []
    for i in range(points.shape[0]):
        hist = orientation[i,:]
        pt = points[i,:]
        x = int(pt[0])
        y = int(pt[1])
        mags = mag[x-width:x+width+1, y-width:y+width+1]
        angles = phase[x-width:x+width+1, y-width:y+width+1]
        mags = mags*gauss
        lo_index = np.floor(angles).astype(np.int32)
        forward_weight = angles-lo_index
        middle_weight = 1 - forward_weight
        middle_weight = middle_weight*mags
        forward_weight = forward_weight*mags
        hi_index=np.mod(lo_index+1, 36)
        mag_dilate = cv2.dilate(mags.astype(np.float32), nms_kernel)
        peak_mat = mags
        peak_mat[np.where(mags < mag_dilate)] = 0
        peak_angles = angles[np.where(peak_test[x-width:x+width+1, y-width:y+width+1] != 0)]-180
        norm_max_mags = peak_test[x-width:x+width+1, y-width:y+width+1]*gauss
        peak_resp = mag[np.where(peak_test[x-width:x+width+1, y-width:y+width+1] != 0)]
        max_angles.append(peak_angles)
        max_responses.append(peak_resp)
        for j in range(36):
            hist[j] += np.sum(middle_weight[np.where(lo_index == j)])
            hist[j] += np.sum(forward_weight[np.where(hi_index == j)])
    w_orientation = np.zeros(orientation.shape)
    for i in range(orientation.shape[1]):
        w_orientation[:, i] = 0.5*orientation[:, i] + .25*orientation[:, (i-1)%36] + .25*orientation[:, (i+1)%36]
    for i in range(points.shape[0]):
        pt = points[i, :]
        x = int(pt[0])
        y = int(pt[1])
        print("\n Point " + str(i) + ": (" + str(x) + "," + str(y) + ")")
        print("Histograms:")
        for j in range(36):
            start = 10*j - 180
            end = start+10
            bin = [start, end]
            print(str(bin) + ": " + format(orientation[i, j], '.2f') + " " + format(w_orientation[i, j], '.2f'))
        for j in range(len(max_angles[i])):
            resp = max_responses[i]
            ang = max_angles[i]
            resp, ang = (list(m) for m in zip(*sorted(zip(resp, ang))))
            resp = resp[::-1]
            ang = ang[::-1]
            print("Peak " + str(j) + ": theta " + format(ang[j], '.1f') +" " +
                  format(resp[j], '.2f'))
            strong_peaks = len(np.where(resp >= .8 * resp[0]))
        print("Number of strong orientation peaks: " + str(strong_peaks))




