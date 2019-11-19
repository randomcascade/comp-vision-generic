import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math


def harris(img, sigma, filename, ext):
    """
    Ripped directly from class code. Thresholding step was removed.
    """
    '''  Gaussian smoothing '''
    ksize = (4 * sigma + 1, 4 * sigma + 1)
    im_s = cv2.GaussianBlur(img.astype(np.float32), ksize, sigma)

    '''  Derivative kernels '''
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2

    '''  Derivatives '''
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)

    ''' Components of the outer product '''
    im_dx_sq = im_dx * im_dx
    im_dy_sq = im_dy * im_dy
    im_dx_dy = im_dx * im_dy

    ''' Convolution of the outer product with the Gaussian kernel
        gives the summed values desired '''
    h_sigma = 2 * sigma
    h_ksize = (4 * h_sigma + 1, 4 * h_sigma + 1)
    im_dx_sq = cv2.GaussianBlur(im_dx_sq, h_ksize, h_sigma)
    im_dy_sq = cv2.GaussianBlur(im_dy_sq, h_ksize, h_sigma)
    im_dx_dy = cv2.GaussianBlur(im_dx_dy, h_ksize, h_sigma)

    ''' Compute the Harris measure '''
    kappa = 0.004
    im_det = im_dx_sq * im_dy_sq - im_dx_dy * im_dx_dy
    im_trace = im_dx_sq + im_dy_sq
    im_harris = im_det - kappa * im_trace * im_trace
    ''' Renormalize the intensities into the 0..255 range '''
    i_min = np.min(im_harris)
    i_max = np.max(im_harris)
    im_harris = 255 * (im_harris - i_min) / (i_max - i_min)

    '''
    Apply non-maximum thresholding using dilation. 
    Comparing the dilated image to the Harris image will preserve
    only those locations that are peaks.
    '''
    max_dist = 2 * sigma
    kernel = np.ones((2 * max_dist + 1, 2 * max_dist + 1), np.uint8)
    im_harris_dilate = cv2.dilate(im_harris.astype(np.float32), kernel)
    im_harris[np.where(im_harris < im_harris_dilate)] = 0

    '''
    Get the normalized Harris measures of the peaks
    '''
    peak_values = im_harris[np.where(im_harris > 0)]
    peak_values = np.sort(peak_values, axis=None)
    '''  Extract all indices '''
    indices = np.where(im_harris > 0)
    ys, xs = indices[0], indices[1]  # rows and columns

    ''' Put them into the keypoint list '''
    kp_size = 4 * sigma
    harris_keypoints = [
        cv2.KeyPoint(xs[i], ys[i], _size=kp_size, _response=im_harris[ys[i], xs[i]])
        for i in range(len(xs))
    ]
    harris_keypoints = sorted(harris_keypoints, key=lambda x: x.response)
    if len(harris_keypoints) > 200:
        harris_keypoints = harris_keypoints[-200:]
    out_im = cv2.drawKeypoints(img.astype(np.uint8), harris_keypoints, None)
    save_image_as(out_im, filename, ext)
    return harris_keypoints


def save_image_as(img, name, ext):
    """
    Saves an image with name.ext. Then prints that info.
    """
    file_mod = name + ext
    cv2.imwrite(file_mod, img)


def ORB(img, filename, ext):
    """
    Ripped directly from the class code
    """
    num_features = 1000
    orb = cv2.ORB_create(num_features)  # See method doc for other parameters
    kp, des = orb.detectAndCompute(img, None)  # The None argument is where a binary mask could be
    '''
    We remove any entries in kp with a size of 45 or greater.
    We sort it by the response attribute.
    '''
    kp = [k for k in kp if k.size <= 45]
    kp = sorted(kp, key=lambda x: x.response)
    if len(kp) > 200:
        kp = kp[-200:]
    out_im = cv2.drawKeypoints(img, kp, None)
    save_image_as(out_im, filename, ext)
    return kp


def print_n_resp(kpoints, n):
    """
    prints the last n responses of kpoints.
    kpoints is a list of keypoints, n is an integer.
    """
    for i in range(n):
        resp = kpoints[n-i-1].response
        x = kpoints[n-i-1].pt[0]
        y = kpoints[n-i-1].pt[1]
        print(format(i) + ": (" + format(x, '.1f') + ", " + format(y, '.1f') + ") " + format(resp, '.4f'))


def dist(pt1, pt2, n=2):
    """
    Computes the n norm of the distance between 2 points.
    By default n = 2 the euclidean distance.
    """
    return ((pt1[0]-pt2[0])**n + (pt1[1]-pt2[1])**n)**(1/n)


def dist1to2(kp1, kp2):
    """
    Get the distances from the min(100, len(kp1)) entries of kp1
    to that min(200, len(kp2)) entries of kp2 both by euclidean distance
    and the rank distance. Note that we take the last entries because they are
    in ascending order. I prefer things to be stored in ascending order if an ordered list
    is to be passed. By all means descended order would be less verbose and wouldn't need reversals.
    """
    if len(kp1) >= 100:
        kp1 = kp1[-100:]
    if len(kp2) >= 200:
        kp2 = kp2[-200:]
    kp1 = kp1[::-1]
    kp2 = kp2[::-1]
    a = np.zeros(len(kp2))
    min_dist = np.zeros(len(kp1))
    rank_dist = np.zeros(len(kp1))
    for k in range(len(kp1)):
        for i in range(len(kp2)):
            a[i] = dist(kp1[k].pt, kp2[i].pt)
        min_dist[k] = np.min(a)
        rank_dist[k] = abs(k-np.where(a == min_dist[k])[0][0])
    return min_dist, rank_dist


if __name__ == "__main__":
    """
        Deal with command line arguments assuming they are correct.
        Then drive the rest of part 1.
        Get the keypoints then calculate the distances and output.
    """
    sigma = int(sys.argv[1])
    img_name = sys.argv[2]
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    file_name, file_ext = os.path.splitext(img_name)
    i_harris = file_name + "_harris"
    i_orb = file_name + "_orb"
    harris_kp = harris(img, sigma, i_harris, file_ext)
    print("\nTop 10 Harris keypoints:")
    print_n_resp(harris_kp[-10:], 10)
    orb_kp = ORB(img, i_orb, file_ext)
    print("\nTop 10 ORB keypoints:")
    print_n_resp(orb_kp[-10:], 10)
    print("\nHarris keypoint to ORB distances:")
    min_dist, rank_dif = dist1to2(harris_kp, orb_kp)
    med1 = np.median(min_dist)
    mean1 = np.mean(min_dist)
    med2 = np.median(rank_dif)
    mean2 = np.mean(rank_dif)
    print("Median difference: " + format(med1, '.1f'))
    print("Average difference: " + format(mean1, '.1f'))
    print("Median index difference: " + format(med2, '.1f'))
    print("Average index difference: " + format(mean2, '.1f'))
    print("\nORB keypoint to Harris distances:")
    min_dist, rank_dif = dist1to2(orb_kp, harris_kp)
    med1 = np.median(min_dist)
    mean1 = np.mean(min_dist)
    med2 = np.median(rank_dif)
    mean2 = np.mean(rank_dif)
    print("Median difference: " + format(med1, '.1f'))
    print("Average difference: " + format(mean1, '.1f'))
    print("Median index difference: " + format(med2, '.1f'))
    print("Average index difference: " + format(mean2, '.1f'))


