import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math

def ORB(img, filename, ext):
    """
    Ripped directly from the class code
    """
    num_features = 500
    orb = cv2.ORB_create(num_features)  # See method doc for other parameters
    kp, des = orb.detectAndCompute(img, None)  # The None argument is where a binary mask could be
    '''
    We remove any entries in kp with a size of 45 or greater.
    We sort it by the response attribute.
    '''

    out_im = cv2.drawKeypoints(img, kp, None)
    save_image_as(out_im, filename, ext)
    return kp


def orb_match(i1, i2):
    num_features = 500
    orb = cv2.ORB_create(num_features)
    kp1, dsc1 = orb.detectAndCompute(i1, None)
    kp2, dsc2 = orb.detectAndCompute(i2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    naive_matches = bf.match(dsc1, dsc2)
    naive_matches = sorted(naive_matches, key=lambda x:x.distance)
    #i_pair = cv2.drawMatches(i1, kp1, i2, kp2, naive_matches[:10], None)
    match_ratio = len(naive_matches)/len(kp1)
    #plt.imshow(i_pair)
    #plt.show()
    if match_ratio > .1:
        pts1 = []
        pts2 = []
        for i in naive_matches:
            pts1.append(kp1[i.queryIdx].pt)
            pts2.append(kp2[i.trainIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        f_mat, inliers = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]
        i_pair = cv2.drawMatches(i1, kp1, i2, kp2, naive_matches, None,  matchesMask=inliers.ravel().tolist())
        #plt.imshow(i_pair)
        #plt.show()
        inlier_ratio = np.count_nonzero(inliers.ravel())/len(naive_matches)
        print("Number of Inliers from Fund. Matrix " + format(np.count_nonzero(inliers.ravel())))
        print("Kept from Fundamental Matrix with Ransac: " + format(inlier_ratio*100, '.1f') + "%")
        #print("The % of matches left as inliers is " + format(inlier_ratio*100, '.3f'))
        in_thresh = .1
        if inlier_ratio > in_thresh:
            pts1 = []
            pts2 = []
            for i in naive_matches:
                pts1.append(kp1[i.queryIdx].pt)
                pts2.append(kp2[i.trainIdx].pt)
            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            h_mat, inliers_h = cv2.findHomography(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
            i_pair = cv2.drawMatches(i1, kp1, i2, kp2, naive_matches, None, matchesMask=inliers_h.ravel().tolist())
            plt.imshow(i_pair)
            plt.show()
            pts1 = pts1[inliers_h.ravel() == 1]
            pts2 = pts2[inliers_h.ravel() == 1]
            h_mat_inv = np.linalg.inv(h_mat)
            #if h_mat_inv[2, 2] < 0:
                #h_mat_inv = h_mat_inv * -1
            #h_mat_inv = h_mat_inv / np.linalg.norm(h_mat_inv)
            print("Number of Inliers From Homography:" + format(np.count_nonzero(inliers_h)))
            #i_pair = cv2.drawMatches(i1, kp1, i2, kp2, naive_matches, None, matchesMask=inliers.ravel().tolist())
            #plt.imshow(i_pair)
            #plt.show()
            print(h_mat_inv)
            UL = h_mat_inv @ [0, 0, 1]
            UL = UL/UL[2]
            x_offset = abs(UL[0])
            y_offset = abs(UL[1])
            x_o = int(x_offset)
            y_o = int(y_offset)
            LR = h_mat_inv @ [i2.shape[1], i2.shape[0], 1]
            LR = LR/LR[2]
            LL = h_mat_inv @ [i2.shape[1], 0, 1]
            LL = LL/LL[2]
            UR = h_mat_inv @ [0, i2.shape[0], 1]
            UR = UR/UR[2]
            y_coords = np.array([UL[1], UR[1], LL[1], LR[1], i1.shape[1], 0])
            x_coords = np.array([UL[0], UR[0], LL[0], LR[0], i1.shape[0], 0])

            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            h_mat[0, 2] += abs(min_x)+500
            h_mat[1, 2] += abs(min_y)+500
            new_size = (int(LR[0])+x_o, int(LR[1])+y_o)
            #pan = cv2.warpPerspective(i1, h_mat_inv, (i1.shape[1]+i2.shape[1], i1.shape[0]))
            pan = cv2.warpPerspective(i2, h_mat_inv, (i1.shape[1]+i2.shape[1], i1.shape[0]+i2.shape[0]))
            print(UL)
            print(UR)
            print(LL)
            print(LR)
            print(h_mat_inv)
            pan[0:i1.shape[0], 0:i1.shape[1]] = i1
            #plt.imshow(i2)
            #plt.show()
            #pan = cv2.warpPerspective(pan, h_mat_inv, (i1.shape[1] + i2.shape[1], i2.shape[0]))
            #pan[0:i1.shape[0], 0:i1.shape[1]] = i1
            plt.imshow(pan)
            plt.show()
            return h_mat
    return None






def save_image_as(img, name, ext):
    """
    Saves an image with name.ext. Then prints that info.
    """
    file_mod = name + ext
    cv2.imwrite(file_mod, img)


if __name__ == "__main__":
    """
        Deal with command line arguments assuming they are correct.
        Then drive the rest of part 1.
        Short Version: Create Image Mosaic of Images that appear the same
        Long Version: To Be Written 11/11/2019
        
    """
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    os.chdir(in_dir)
    img_list = os.listdir('./')
    img_list.sort()
    for i in range(len(img_list)):
        for j in range(len(img_list)):
            if j > i:
                img1 = cv2.imread(img_list[i])
                img2 = cv2.imread(img_list[j])
                im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                H = orb_match(im1, im2)
                if H is not None:
                    pan = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))
                    pan[0:img2.shape[0], 0:img2.shape[1]] = img2
                    H = np.linalg.inv(H)
                    pan = cv2.warpPerspective(pan, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))
                    pan[0:img1.shape[0], 0:img1.shape[1]] = img1
                    name1, ext = os.path.splitext(img_list[i])
                    name2, ext = os.path.splitext(img_list[j])
                    filename = os.path.dirname(os.getcwd()) + "\\" + out_dir + "\\" + name1 + "_" + name2 + ext
                    print(filename)
                    cv2.imwrite(filename, pan)
