# -*- coding: utf-8 -*-
import cv2
import os
import glob

def compare2images(imag1,imag2,folder_path):
    """ comparing two images base on common good point
    
    Input     img1,img2 String who is image name like "frame0.jpg", "frame1.jpg",....
              folder_path String who is folder where are images
              
    Output    an integer who is the number of common good pixels points
    """
    
    img1 = cv2.imread(folder_path +'/'+imag1)
    img2 = cv2.imread(folder_path +'/'+imag2)
    t = 1
    sift = cv2.xfeatures2d.SIFT_create()
    try:
     kp1,desc1 = sift.detectAndCompute(img1,None)
     kp2,desc2 = sift.detectAndCompute(img2,None)
    except :
     t = 0
    if t :
        index_param = dict(algorithm=0,trees=5)
        search_param = dict()
        flann = cv2.FlannBasedMatcher(index_param,search_param)
    
        matches = flann.knnMatch(desc1,desc2,k=2)
    
        good_points = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_points.append(m)
        return len(good_points)
    else :
        pass  
def deleteSimilarImage(folderpath):
    """ For each image in folder, delete all common images
    
    Input  folderpath String who is the path of images folder
    """
    images = [os.path.basename(img) for img in glob.iglob(f"{folderpath}/*.jpg")]
    for image in images:
        l = images
        l.remove(image)
        for image_ in l:
            # We will assume that two images are similar if there have more than 100
              # common good points
            try :
               if compare2images(image,image_,folderpath) > 100: 
                 os.remove(folderpath + '/' + image_)
            except :
                pass

def main_():
 path_ = "data/image"
 glob_ = glob.iglob(f"{path_}/*")
 sequences = [os.path.basename(seq) for seq in glob_] 

 for sequence in sequences :
    folderpath = path_ + "/" + sequence
    deleteSimilarImage(folderpath)              