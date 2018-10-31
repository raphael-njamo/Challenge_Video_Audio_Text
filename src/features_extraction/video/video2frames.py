# -*- coding: utf-8 -*-
import os
import cv2
import glob

def video_to_frames(videopath, framepath, videofps):
    """ segmentation a video to frames
    Input  videopath a string, path who is the video
           framepath a string, path who will store frames
           videofps an integer, represente the number of segmentation
    Output images with format jpg 
    """
    # read video
    vidcap = cv2.VideoCapture(videopath)
    
    # read next frame
    success,image = vidcap.read()
    
    # create the output folder to store frames
    try: 
        if not os.path.exists(framepath): 
            os.makedirs(framepath) 
    except OSError: 
        print ('Error: Creating directory of data') 

    count = 0
    while success:
        # save frame as JPEG file 
        cv2.imwrite(f"{framepath}/frame%d.jpg" % count, image)
        i=0
        # read all frames of the current second (only the last will be saved)
        for i in range(videofps):
            success,image = vidcap.read()
        count += 1
        


def parse_videos():
    videopath = "data/video"
    imagepath = "data/image"

    for vp in glob.iglob(f"{videopath}/*.mp4"):
        moviename = os.path.basename(vp)
        moviename = os.path.splitext(moviename)[0]
        
        moviefolder = f"{imagepath}/{moviename}"
        
        try: 
            if not os.path.exists(moviefolder): 
                os.makedirs(moviefolder) 
        except OSError: 
            print ('Error: Creating directory of data') 
        video_to_frames(vp, moviefolder,30)
