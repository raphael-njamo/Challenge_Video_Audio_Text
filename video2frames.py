# -*- coding: utf-8 -*-
import os
import cv2


'''
Transform a video into frames (one frame per second)
(to go deeper => https://www.linkedin.com/pulse/fun-opencv-video-frames-arun-das/)

videopath : where the video is stored
framepath : where the frames will be stored
videofps : frame per second of the video

TODO: check if we get the first frame of the video
TODO: check if the functions works well with video shorter than 3 seconds
TODO: create a function to get fps of a video
TODO: create a function to iterate on all videos
'''
def video_to_frames(videopath, framepath, videofps):

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
        
# test for the first video
#video_to_frames('corpus_part1/corpus_part1/SEQ_001_VIDEO.mp4','data',30)
