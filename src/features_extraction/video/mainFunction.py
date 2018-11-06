# -*- coding: utf-8 -*-
import os,os.path
import glob
from video2frame_ import video_to_frames, parse_videos
from NewCompare import compare2images, deleteSimilarImage, main_
import shutil
import pathlib 

def select():
    part = "data"
    #imagepath = "data/image"
    videopath = "data/video"
    for Pathpart in glob.iglob(f"{part}/*"):
        if os.path.basename(Pathpart) != "image" and os.path.basename(Pathpart) != "video":
            for seq in  glob.iglob(f"{Pathpart}/*.mp4"):
                try:
                    p = pathlib.Path(seq)
                    newpath = os.path.join(pathlib.Path(*p.parts[0:1]),"video",os.path.basename(seq))
                    shutil.move(seq,newpath)
                    os.makedirs(videopath+"/"+os.path.basename(seq))
                except FileExistsError :
                    pass
    parse_videos()
    main_() 