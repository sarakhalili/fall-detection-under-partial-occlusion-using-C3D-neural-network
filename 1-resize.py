# this code tries to convert the fall and not-fall videos to the desired form which is 20*15-sized-frame videos, then it saves the new movies in another folder

import numpy as np 
import cv2
import os; import glob
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def resize(first_folder,target_folder):

       	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
       	vid_num=0
       
        videos = glob.glob(first_folder+"/*.avi")     #reading videos of the first folder
        videos.sort(key=natural_keys)
        for i,vid in (enumerate(videos)):
           vid_num+=1           
           cap = cv2.VideoCapture(vid)
           num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
           framerate = cap.get(cv2.CAP_PROP_FPS)
           out = cv2.VideoWriter('{}/video_{}.avi'.format(target_folder,vid_num),fourcc, framerate, (112,112),True)   #creating the resized video
           for i in range(0, num_frame+1):
               ret, frame=cap.read()
               if ret==True:
                   if i==0:
                        r,c,_ = frame.shape # here the biggest square in the middle of frame is cropped
                        d = (c-r)//2
                        sc = d
                        ec = r+d
                   resizedfram = cv2.resize(frame[:, sc:ec], (112,112), interpolation = cv2.INTER_AREA)
                   out.write(resizedfram)
               else:
                   out.release()
                   break
#-------------------------resize data set with the defined function named resize which receives the lower part files path and resize the movies in the directory to 15*20 movies in the new directory path which also should be received as one of the inputs------------------------------
# resize(r"/content/drive/MyDrive/openpose/circle/org/Coffee_room_01",r"/content/drive/MyDrive/openpose/circle/original/Coffee_room_01")
