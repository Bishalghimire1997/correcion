import os
import cv2
import random
import numpy as np
class Data_generation():
    def __init__(self,path):
      self.path = path
      self.cap = cv2.VideoCapture(self.path)
      self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
      self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      print("Total Frames = ",self.total_frames)
      print("frame rate = ",self.fps)

    def read_video(self,frame_index,max_interval = 0.5):
        #inatilizing refrence and targer frame index

        #inatilizing video capture


        #capturing the target frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret,frame0 = self.cap.read()

        #randomly capturing two more frames up 2 seconds
        max_frame_offset= int(min(max_interval*self.fps,self.total_frames-1))


        frame1_idx = random.randint(frame_index + 1, min(frame_index + max_frame_offset, self.total_frames - 1))
        frame2_idx = random.randint(frame_index + 1, min(frame_index + max_frame_offset, self.total_frames - 1))

        # Ensure unique frame selection and order
        if frame1_idx == frame2_idx:
            frame1_idx += 1

        frame1_idx, frame2_idx = sorted(set([frame1_idx, frame2_idx]))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
        ret, frame1 = self.cap.read()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
        ret, frame2 = self.cap.read()

        blue_channel = frame0[:, :, 0]  # Blue from frame0
        green_channel = frame1[:, :, 1]  # Green from frame1
        red_channel = frame2[:, :, 2]  # Red from frame2

        refrence = cv2.merge([blue_channel, green_channel, red_channel])
        target = frame0

        return [self.resize_frame(128,128,refrence),self.resize_frame(128,128,target)]



    def run(self):
        path = "vid.mp4"
        refrence,target= self.read_video(path,frame_index=20,max_interval=0.3)
        self.display(refrence)



    def get_n_samples(self,n=100):
      refrence_list = []
      target_list = []
      for i in range(n):
        if(i%100 == 0):
           print(i)
        ref,target = self.read_video(frame_index=i+25)
        refrence_list.append(ref)
        target_list.append(target)
      return (refrence_list,target_list)

    def emulate_lower_lighting(self):
      pass

    def resize_frame(self,len,wid,image):
        size=(wid,len)
        return cv2.resize(image,size,interpolation=cv2.INTER_CUBIC)

    def display(self,image):
        cv2.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
