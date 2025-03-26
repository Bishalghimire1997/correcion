import os
import cv2
import random
import numpy as np
class preprocessing():
    def read_video(self,path,frame_index,max_interval = 2):
        #inatilizing refrence and targer frame index
        
        #inatilizing video capture 
        cap = cv2.VideoCapture(path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #capturing the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret,frame0 = cap.read()

        #randomly capturing two more frames up 2 seconds
        max_frame_offset= min(max_interval*fps,total_frames-1)


        frame1_idx = random.randint(frame_index + 1, min(frame_index + max_frame_offset, total_frames - 1))
        frame2_idx = random.randint(frame_index + 1, min(frame_index + max_frame_offset, total_frames - 1))
    
        # Ensure unique frame selection and order
        frame1_idx, frame2_idx = sorted(set([frame1_idx, frame2_idx]))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
        ret, frame1 = cap.read()

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
        ret, frame2 = cap.read()

        blue_channel = frame0[:, :, 0]  # Blue from frame0
        green_channel = frame1[:, :, 1]  # Green from frame1
        red_channel = frame2[:, :, 2]  # Red from frame2

        refrence = cv2.merge([blue_channel, green_channel, red_channel])
        target = frame0

        return [refrence,target]



    def run(self):
        path = "vid.mp4"
        refrence,target= self.read_video(path,frame_index=20,max_interval=2)
        self.display(refrence)


        


        pass
    def resize_frame(self,len,wid,image):
        size=(wid,len)
        return cv2.resize(image,size,interpolation=cv2.INTER_CUBIC)
    def display(self,image):
        cv2.imshow("Image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
obj = preprocessing()
obj.run()
        

