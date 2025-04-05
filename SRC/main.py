
import cv2
import DataGeneration
import ImageDataset
import model
import Preprocessing
from train import TrainModel
class main():
   def __init__(self):
      pass
   def run(self):
      pre = Data_generation("vid.mp4")
      train_images,target_images = pre.get_n_samples(1500)
      cv2.imshow(train_images[0])
      cv2.imshow(target_images[0])

      preprocess =Preprocessing()
      train = preprocess.preprocessing(train_images)
      target = preprocess.preprocessing(target_images)
      preprocess.show_n_images(train,n=4)
      preprocess.show_n_images(target,n=4)

      mod = Model()
      mod.display_n_image_with_noise(train_images,4)
      mod.display_n_image_without_noise(train_images,4)

      train_m = TrainModel(train, target)
      train_m.train()
obj = main()
obj.run()







      
      
