import realsenseconfig as rs_config  ##初始化参数
import cv2   
import os
import numpy as np
from pynput import keyboard

RECORD = False
END = False
def on_press(key):
      if key == keyboard.Key.esc:
          global END 
          END = True
          return False  # stop listener
      if key == keyboard.Key.space:
          global RECORD 
          RECORD = not RECORD
          
    

if __name__ == '__main__':
# print(rs_config.D435_para.color_to_depth_translation)
    rs_config.pipeline.start(rs_config.config)
    i = 0
    import shutil
    shutil.rmtree('./dataset_unlabeled')
    os.mkdir('./dataset_unlabeled')
    cv2.namedWindow('RealSenseRGB', cv2.WINDOW_AUTOSIZE)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    while not END:
      rs_config.D435_para.refresh_mat()
      img = rs_config.D435_para.colormat
      if RECORD:
          cv2.imwrite('./dataset_unlabeled/{}.jpeg'.format(i), img.squeeze())
          i += 1
          print(i)
      cv2.imshow('RealSenseRGB',img)
      cv2.waitKey(1)
    listener.join()