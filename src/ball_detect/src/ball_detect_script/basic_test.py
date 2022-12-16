
from realsense_pipeline import realsense_pipeline
import cv2
import time

pipeline = realsense_pipeline()
cv2.namedWindow('RealSenseRGB', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('RealSenseDepth', cv2.WINDOW_AUTOSIZE)


while True:
    Position = None
    # time_hist.append(t1)
    
    image = pipeline.get_rgb_image()
    cv2.imshow('RealSenseRGB',image)
    Keyvalue = cv2.waitKey(1)
    