from realsense_pipeline import realsense_pipeline
import cv2

id = int(input("image id: "))
pipeline = realsense_pipeline()
img = pipeline.get_rgb_image()

# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite("%d_ball.jpg"%id, img)

