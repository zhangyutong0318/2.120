import cv2
import numpy as np
from pupil_apriltags import Detector

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

### Some utility functions to simplify drawing on the camera feed
# draw a crosshair
def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

# plot a little text
def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 3)



cap=cv2.VideoCapture(1)  #camera used
detector = Detector(families='tag36h11', 
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0,
                    ) #physical size of the apriltag

intrisic = [500,500,960,540] # camera parameters, [fx, fy cx, cy]
tagsize = 0.036  #physical size of printed tag, unit = meter

looping = True

while (looping):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=True, camera_params=intrisic, tag_size=tagsize)
    
    if not tags:
        print("Nothing")
    else:
        for tag in tags:
            print(tag.pose_R)
            frame = plotPoint(frame, tag.center, CENTER_COLOR)
            frame = plotText(frame, tag.center, CENTER_COLOR, tag.pose_R)
            for corner in tag.corners:
                frame = plotPoint(frame, corner, CORNER_COLOR)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1000) #ms
	# terminate the loop if the 'Return' key is hit
    if key == 13:
        looping = False

cv2.destroyAllWindows()


