'''
Tracking particle using opencv
'''
# To detect a particle using opencv colour based threshold and then its trajectory using deque

# importing libraries 
import cv2
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import datetime

# constructing the argument parse and then parsing the arguments
# here a video (already recorded) will be given in as an argument wherein the particle will be detected
# buffer is the max size of the list of the tracked points 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# defining the lower and upper boundaries of the particle color (here blue) in the HSV color space
Lower = (100, 50, 50)
Upper = (140, 255, 255)

# initialize the list of tracked points
pts = deque(maxlen=args["buffer"])
T = deque(maxlen=args["buffer"])

# if a video path was not supplied, then grabbing the reference to the webcam using imutils.video 
if not args.get("video", False):
    vs = VideoStream(src=0).start()
# otherwise, grabbing a reference to the video file using opencv
else:
    vs = cv2.VideoCapture(args["video"])
    
# allow the camera or video file to warm up
time.sleep(2.0)

# Now, keep looping on frames (infinite loop till the key q is pressed or we reach the end of video)

while True:
    t = datetime.datetime.now()
    # grabbing the current frame
    frame = vs.read() # returns a tuple containing (a boolean indicating if the frame was successfully read or not, video frame)
    # handling the frame from VideoCapture or VideoStream implementation
    frame = frame[1] if args.get("video", False) else frame
    # if we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if frame is None:
        break

    #pre-processing our frame
    # resizing the frame(down-sizing makes processing of the frame faster, hence increases FPS), blurring it(to reduce high frequency
    # noise allowing us to focus on the structural objects inside the frame here particle), and converting it to the HSV
    # color space(Hue Saturation Value is the best form for the image for color detection)
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # constructing a mask for the color "blue" and performing a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, Lower, Upper)#actual localization of the blue particle
    #erosion and dilation helps to eliminate irrelevant details
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # finding contours (an outline of the detected object) in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)# making compatible with all versions of opencv
    
    # initializing the current (x, y) center of the ball
    center = None

    # drawing two circles: one surrounding the ball itself and another to indicate the centroid of the ball.
    # proceeding only if at least one contour was found (only if the blue particle was detected)
    if len(cnts) > 0:
        # finding the largest contour in the mask, then using it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # proceeding only if the radius meets a minimum size
        if radius > 10:
            # drawing the circle and centroid on the frame, then updating the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
    # updating the points queue (appending the centroid to the list)
    pts.appendleft(center)
    T.appendleft(t)
    # drawing the contrail of the ball, the past N (x, y)-coordinates the ball has been detected at
    # looping over the set of tracked points
    for i in range(1, len(pts)):
        # ignoring the tracked points that are None 
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, computing the thickness of the line and drawing the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        
    # showing the frame to our screen
    cv2.imshow("Detecting particle trajectory", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stopping the loop
    if key == ord("q"):
        break
    
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
    
# closing all windows
cv2.destroyAllWindows()

# printing the trajectory of particle 
print(pts)
print(' ')
print(T)

