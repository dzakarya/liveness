from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import math
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth): 
    A = dist.euclidean(mouth[3], mouth[9]) 
    B = dist.euclidean(mouth[2], mouth[10]) 
    C = dist.euclidean(mouth[4], mouth[8])
    L = (A+B+C)/3 
    D = dist.euclidean(mouth[0], mouth[6]) 
    mar=L/D 
    return mar

EYE_AR_THRESH = 0.2
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)
mouth_status = ""
left_eye_status = ""
rigth_eye_status = ""
head_status = ""
eth = 0.2
muth = 0.5
mlth = 0.3
hdth = 10

def angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))

while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 1)
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        lEye_Coor = shape[lStart]
        rEye_Coor = shape[rStart]
        normStart = lEye_Coor
        normEnd = (rEye_Coor[0],normStart[1])

        # calculate deg
        x = abs(normEnd[0]-normStart[0])
        z = abs(rEye_Coor[1]-normStart[1])
        y = math.sqrt((x*x)+(z*z))
        if normEnd[1] > rEye_Coor[1]:
            headDeg = math.degrees(math.acos(x/y)) 
        else:
            headDeg = -math.degrees(math.acos(x/y)) 
        cv2.line(frame,rEye_Coor,lEye_Coor,(0,255,0),1)
        cv2.line(frame,normStart,normEnd,(0,0,255),1)

        LEAR = eye_aspect_ratio(leftEye)
        REAR = eye_aspect_ratio(rightEye)
        MAR = mouth_aspect_ratio(mouth)
        
        if MAR < mlth:
            mouth_status = "smiling face"
        elif MAR > muth:
            mouth_status = "open"
        else:
            mouth_status = "normal"

        if LEAR < eth:
            left_eye_status = "closed"
        else:
            left_eye_status = "open"

        if REAR < eth:
            rigth_eye_status = "closed"
        else:
            rigth_eye_status = "open"

        if headDeg < -hdth:
            head_status = "miring kanan"
        elif headDeg > hdth:
            head_status = "miring kiri"
        else:
            head_status = "normal"
        # average the eye aspect ratio together for both eyes
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "LEAR: {:.2f}".format(LEAR), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "REAR: {:.2f}".format(REAR), (150, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(MAR), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Head_Degree: {}".format(headDeg), (450, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Mouth: {}".format(mouth_status), (450, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
        cv2.putText(frame, "Left_Eye: {}".format(left_eye_status), (450, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
        cv2.putText(frame, "Rigth_Eye: {}".format(rigth_eye_status), (450, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
        cv2.putText(frame, "Head: {}".format(head_status), (450, 230),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2)
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        cv2.imwrite("smilling.jpeg",frame)
        break
cv2.destroyAllWindows()
vs.stop()