# importing necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


""" Constructing the argument parsers """

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default = "face_detector",
        help = "path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default = "mask_detector.model",
        help = "path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default = 0.5,
        help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # passing the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initializing our list of faces their corresponding locations and list of
    # predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over detections
    for i in range(0, detections.shape[2]):
        # extract confidence(probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weaak detections by ensuring the confidence is greater
        # than minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y) coordinates of bounding box for object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering
            # resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # adding face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

        # only make predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we make batch predictions on *a;;*
            # faces at the same time rather than one-by-one preictions
            # in the above for loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # returns a 2 - tuple of the face locations and their corresponding
        # locations
        return (locs, preds)


# loading our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initializing the video stream and allow the camera_sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)



""" Loop over frames in stream """

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we will use to draw bouding box
        # and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # displate the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show ouput frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from loop
        if key == ord("q"):
            break

# cleaning up
cv2.destroyAllWindows()
vs.stop()
