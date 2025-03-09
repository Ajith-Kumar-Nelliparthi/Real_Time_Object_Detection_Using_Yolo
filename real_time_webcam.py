# import libraries
import cv2
import numpy as np 
import os

weights_path = os.path.abspath("weight/yolov3.weights")
cfg_path = os.path.abspath("cfg/yolov3.cfg")
# load yolo
net = cv2.dnn.readNet(weights_path, cfg_path)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam")
else:
    print("Webcam opened successfully")

while True:
    # read from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam")
        break
    # detecting objects
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # show information on the detected objects
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)
    # display the resulting frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break

cap.release()
cv2.destroyAllWindows()