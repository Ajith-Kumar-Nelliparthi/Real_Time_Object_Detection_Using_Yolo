import cv2
import numpy as np
import time
import os

# Load YOLO
net = cv2.dnn.readNet("weight/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load video
video_path = "usa-street.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    exit()

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5)) if cap.get(5) > 0 else 20  # Use 20 FPS if unavailable

print(f"Video Resolution: {frame_width}x{frame_height}, FPS: {fps}")


if frame_width == 0 or frame_height == 0:
    print("Error: Invalid frame dimensions!")
    cap.release()
    exit()

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
output_video_path = "output/yolo_output.mp4"
os.makedirs("output", exist_ok=True)  

out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


if not out.isOpened():
    print("Error: VideoWriter failed to open!")
    cap.release()
    exit()

print("VideoWriter initialized successfully.")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while cap.isOpened():  
    ret, frame = cap.read()
    
    if not ret:  
        print("End of video or error reading frame.")
        break  

    frame_id += 1
    height, width, _ = frame.shape

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Store detected objects
    class_ids = []
    confidences = []
    boxes = []

    for out_layer in outs:
        for detection in out_layer:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    if len(indexes) > 0:
        indexes = indexes.flatten()

    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {round(confidences[i] * 100, 2)}%"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), font, 2, color, 2)

    # Display FPS
    elapsed_time = time.time() - starting_time
    fps_display = frame_id / elapsed_time
    cv2.putText(frame, f"FPS: {round(fps_display, 2)}", (10, 50), font, 2, (0, 0, 0), 3)

    
    if out.isOpened():
        out.write(frame)  
    else:
        print("Warning: Video writer is not open.")

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break loop if 'ESC' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved at: {output_video_path}")
