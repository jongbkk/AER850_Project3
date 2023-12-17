import os
import numpy as np
import cv2
from ultralytics import YOLO

# Loading the dataset
train_dir = "C:\\Users\\user\\OneDrive\\Documents\\AER850\\data3\\train\\images"
eval_dir = "C:\\Users\\user\\OneDrive\\Documents\\AER850\\data3\\evaluation\\images"

# Loading the YOLOv8 model
model = YOLO('yolov8s.pt')

# Define hyperparameters
epochs = 150
batch = 16
img_size = 1200

# Train the model
model.train(train_dir, epochs=epochs, batch=batch, imgsz=img_size)

# Evaluate the model
img1 = cv2.imread(os.path.join(eval_dir, 'image1.jpg'))
img2 = cv2.imread(os.path.join(eval_dir, 'image2.jpg'))
img3 = cv2.imread(os.path.join(eval_dir, 'image3.jpg'))

# Run object detection
outputs = model.predict(img1)
outputs = model.predict(img2)
outputs = model.predict(img3)

# Print the detection results
print('Image 1:')
print(outputs[0])
print('Image 2:')
print(outputs[1])
print('Image 3:')
print(outputs[2])

# Analyze the results
img1_detected_components = []
img2_detected_components = []
img3_detected_components = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id in np.argwhere(scores > 0.5):
            img1_detected_components.append(detection[0])
            img2_detected_components.append(detection[0])
            img3_detected_components.append(detection[0])

print('Image 1: Detected components:', img1_detected_components)
print('Image 2: Detected components:', img2_detected_components)
print('Image 3: Detected components:', img3_detected_components)

# Summarize the model's performance
accuracy = 0
for image in img1_detected_components + img2_detected_components + img3_detected_components:
    accuracy += 1
accuracy /= 3
print('Model accuracy:', accuracy)
