import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)

import cv2
import numpy as np
import easyocr

cap = cv2.VideoCapture('pexels_videos.mp4')

# Define the desired width and height for the display window
desired_width = 1080
desired_height = 720

# Define the frame skip value
frame_skip = 10  # process every 10th frame

frame_counter = 0

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define the confidence threshold
confidence_threshold = 0.3

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        frame_counter += 1
        if frame_counter % frame_skip == 0:  # only process every nth frame
            # Resize the frame to the desired width and height
            resized_frame = cv2.resize(frame, (desired_width, desired_height))
            
            # Make detections 
            yolo_results = model(resized_frame)
            print("SVP = ",yolo_results)
            # Extract bounding boxes from the results
            boxes = yolo_results.xyxy[0].cpu().numpy()  # adjust this line based on your model's output
            
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box[:4].astype(int)
                
                # Extract region from the frame
                region = resized_frame[y1:y2, x1:x2]
                region  = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                _, region = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Use EasyOCR to detect text
                ocr_results = reader.readtext(region)
                
                for result in ocr_results:
                    text, confidence = result[1], result[2]
                    if confidence > confidence_threshold:
                        print(text)
            
            cv2.imshow('YOLO', np.squeeze(yolo_results.render()))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()