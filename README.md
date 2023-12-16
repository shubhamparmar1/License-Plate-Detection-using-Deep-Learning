# Realtime-License-Plate-Detection-using-Deep-Learning (Indian Scenario)

This project presents a custom model for real-time license plate detection, specifically tailored for the complex scenarios encountered in India. The model is trained to recognize six classes: license plates, cars, motorcycles, buses, trucks, and rickshaws.

## Model Training

- The model was trained on <a href="https://colab.research.google.com"><img src="https://cdn.jsdelivr.net/gh/shubhamparmar1/License-Plate-Detection-using-Deep-Learning/.config/colab1.png" alt="Google Colab" width = 90px></a>. 
- It is based on the YOLOv5m custom architecture and implemented in Python using the PyTorch library.
- The custom dataset for Indian license plate made using <a href="https://roboflow.com/"><img src="https://cdn.jsdelivr.net/gh/shubhamparmar1/License-Plate-Detection-using-Deep-Learning/.config/roboflow1.png" alt="roboflow" width = 80px></a>. 

For real-time operation, the system is set up to utilize GPU performance with CUDA. The system is designed to detect and locate license plates and recognize their characters simultaneously using EasyOCR. This is achieved by modifying the `detect.py` file and creating a new `detect1.py` file.

## Running the Model

### Prerequisites

- Python
- PyTorch
- CUDA for GPU acceleration
- EasyOCR

### Installation

1. Clone the repository:

```bash
git clone https://github.com/shubhamparmar1/License-Plate-Detection-using-Deep-Learning.git
cd License-Plate-Detection-using-Deep-Learnin
```
2. Clone YOLOv5:

- You can see that in yolov5 folder `detect1.py` file is there. In this folder we have to clone yolov5 repo from [Ultralytics](https://github.com/ultralytics/yolov5).
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```
3. Running the Model:

- You can run the model using the following command:
```bash
python detect1.py --weights ../best.pt --img 416 --conf 0.4 --source 0
```
This command utilizes our custom model weights (best.pt) and EasyOCR to perform real-time detection on a live video stream. Adjust parameters as needed.


## Result

<img alt="Result" width="400" src="https://cdn.jsdelivr.net/gh/shubhamparmar1/License-Plate-Detection-using-Deep-Learning/.config/result1.gif">

The image above showcases the successful detection and localization of license plates in real-time.


