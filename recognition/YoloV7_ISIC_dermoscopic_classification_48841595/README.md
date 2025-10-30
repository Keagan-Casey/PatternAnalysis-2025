Lesion Detection and Classification using YOLOv7  
COMP3710 – Pattern Recognition and Analysis  
Author: Keagan Casey (48841595)  
Semester: 2, 2025  

Project Overview  
This project focuses on detecting and classifying skin lesions in dermoscopic images using the **ISIC 2017/2018 dataset**. The goal is to develop a robust computer vision pipeline that can **localise** lesion regions and **classify** them into diagnostic categories with high accuracy. The implemented algorithm is based on **YOLOv7**, a state-of-the-art object detection model, which is fine-tuned for medical image analysis. The system aims to achieve a minimum **Intersection over Union (IoU)** of **0.8** for all detections on the test set, while maintaining strong classification accuracy across lesion types.


How It Works:  
The YOLOv7 architecture divides each input image into a grid and predicts bounding boxes and class probabilities for each cell. For this project, the model was fine-tuned on the ISIC dermoscopic dataset using transfer learning — leveraging pretrained weights on COCO to accelerate convergence. The model outputs both **bounding boxes** around lesions and **classification scores** for lesion types.  
A custom data preprocessing pipeline standardises image sizes, normalises pixel intensities, and applies data augmentation (random rotations, brightness changes, and flips) to improve generalisation. Predictions are filtered based on a confidence threshold, and bounding boxes are merged via **Non-Maximum Suppression (NMS)** to ensure that only high-IoU detections remain.


Visualisation:  
Below is a visual representation of the lesion detection pipeline:

INSERT IMAGE


Dependencies:  
| Package | Version | Purpose |
|----------|----------|----------|
| Python | 3.10+ | Core language |
| PyTorch | 2.0.0 | Deep learning framework |
| OpenCV | 4.8.1 | Image processing and visualization |
| NumPy | 1.26 | Numerical operations |
| Matplotlib | 3.8 | Plotting and evaluation visuals |
| ONNXRuntime | 1.17 | Model inference for YOLOv7 ONNX format |
| scikit-learn | 1.4 | Metrics and data splitting |
| tqdm | 4.66 | Progress tracking |

**Installation Example:**

pip install torch==2.0.0 torchvision==0.15.0
pip install opencv-python==4.8.1 onnxruntime==1.17 numpy==1.26 matplotlib==3.8 scikit-learn==1.4 tqdm==4.66


Preprocessing

- Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
-justify training, validation and testing splits of the data.


Example usuage

-provide example 
