								Skin Cancer Detection and Classification using YOLOv7  
COMP3710 – Pattern Recognition and Analysis  
Author: Keagan Casey (48841595)  
Semester 2 2025  

Project Overview
----------------  
This project focuses on detecting skin lesions dermoscopic images using the ISIC 2018 dataset. The goal is to develop a robust computer vision system that can localise lesion regions and classify them accordinly (into 1 of 7 predefined classes) with high accuracy. The implemented model is based on YOLOv7, a pretrained object detection model, which is fine-tuned for dermatological image analysis. 

Problem Definition
------------------

Accurate identification of skin lesions is critical within dermatology, as it directly impacts early diagnosis and treatment of potentially life-threatening skin conditions. However, skin lesions vary widely in appearance, size, colour, and texture, making manual diagnosis challenging. This can lead to incorrect dismissal of a dangerous skin lesion due to an incorrect Benign classifcation.  Automated classification using computer vision can assist dermatologists by providing consistent, rapid, and objective assessments of lesion types. Therefore, this model will be trained on 6 of the most common malignant legions along with benign skin spots from the ISIC 2018 challenge data set, making it a useful tool in the early detection and treatment of malignant skin conditions. To achieve consistantly viable results, this model aims to achieve a miniumum Intersection over Union (IoU) of 0.8 on the test set.


file structures and definitions
-------------------------------
- “modules.py" contains the source code of the components of the model. 
- “dataset.py" contains the data loader for loading and preprocessing the ISIC 2018 data set
- “train.py" containing the source code for training, validating, testing and saving the model. Also plots the losses and metrics during training 
- “predict.py" showing example usage of the trained model. Print out any results and provides visualisations 
- “README.MD” Provides an overview of the project, architecture, and setup instructions to ensure reproducibility.  
- "util" Directory containg important utility files

Data Set
--------

The International Skin Imaging Collaboration (ISIC) 2018 challenge data set contains a total of 11723 images of skin legions and an accompanying 11723 entries within CSV files as ground truth annotations to classify the respective image into a legion type. Within the CSV classification files, the respective class was indicated using a one hot encoding system. The legion types covered within this data set include:

|   Class Index   |   Label   |   Full Name         |   Description                                                                |
|-----------------|-----------|---------------------|------------------------------------------------------------------------------|
| 0               | MEL       | Melanoma            | Malignant tumor arising from melanocytes (a serious form of skin cancer).    |
| 1               | NV        | Melanocytic Nevus   | Common benign mole composed of melanocytes.                                  |
| 2               | BCC       | Basal Cell Carcinoma| Most common form of skin cancer, typically slow-growing and less aggressive. |
| 3               | AKIEC     | Actinic Keratosis   | Precancerous lesion caused by prolonged sun exposure.                        |
| 4               | BKL       | Benign Keratosis    | Group of benign keratinocyte skin lesions.                                   |
| 5               | DF        | Dermatofibroma      | Benign fibrous skin nodule, often firm and small.                            |
| 6               | VASC      | Vascular Lesion     | Lesions related to blood vessels, such as angiomas or hemangiomas.           |

The 11723 images were subdivided into training data, validation data and testing data using a 85: 14: 1 (training : test : val) split as provided from the data source. This split provides a large training subset required for generalisation, with sufficient validation data for tuning and a small test subset for unbiased evaluation.

The images were of type jpg, and were subdivided using the following file structure:

ISIC2018/
├── train/
│   ├── IMG/
│   │   ├── image_0001.jpg
│   │   ├── image_0002.jpg
│   │   └── ... 
│   └── anno/
│       └── train_annotations.csv
├── val/
│   ├── IMG/
│   │   ├── image_08013.jpg
│   │   ├── image_08014.jpg
│   │   └── ... 
│   └── anno/
│       └── val_annotations.csv
└── test/
    ├── IMG/
    │   ├── image_09014.jpg
    │   ├── image_09015.jpg
    │   └── ... 
    └── anno/
        └── test_annotations.csv

The dataset was provided under the licence CC-BY-NC and is avaliable open source at: https://challenge.isic-archive.com/data/#2018







YoloV7 model architecture
-------------------------
The YOLOv7 architecture divides each input image into a grid and predicts bounding boxes and class probabilities for each cell. For this project, the model was fine-tuned on the ISIC dermoscopic dataset using transfer learning which leverages pretrained weights on COCO to accelerate convergence. The model outputs both bounding boxes around lesions and classification scores for lesion types. the provided coordinates for the boundary boxes along with the label from the class with the highest classification socre can then be mapped onto the original image to visualise the model's prediction.

Within YoloV7, A custom data preprocessing pipeline standardises image sizes, normalises pixel intensities, and applies data augmentation (including random rotations, brightness changes, and flips) to improve generalisation.


Application of YoloV7 to ISIC 2018 data set
-------------------------------------------
- why is yolov7 good for this specific data set


Model data preprocessing
------------------------
Each test image is resized to 640x640, converted to RGB, and normalized. These transformations ensure consistency with the model’s expected input.

- Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
-justify training, validation and testing splits of the data.


- to check for an approximately equal class distribution, some python code was developed (found at utils/splitTestTrain), to check count the instances of each class. this revealed a highly disproportional data set as highlighted by: {'MEL': 1305, 'NV': 7737, 'ВСС': 622, 'AKIEC': 378, 'BKL': 1338, 'DF': 160, 'VASC': 180}

- to reduce this effect, the data was split into test/train/val from within each class itself. resultin in the following representation within each test/val/train:

MEL: train=1045, val=130, test=130
NV: train=6191, val=773, test=773
BCC: train=498, val=62, test=62
AKIEC: train=304, val=37, test=37
BKL: train=1072, val=133, test=133
DF: train=128, val=16, test=16
VASC: train=144, val=18, test=18



Model Training
-------------

-built using pyTorch

hyper parameters:

+----------------+------------------------------+--------------------------------------------------------------+
| Parameter      | Value                        | Description                                                  |
+----------------+------------------------------+--------------------------------------------------------------+
| Learning Rate  | 0.001                        | Initial rate with cosine scheduler                           |
| Epochs         | 50                           | Allows convergence without overfitting                       |
| Batch Size     | 16                           | Optimised for GPU memory                                     |
| Optimizer      | Adam                         | Adaptive optimization for stability                          |
| Loss Function  | BCEWithLogitsLoss + IoU Loss | Combines classification and localization objectives          |
| Device         | GPU (CUDA)                   | Fallback to CPU if unavailable                               |
+----------------+------------------------------+--------------------------------------------------------------+

Loss Function:

Optimiser:

Training steps for each epoch:

1. Forward pass over training set

2. Compute detection loss (classification + bounding box)

3. backpropagate and update weights

4. Evaluate validation loss and IoU metrics

5. plot performanc metrics (loss, mAP@0.5, precision, recall) to determine model convergence

Model Testing
------------

-how was testing performed



Results
-------

-loss graph
-for each class
precision and recall
-IoU


Model Inference
---------------

The prediction pipeline in `predict.py` loads the trained YOLOv7 model into a ONNX format in order to detect and classify lesions in new images. this includes model inference and result visualisation.
- technical details (the func ___ does __, generating predictions for the lesions. Each prediction is comprised of coordinates for a bounding box (x_min, y_min, x_max, y_max) a predicted class label and the associated confidence score (as long as it met the required 0.40 confidence threshold). This data, along with the frame it came from, is processed in OpenCV to display the bounding box, label and confidenc score onto the image.


Visualisation
------------- 
Below is a visual representation of the lesion detection pipeline:

INSERT IMAGE


Usuage
------

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

to automatically install all required dependencies, run: pip install -r requirements.txt


To run the training and prediction scripts, follow these instructions:

1. Download the training, testing and validation data sets and ground truths for task 3 from the ISIC 2018 challenge website (avaliable at: https://challenge.isic-archive.com/data/#2018)
2. Ensure file structures match the file structure provided above
3. update the file paths to point to your correct data directories (for both the training data and the saved model weights for the predictions)
4. train the model:
5. test the model:
6. perform inferences onto new data: 


future recommendations
---------------
- All skin legions from the ISIC dataset are from individuals with light or white skin (Fitzpatrick types I–III). Therefore the model will not generalise to individuals with darker skin tones and will likely struggle to detect and classify legions on other skin types. This is a major limitation of the model's clinical applicability across wider populations and should be fixed by adding data from darker skin tones.

conclusion
----------

This project successfully demonstrates how YOLOv7 can be adapted for skin lesion detection and classification using dermoscopic images. The model achieved strong object precision and classification accuracy on the ISIC 2018 dataset. While limited by dataset diversity of skin tones, the trained model shows potential as a decision support tool for dermatologists, capable of enhancing early detection of malignant skin conditions if provided with more diverse training data and more total training time.


References
------------

HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161

MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M. (2023). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7464–7475.
