# ASL Alphabet Detector

Simple and live American Sign Language (ASL) alphabet detection using a convolutional neural network.  
The model distinguishes 29 classes: A–Z plus special tokens like `SPACE`, `DELETE`, `NOTHING` (depending on the dataset variant).

## Features

- Train a regularized CNN with augmentation to reduce overfitting.
- Save and load the trained model (`asl_high_acc_model.h5`).
- Predict from a single sample image.
- Real-time live ASL detection via webcam.
- Easy-to-follow setup from scratch with virtual environment.

## Dataset

This project uses the **ASL Alphabet** dataset (29 classes) originally from Kaggle (`grassknoted/asl-alphabet`).  
You need to download and unzip the training data so that the directory structure looks like:

- data/
  - asl_alphabet_train/
    - A/
    - B/
    - ...
    - Z/
    - SPACE/
    - DELETE/
    - NOTHING/



## Getting Started

### 1. Clone / Prepare Project

```bash
git clone https://github.com/Yashmalik2004/Unified-Mentor-ASL-detection
cd asl-alphabet-detector
```
### 2. Create and Activate Virtual Environment
 ## Windows

```bash
python -m venv venv
venv\Scripts\activate

```
 ## macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate

```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model
## Run the training script (e.g., model.py) which performs:

- Augmentation (rotation, zoom, shift, shear, horizontal flip)

- CNN with dropout, batch norm, and L2 regularization

- Learning rate reduction and early stopping

- Saves the best model as asl_high_acc_model.h5

- Outputs training accuracy/loss plots (training_accuracy.png, training_loss.png)

```bash
python model.py
```

### 5. Predict / Inference

```bash
python test_model.py
```
- Choose option 1 and enter the path to a test image. The script will display the image with the predicted ASL letter.

- Live Webcam Prediction
- From the same script, choose option 2 to launch the webcam ASL detector. Show your hand inside the predefined ROI box; prediction and       confidence will overlay on the feed.
- Press CTRL+C to quit.
