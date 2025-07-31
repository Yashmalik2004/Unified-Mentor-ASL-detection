import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model("asl_high_acc_model.h5")

classes = ['A', 'B', 'C', 'D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y', 'Z', 'del', 'nothing', 'space']

# Preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict from a sample image
def predict_sample_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from path {image_path}")
        return
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    predicted_class = classes[np.argmax(prediction)]
    print(f"Prediction: {predicted_class}")

# Live detection from webcam
def live_asl_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Starting live webcam ASL detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI (just a square in the middle)
        x1, y1, x2, y2 = 150, 100, 350, 300
        roi = frame[y1:y2, x1:x2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(rgb)

        # Predict
        prediction = model.predict(processed)[0]
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Show predictions
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"{predicted_class} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("Live ASL Detection", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Choose mode
mode = input("Choose mode:\n1. Predict on sample image\n2. Live webcam ASL detection\nEnter 1 or 2: ")

if mode == "1":
    path = input("Enter image path: ")
    predict_sample_image(path)
elif mode == "2":
    live_asl_detection()
else:
    print("Invalid input.")
