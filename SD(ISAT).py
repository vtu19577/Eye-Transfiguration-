import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load pre-trained MobileNetV2 model for image classification
model = MobileNetV2(weights='imagenet')

# Function to detect eyes in an image
def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes

# Load and preprocess images
img_path1 = 'C:\\Users\\dhara\\Documents\\Images\\Test img-2.jpg'  # Image from which eyes will be removed
img_path2 = 'C:\\Users\\dhara\\Documents\\Images\\Test img-4.jpg'  # Image from which eyes will be taken
img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

# Detect eyes in both images
eyes1 = detect_eyes(img1)
eyes2 = detect_eyes(img2)

# Ensure both images have detected eyes
if len(eyes1) > 0 and len(eyes2) > 0:
    # Take the first detected eye from each image
    (x1, y1, w1, h1) = eyes1[0]
    (x2, y2, w2, h2) = eyes2[0]

    # Extract eye regions
    eye_region1 = img1[y1:y1+h1, x1:x1+w1]
    eye_region2 = img2[y2:y2+h2, x2:x2+w2]

    # Resize eye region 2 to match the size of eye region 1
    eye_region2 = cv2.resize(eye_region2, (w1, h1))

    # Replace eye region in image 1 with eye region from image 2
    img1[y1:y1+h1, x1:x1+w1] = eye_region2

    # Display the modified image using matplotlib
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("Eyes not detected in one or both images.")
