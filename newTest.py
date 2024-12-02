import cv2
import os
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from datetime import datetime
import time

# Path to your images folder
IMAGES_FOLDER = "images"
MODEL_PATH = "Model2/keras_model_compatible.h5"
LABELS_PATH = "Model2/labels.txt"
IMG_SIZE = 300
OFFSET = 20

# Initialize the HandDetector and Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)

# Labels for the prediction
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

# Function to get all images sorted by modification time
def get_recent_images():
    # List all files in the folder
    files = [f for f in os.listdir(IMAGES_FOLDER) if os.path.isfile(os.path.join(IMAGES_FOLDER, f))]

    # If the folder is empty, return an empty list
    if not files:
        print("No images found.")
        return []

    # Get the full path of the files
    full_paths = [os.path.join(IMAGES_FOLDER, f) for f in files]

    # Sort the files by modification time (most recent first)
    full_paths.sort(key=os.path.getmtime, reverse=True)

    return full_paths


# Function to process an image
def process_image(image_path):
    img = cv2.imread(image_path)  # Read the image
    if img is None:
        print(f"Error reading image {image_path}. Skipping.")
        return

    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hands in the image

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the bounding box is within the image bounds
        if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            print(f"Invalid bounding box for image {image_path}. Skipping.")
            return

        # Prepare the white image for resizing
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]

        # Ensure the cropped image is not empty
        if imgCrop.size == 0:
            print(f"Empty crop for image {image_path}. Skipping.")
            return

        aspectRatio = h / w
        if aspectRatio > 1:
            k = IMG_SIZE / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
            wGap = math.ceil((IMG_SIZE - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
            print(index)
        else:
            k = IMG_SIZE / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
            hGap = math.ceil((IMG_SIZE - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Make the prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display the result on the image
        cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET - 50), (x - OFFSET + 95, y - OFFSET - 50 + 50), (255, 0, 255),
                      cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.9, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (255, 0, 255), 4)

        # Show the images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    else:
        print(f"No hand detected in image {image_path}. Skipping.")

    # Show the final output
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)  # Wait 1ms before processing the next image

    # After processing, delete the image
    os.remove(image_path)  # Delete the image after processing
    print(f"Deleted image: {image_path}")


# Main function to continuously process new images
def main():
    processed_images = set()  # Keep track of processed images to avoid duplication

    while True:
        # Get all recent images sorted by modification time
        recent_images = get_recent_images()

        # If no images are found, stop the program
        if not recent_images:
            print("No images left to process. Exiting.")
            break  # Exit the loop and stop the program

        # Process any new images
        for image_path in recent_images:
            if image_path not in processed_images:
                print(f"Processing the image: {image_path}")
                process_image(image_path)
                processed_images.add(image_path)  # Mark the image as processed

        # Wait for a short time before checking again (to avoid excessive CPU usage)
        time.sleep(1)


if __name__ == "__main__":
    main()
