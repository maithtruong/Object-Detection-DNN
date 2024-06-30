import cv2
import streamlit as st
from PIL import Image
import numpy as np

MODEL = "model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "model/MobileNetSSD_deploy.prototxt.txt"


def process_image(image):
    # Load the model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    # Preprocess the image for DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the network
    net.setInput(blob)

    # Forward pass and get detections
    detections = net.forward()

    return detections


def annotate_image(image, detections, confidence_threshold=0.5):
    (h, w) = image.shape[:2]

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by confidence threshold
        if confidence > confidence_threshold:
            # Extract class label index and compute bounding box coordinates
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(image, (startX, startY), (endX, endY), (70, 0, 0), 2)

    return image


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if file is not None:
        st.image(file, caption="Uploaded Image")

        # Open and convert the uploaded image to a NumPy array
        image = Image.open(file)
        image = np.array(image)

        # Process the image to detect objects
        detections = process_image(image)

        # Annotate the image with detected objects
        processed_image = annotate_image(image, detections)

        # Display the processed image
        st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
