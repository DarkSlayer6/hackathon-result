import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os

# Load your pre-trained model
model = load_model(r'c:\Users\Lenovo\Desktop\keras.h\keras_model.h5')

# Define paths for saving images
safe_dir = r'c:\Users\Lenovo\Desktop\keras.h\safe'
unsafe_dir = r'c:\Users\Lenovo\Desktop\keras.h\unsafe'

# Create directories if they don't exist
os.makedirs(safe_dir, exist_ok=True)
os.makedirs(unsafe_dir, exist_ok=True)

# Function to classify an image captured from the webcam
def classify_image_from_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        return

    # Release the webcam
    cap.release()

    # Preprocess the image
    image = cv2.resize(frame, (224, 224))  # Adjust size as per your model
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize if required by your model

    # Make predictions
    predictions = model.predict(image_array)

    # Assuming model outputs probabilities for two classes: safe and unsafe
    safe_prob = predictions[0][1]  # Adjust index based on your output structure
    unsafe_prob = predictions[0][0]

    # Define a threshold for classification
    threshold = 0.9  # You can adjust this threshold

    # Check probabilities and classify
    if safe_prob > threshold:
        print("Predicted Class: safe")
        # Save the image to the safe directory
        cv2.imwrite(os.path.join(safe_dir, 'safe_image.jpg'), frame)
        print("Image saved in safe directory.")
    elif unsafe_prob > threshold:
        print("Predicted Class: unsafe")
        # Save the image to the unsafe directory
        cv2.imwrite(os.path.join(unsafe_dir, 'unsafe_image.jpg'), frame)
        print("Image saved in unsafe directory.")
    else:
        # Ask user for input if neither class exceeds threshold
        user_input = input("Neither class exceeded the threshold. Which class would you add it to? (safe/unsafe): ").strip().lower()
        if user_input in ["safe", "unsafe"]:
            # Save the image to the corresponding directory
            target_dir = safe_dir if user_input == "safe" else unsafe_dir
            cv2.imwrite(os.path.join(target_dir, f'{user_input}_image.jpg'), frame)
            print(f"Image added to {user_input} directory.")
        else:
            print("Invalid class input.")

# Run the classification function
classify_image_from_webcam()
