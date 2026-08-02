import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

from tensorflow.keras.preprocessing import image


def predict_image(img_path):
    # Load Image
    img = image.load_img(img_path, target_size=(224, 224))

    # Display Image
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # Convert image to array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    # Decode Predictions
    results = decode_predictions(prediction, top=5)[0]

    print("\nTop 5 Predictions:\n")

    for _, label, confidence in results:
        print(f"{label:<25} {confidence * 100:.2f}%")


if __name__ == "__main__":

    print("Loading MobileNetV2 Model...")
    model = MobileNetV2(weights="imagenet")
    print("✅ Model Loaded Successfully!\n")

    while True:

        img_path = input("Enter image path (or type 'exit'): ")

        if img_path.lower() == "exit":
            print("Thank You!")
            break

        try:
            predict_image(img_path)

        except FileNotFoundError:
            print("\n❌ Image not found!")
            print("Example: images/dog.jpg\n")

        except Exception as e:
            print("\nError:", e)
