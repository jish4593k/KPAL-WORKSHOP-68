import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    img = img / 255.0  # Normalize to [0, 1]
    return img

def load_pretrained_model():
    model = InceptionResNetV2(weights='imagenet')
    return model

def predict_color(image, model):
    image_batch = np.expand_dims(image, axis=0)
    processed_image = preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions)
    color_name = decoded_predictions[0][0][1]
    return color_name

def main(image_path):
    print("Loading Pre-trained Model...")
    colorization_model = load_pretrained_model()

    print("Resizing and Normalizing Image...")
    input_image = resize_image(image_path)

    print("Predicting Color...")
    predicted_color = predict_color(input_image, colorization_model)

    print(f"The predicted dominant color is: {predicted_color}")

    plt.imshow(input_image)
    plt.axis('off')
    plt.title('Original Image')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path-to-image>")
    else:
        main(sys.argv[1])
