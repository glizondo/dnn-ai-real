import tensorflow as tf
# import cv2
from pathlib import Path
import numpy as np

# Load the model we trained
model = tf.keras.models.load_model('model.h5')
# Next, we’ll loop through all PNG image files in the current folder and load each one
for f in sorted(Path(r".\ai-photos\use").glob("*.jpg")):
    # Load an image file to test
    image_to_test = tf.keras.preprocessing.image.load_img(str(f), target_size=(32, 32))

    # Convert the image data to a numpy array suitable for Keras
    image_to_test = tf.keras.preprocessing.image.img_to_array(image_to_test)

    # Normalize the image the same way we normalized the training data (divide all numbers by 255)
    image_to_test /= 255

    # Add a fourth dimension to the image since Keras expects a list of images
    list_of_images = np.expand_dims(image_to_test, axis=0)

    # Make a prediction using the bird model
    results = model.predict(list_of_images)

    # Since we only passed in one test image, we can just check the first result directly.
    image_likelihood = results[0][0]

    # The result will be a number from 0.0 to 1.0 representing the likelihood that this image is a bird.
    # up to you to decide , 0.5 seems like a good choice.
    if image_likelihood > 0.5:
        print(f'Predicted class is REAL')
    else:
        print(f'Predicted class is FAKE')
# Run and evaluate your model quality!
