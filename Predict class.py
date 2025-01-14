import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Function to read the file and convert to a 2D array
def read_2d_array_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Convert each line to a list of integers and store in a 2D array
    array_2d = [list(map(float, line.strip().split())) for line in lines]
    
    return array_2d

# Example usage
filename = 'predicate-matrix-continuous.txt'  # Predicate-matrix-continuous file path
array_2d = read_2d_array_from_file(filename)

predicate_continuous = np.array(array_2d) # Convert to numpy array
predicate_continuous[predicate_continuous < 0] = 0 # Replace negative values with 0
del array_2d # Free up memory

# Taking the names of the classes from classes.txt
class_dir = 'classes.txt' # Path to the classes.txt file
classes = []
with open (class_dir, 'r') as f:
    for line in f:
        line = line.split('\n')[0]
        if line.strip():  # Ignore empty lines
            _, name = line.split(maxsplit=1)  # Split into number and name
            classes.append(name)

# Distinguishing between seen and unseen classes
unseen_classes_index = [6, 15, 19, 21, 22, 24, 26, 27, 28, 45]
seen_classes_index = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 23, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49]

# Load the model
model = load_model('Animal_classification_model(2).h5', compile = False)

# Preprocessing function

def preprocess_image_2(image_path, target_size=346):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, target_size, target_size)
    image = image / 255.0  # Normalize to [0, 1]

    return tf.expand_dims(image,axis=0) # expanding the dimension in order to fit the batch size of 1

#Using 2 types of evaluation for image classification
def evaluate_image(image_path):
    image = preprocess_image_2(image_path)
    embedding_pred = model.predict(image)
    similiarity = cosine_similarity(embedding_pred, predicate_continuous)
    predicted_label_index = tf.argmax(similiarity, axis =1).numpy()[0]
    print(classes[predicted_label_index])
    return classes[predicted_label_index]

#using this so that the model does not get biased towards seen classes
def evaluate_image_2(image_path):
    threshold_perc= 0.89
    image = preprocess_image_2(image_path)
    embedding_pred = model.predict(image)
    similarity = cosine_similarity(embedding_pred, predicate_continuous)
    for i in seen_classes_index:
        similarity[0,i] *= threshold_perc
    predicted_label_index = tf.argmax(similarity, axis=1).numpy()[0]
    print(classes[predicted_label_index])
    return classes[predicted_label_index]

# Path of the image on which you have to test the model
image_path = '00010.jpg' # Replace this file path with the path of the image you want to test
print(evaluate_image_2(image_path))