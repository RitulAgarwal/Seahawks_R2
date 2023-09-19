import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(0)
from tensorflow import keras
import numpy as np
np.random.seed(0)
import itertools
from keras.preprocessing import image_dataset_from_directory
from keras.layers.experimental.preprocessing import Rescaling
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

model = keras.models.load_model('plant_disease_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('cropAgent.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='cropAgent.tflite')
interpreter.allocate_tensors()

image_path = 'plant.jpg'
image = keras.preprocessing.image.load_img(image_path, target_size=(256, 256,3))
input_image = keras.preprocessing.image.img_to_array(image)
input_image = tf.expand_dims(input_image, axis=0)
input_image /= 255.0  

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])

prediction_result = np.argmax(predictions)
reversed_dict = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
}
prediction_result = reversed_dict[prediction_result]

import re

prediction_result = re.sub(r'_{1,}', ' ', prediction_result)
print(prediction_result)
from transformers import pipeline


qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

question = "How can I cure a plant affected by Apple Cedar apple rust?"
context = "Apple Cedar apple rust is a fungal disease that affects apple trees. It can be treated by..."


answer = qa_pipeline(question=question, context=context, max_tokens=150)


print("Answer:", answer['answer'])
