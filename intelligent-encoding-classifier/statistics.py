from __future__ import absolute_import, division, print_function, unicode_literals

# Disable tensorflow debug logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from sys import path
import numpy as np
from tqdm import tqdm
import random

# Define constants
CHARACTERS_IN_DATASET = (
    "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM="
)
TEST_DATA_PATH: str = os.path.join("datasets", "test_data")
TENSORFLOW_MODEL_PATH: str = os.path.join("models", "tensorflow_model.h5")
LEAVE_TIME_STATISTICS: bool = False

# Set working dir to actual file dir
os.chdir(path[0])

# Load class names from training data directory
CLASS_NAMES = [directory for directory in os.listdir(TEST_DATA_PATH)]


# Define functions
def string_to_onehot(input_string):
    return_list = []

    for char in input_string:
        temp_onehot_list = [0] * len(CHARACTERS_IN_DATASET)
        try:
            temp_onehot_list[CHARACTERS_IN_DATASET.index(char)] = 1
        except:
            print(char)
        return_list.append(temp_onehot_list)

    for i in range(64 - len(return_list)):
        return_list.append([0] * len(CHARACTERS_IN_DATASET))
    return return_list


def load_test_data(_class):
    with open(os.path.join(TEST_DATA_PATH, _class, "data.txt")) as f:
        data = f.read().split("\n")

    random.shuffle(data)
    return data


# Load model from save
model = keras.models.load_model(TENSORFLOW_MODEL_PATH)

# Iterate through classes
for class_name in CLASS_NAMES:

    # Load test data
    SAMPLE_STRINGS = load_test_data(class_name)
    SAMPLE_STRINGS = [
        sample_string for sample_string in SAMPLE_STRINGS if len(sample_string) < 63
    ][:100000]

    # Define iterative counter variable
    correct = 0

    # Iterate through individual strings
    for prediction_string in tqdm(SAMPLE_STRINGS, desc=class_name, leave=LEAVE_TIME_STATISTICS):
        # Pre-process data
        INPUT_DATA = []
        INPUT_DATA.append(string_to_onehot(prediction_string))
        processed_input = np.array(INPUT_DATA).astype("float32")

        # Make prediction with model
        prediction_list = model.predict(INPUT_DATA)[0]

        # Print the results
        correct += CLASS_NAMES[np.argmax(prediction_list)] == class_name

    print(
        f"{class_name}: {(correct / len(SAMPLE_STRINGS))*100}% accuracy")
