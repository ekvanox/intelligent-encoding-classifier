from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import random
from tqdm import tqdm
import numpy as np
from tensorflow.keras.utils import to_categorical
from numpy import array
from sys import path

# Disable tensorflow debug logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define constants
CHARACTERS_IN_DATASET = (
    "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM= "
)
TEST_DATA_PATH: str = os.path.join("datasets", "test_data")
TENSORFLOW_MODEL_PATH: str = os.path.join(
    "models", "tensorflow_model_latest.h5")
LEAVE_TIME_STATISTICS: bool = False

# Set working directory to script directory
os.chdir(path[0])

# Load class names from training data directory
CLASS_NAMES = [directory for directory in os.listdir(TEST_DATA_PATH)]


# Define functions
def string_to_onehot(input_string: str):
    characters = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '0': 9, 'q': 10, 'w': 11, 'e': 12, 'r': 13, 't': 14, 'y': 15, 'u': 16, 'i': 17, 'o': 18, 'p': 19, 'a': 20, 's': 21, 'd': 22, 'f': 23, 'g': 24, 'h': 25, 'j': 26, 'k': 27, 'l': 28, 'z': 29, 'x': 30,
                  'c': 31, 'v': 32, 'b': 33, 'n': 34, 'm': 35, 'Q': 36, 'W': 37, 'E': 38, 'R': 39, 'T': 40, 'Y': 41, 'U': 42, 'I': 43, 'O': 44, 'P': 45, 'A': 46, 'S': 47, 'D': 48, 'F': 49, 'G': 50, 'H': 51, 'J': 52, 'K': 53, 'L': 54, 'Z': 55, 'X': 56, 'C': 57, 'V': 58, 'B': 59, 'N': 60, 'M': 61, '=': 62, ' ': 63}

    input_string = input_string.ljust(64, ' ')
    data = array([characters[c] for c in input_string])
    onehot_list = to_categorical(data, num_classes=64)

    return onehot_list


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
    ][:100]

    # Define iterative counter variable
    correct = 0

    # Iterate through individual strings
    for prediction_string in tqdm(SAMPLE_STRINGS, desc=class_name, leave=LEAVE_TIME_STATISTICS):
        # Pre-process data
        INPUT_DATA = []
        INPUT_DATA.append(string_to_onehot(prediction_string).tolist())
        # processed_input = np.array(INPUT_DATA).astype("float32")

        # Make predictions with model
        prediction_list = model.predict(INPUT_DATA)[0]

        print(prediction_list)

        # Print the results
        correct += CLASS_NAMES[np.argmax(prediction_list)] == class_name

    print(
        f"{class_name}: {(correct / len(SAMPLE_STRINGS))*100}% accuracy")
