from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tqdm import tqdm
import numpy as np
from sys import path
import tensorflow as tf
from numpy import array
from tensorflow.keras.utils import to_categorical
import random

# Disable tensorflow debug logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Set working directory to script directory
os.chdir(path[0])

# User configurable variables
NEURONS = 54
DROPOUT = 0.0
BATCH_SIZE = 32
EPOCHS = 1

# Define constants
TRAINING_DATA_PATH: str = os.path.join("datasets", "training_data")
TENSORFLOW_MODELS_PATH: str = "models"
NAME = f"NEURONS={NEURONS},DROPOUT={DROPOUT},BATCH_SIZE={BATCH_SIZE},EPOCHS={EPOCHS}"

# Load class names from training data directory
CLASS_NAMES = [directory for directory in os.listdir(TRAINING_DATA_PATH)]


def load_test_data(_class):
    with open(os.path.join(TRAINING_DATA_PATH, _class, "data.txt")) as f:
        data = f.read().split("\n")

    return data


def chunks(l, n):
    n = max(1, n)
    return [l[i: i + n] for i in range(0, len(l), n)]


def string_to_onehot(input_string: str):
    characters = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '0': 9, 'q': 10, 'w': 11, 'e': 12, 'r': 13, 't': 14, 'y': 15, 'u': 16, 'i': 17, 'o': 18, 'p': 19, 'a': 20, 's': 21, 'd': 22, 'f': 23, 'g': 24, 'h': 25, 'j': 26, 'k': 27, 'l': 28, 'z': 29, 'x': 30,
                  'c': 31, 'v': 32, 'b': 33, 'n': 34, 'm': 35, 'Q': 36, 'W': 37, 'E': 38, 'R': 39, 'T': 40, 'Y': 41, 'U': 42, 'I': 43, 'O': 44, 'P': 45, 'A': 46, 'S': 47, 'D': 48, 'F': 49, 'G': 50, 'H': 51, 'J': 52, 'K': 53, 'L': 54, 'Z': 55, 'X': 56, 'C': 57, 'V': 58, 'B': 59, 'N': 60, 'M': 61, '=': 62, ' ': 63}

    input_string = input_string.ljust(64, ' ')
    data = array([characters[c] for c in input_string])
    onehot_list = to_categorical(data, num_classes=64)

    return onehot_list


model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(64, 64)),
        keras.layers.Dense(NEURONS, activation=keras.activations.relu),
        keras.layers.Dense(NEURONS, activation=keras.activations.relu),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(len(CLASS_NAMES), activation=tf.nn.softmax),
    ]
)

sgd = keras.optimizers.SGD(lr=0.10, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy")

training_data = []

for i, class_name in enumerate(CLASS_NAMES):
    print(f"Loading class {class_name}")
    training_data.append(load_test_data(class_name))

random.shuffle(training_data)
print(training_data)
training_data = chunks(training_data, 100_000)

# Load a user specified number inputs at a time
for training_iteration in tqdm(range(len(training_data[0]))):
    # print(f"Iteration {training_iteration}/{len(training_data[0])}")

    INPUT_DATA = []
    INPUT_LABLES = []

    for i, encoding_data in enumerate(training_data):
        for _string in encoding_data[training_iteration]:
            INPUT_DATA.append(string_to_onehot(_string))
        INPUT_LABLES += [i for _ in encoding_data[training_iteration]]
        # print(i, end=',', flush=True)

    print(training_data[0])

    print(INPUT_LABLES)

    try:
        INPUT_LABLES = np.array(INPUT_LABLES).astype("float32")
        INPUT_DATA = np.array(INPUT_DATA).astype("float32")
    except Exception as e:
        print(e)
        continue

    training_history = model.fit(
        INPUT_DATA,
        INPUT_LABLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
        shuffle=False,
        use_multiprocessing=True,
    )

    # Make predictions with model
    prediction_list = model.predict(INPUT_DATA[:1])[0]

    # Print the results
    print(CLASS_NAMES[np.argmax(prediction_list)], INPUT_LABLES[0])

loss = training_history.history['val_loss']

# Save model
model.save(os.path.join(TENSORFLOW_MODELS_PATH, f"{loss}-{NAME}.h5"))
model.save(os.path.join(TENSORFLOW_MODELS_PATH, "tensorflow_model_latest.h5"))
