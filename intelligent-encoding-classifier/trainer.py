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

os.chdir(path[0])  # Set default dir to actual file dir

# User configurable variables
NEURONS = 54
DROPOUT = 0.0
BATCH_SIZE = 32
EPOCHS = 1

# Define constants
TRAINING_DATA_PATH: str = os.path.join("datasets", "training_data")
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


def string_to_onehot(input_string):
    characters = "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM="
    return_list = []

    for char in input_string:
        temp_onehot_list = [0] * len(characters)
        try:
            temp_onehot_list[characters.index(char)] = 1
        except:
            print("Error: " + char)
        return_list.append(temp_onehot_list)

    for i in range(64 - len(return_list)):
        return_list.append([0] * len(characters))
    return return_list


model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(64, 63)),
        keras.layers.Dense(NEURONS, activation=tf.keras.activations.relu),
        keras.layers.Dense(NEURONS, activation=tf.keras.activations.relu),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(len(CLASS_NAMES), activation=tf.nn.softmax),
    ]
)

sgd = keras.optimizers.SGD(lr=0.10, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy")

training_data = []

for i, class_name in enumerate(CLASS_NAMES):
    print(f"Loading class {class_name}")
    training_data.append(chunks(load_test_data(class_name), 10_000))


# Load 10.000 inputs at a time
for training_iteration in range(len(training_data[0])):
    print(f"Iteration {training_iteration}")

    INPUT_DATA = []
    INPUT_LABLES = []

    for i, encoding_data in enumerate(training_data):
        for _string in encoding_data[training_iteration]:
            INPUT_DATA.append(string_to_onehot(_string))
        INPUT_LABLES += [i for _ in encoding_data[training_iteration]]

    try:
        INPUT_LABLES = np.array(INPUT_LABLES).astype("float32")
        INPUT_DATA = np.array(INPUT_DATA).astype("float32")
    except Exception as e:
        print(e)
        continue

    model.fit(
        INPUT_DATA,
        INPUT_LABLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
        shuffle=True,
        use_multiprocessing=True,
    )


# Creates model name
save_model_name = f"tensorflow_model-{NAME}.h5"
# Saves model with time and name to prevent over writing
model.save(save_model_name)
model.save("tensorflow_model_latest.h5")
