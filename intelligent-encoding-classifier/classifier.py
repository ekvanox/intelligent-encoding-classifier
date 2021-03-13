from sys import path

# Disable tensorflow debug logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Finally import keras from tensorflow
from tensorflow import keras

# Set working dir to actual file dir
os.chdir(path[0])

# User configuration variables
PREDICTION_STRING: str = input("Input to predict: ")
CHARACTERS_IN_DATASET: str = (
    "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM="
)


# Load class names from training data directory
CLASS_NAMES: list = [directory for directory in os.listdir(os.path.join("datasets", "training_data"))]


# Define functions
def string_to_onehot(input_string: str) -> list:
    return_list = []

    for char in input_string:
        temp_onehot_list = [0] * len(CHARACTERS_IN_DATASET)
        try:
            temp_onehot_list[CHARACTERS_IN_DATASET.index(char)] = 1
        except:
            print(char)
        return_list.append(temp_onehot_list)

    for _ in range(64 - len(return_list)):
        return_list.append([0] * len(CHARACTERS_IN_DATASET))
    return return_list


# Load model from save
model = keras.models.load_model(os.path.join("models", "tensorflow_model.h5"))

# Pre-process data
INPUT_DATA = [string_to_onehot(PREDICTION_STRING)]

# Make prediction with model
prediction_list = model.predict(INPUT_DATA)[0]

# Sort predictions made by network
sorted_predictions = [(encoding,prediction_list[i]) for i,encoding in enumerate(CLASS_NAMES)]
sorted_predictions.sort(key=lambda tup: tup[1],reverse=True)

# Print the predictions
for encoding, probability in sorted_predictions:
    print(f"{encoding}: {round(probability*100, 2)}%")