import random
from tqdm import tqdm
import os
from sys import path


def classify_string(string_to_classify: str):
    if string_to_classify.islower() and string_to_classify.isalpha():
        return 'plain'
    else:
        try:
            # Check if string can be converted from base 16 to base 10
            int(string_to_classify, 16)

            if len(string_to_classify) == 32:
                return random.choice(['AES128', 'MD5'])
            else:
                return 'hex'
        except:
            if '=' in string_to_classify:
                return 'base64'
            else:
                return 'random'


# Set working dir to actual file dir
os.chdir(path[0])

# Define constants
TEST_DATA_PATH: str = os.path.join("datasets", "test_data")
LEAVE_TIME_STATISTICS: bool = False

with open(os.path.join(TEST_DATA_PATH, 'aes128', 'data.txt'), 'r') as f:
    string_list = f.read().split('\n')[:100_000]
    correct_guesses = 0

    for string in tqdm(string_list, leave=LEAVE_TIME_STATISTICS):
        correct_guesses += classify_string(string) == "AES128"

    print(f'AES128: {(correct_guesses / len(string_list))*100}%')

with open(os.path.join(TEST_DATA_PATH, 'base64', 'data.txt'), 'r') as f:
    string_list = f.read().split('\n')[:100_000]
    correct_guesses = 0

    for string in tqdm(string_list, leave=LEAVE_TIME_STATISTICS):
        correct_guesses += classify_string(string) == "base64"

    print(f'base64: {(correct_guesses / len(string_list))*100}%')

with open(os.path.join(TEST_DATA_PATH, 'hex', 'data.txt'), 'r') as f:
    string_list = f.read().split('\n')[:100_000]
    correct_guesses = 0

    for string in tqdm(string_list, leave=LEAVE_TIME_STATISTICS):
        correct_guesses += classify_string(string) == "hex"

    print(f'hex: {(correct_guesses / len(string_list))*100}%')

with open(os.path.join(TEST_DATA_PATH, 'md5', 'data.txt'), 'r') as f:
    string_list = f.read().split('\n')[:100_000]
    correct_guesses = 0

    for string in tqdm(string_list, leave=LEAVE_TIME_STATISTICS):
        correct_guesses += classify_string(string) == "MD5"

    print(f'MD5: {(correct_guesses / len(string_list))*100}%')

with open(os.path.join(TEST_DATA_PATH, 'plain', 'data.txt'), 'r') as f:
    string_list = f.read().split('\n')[:100_000]
    correct_guesses = 0

    for string in tqdm(string_list, leave=LEAVE_TIME_STATISTICS):
        correct_guesses += classify_string(string) == "plain"

    print(f'Plain: {(correct_guesses / len(string_list))*100}%')

with open(os.path.join(TEST_DATA_PATH, 'randomized', 'data.txt'), 'r') as f:
    string_list = f.read().split('\n')[:100_000]
    correct_guesses = 0

    for string in tqdm(string_list, leave=LEAVE_TIME_STATISTICS):
        correct_guesses += classify_string(string) == "random"

    print(f'Randomized: {(correct_guesses / len(string_list))*100}%')
