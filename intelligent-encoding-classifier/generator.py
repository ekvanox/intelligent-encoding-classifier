import os
import random
from dependencies.AES128bitmaster.aes import *
import base64
import hashlib
import string
from sys import path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


def random_string(data: str, length: int = None) -> str:
    """Generates random string with set length

    Args:
        length (int): Length of output string

    Returns:
        string: Randomized string
    """

    randomized_string = ""

    for _ in range(len(data) or length):
        randomized_string += random.choice(
            string.ascii_letters + string.digits)

    return randomized_string


def hex_encode(data: str) -> str:
    """Encodes string to hexadecimal form

    Args:
        data (string): Data to be encoded

    Returns:
        string: Hex encoded data as a string
    """
    return "".join("{:02x}".format(ord(c)) for c in data)


def AES128_encrypt(data: str, key: str = None, aes: object = None) -> str:
    """Encrypts string with AES128

    Args:
        key (string): Key used for encryption. 0-16 characters long.
        data (string): Data to be encrypted. 0-16 characters long.

    Returns:
        [string]: Encrypted data. Hex encoded. 32 characters long.
    """

    # Generate AES object
    aes = AES(mode="ecb", input_type="hex")

    # Add padding to strings to be compliant with AES
    key = key or random_string('', length=16)
    data = data[:16].ljust(16, "0")

    # Convert to hex
    hex_key = hex_encode(key)
    hex_data = hex_encode(data)

    # Encrypt data with key
    encrypted_data: str = aes.encryption(hex_data, hex_key)

    # Return hex data
    return encrypted_data


def shuffle_string(data: str) -> str:
    """Shuffles the input data (maintains length and characters)

    Args:
        data (string): Data to be shuffled

    Returns:
        string: Shuffled input string
    """
    # Takes string and shuffles the characters
    return "".join(random.sample(data, len(data)))


def base64_encode(data: str) -> str:
    """Encode input with base64

    Args:
        data (string): Data to be base64 encoded

    Returns:
        string: Base64 encoded string
    """
    # Convert string to bytes
    data_bytes = data.encode("ascii")

    # Base64 encode bytes
    base64_bytes = base64.b64encode(data_bytes)

    # Convert base64 encoded bytes to string
    base64_string = base64_bytes.decode("ascii")

    # Return base64 data
    return base64_string


def MD5_encrypt(data: str) -> str:
    """Encrypts input to MD5 hash

    Args:
        data (string): Data to be encrypted

    Returns:
        string: MD5 converted string, in hexadecimal format
    """
    # Convert data to bytes
    data_bytes = data.encode("ascii")

    # Encrypt data with MD5
    encrypted_bytes = hashlib.md5(data_bytes)

    # Convert to hex
    encrypted_data = encrypted_bytes.hexdigest()

    return encrypted_data


def no_encoding(data: str) -> str:
    """Returns input string without any modification

    Args:
        data (string): Data to be returned

    Returns:
        string: The provided input
    """

    return data


def write_list_to_file(_list: list, path: str, shuffle: bool = True) -> None:
    """Saves 1 dimensional list as a file with each element in a row

    Args:
        _list (list): List to be saved as file
        path (string): Save file path (relative or absolute)
    """
    # Mix list
    if shuffle:
        random.shuffle(_list)
    # Open file
    with open(path, "w+") as f:
        # Write elements to file, with linebreak
        f.write("\n".join(_list))


# Set working dir to actual file dir
os.chdir(path[0])

# Set constants
TRAINING_DATA_PATH: str = os.path.join("datasets", "training_data")
DICTIONARY_ROOT_PATH: str = "dictionaries"

# User configuration variables
encoding_methods_tuple_set: set = {('AES128', AES128_encrypt), ('base64', base64_encode), (
    'hex', hex_encode), ('MD5', MD5_encrypt), ('plain', no_encoding), ('randomized', random_string)}

# Load dictionaries to memory
real_words_list = []
for dictionary in os.listdir(DICTIONARY_ROOT_PATH):
    with open(os.path.join(DICTIONARY_ROOT_PATH, dictionary)) as file:
        try:
            real_words_list += file.read().split("\n")
        except:
            logging.error(
                "Error reading character in dictionary:" + dictionary
            )


for encoding, function in encoding_methods_tuple_set:
    modified_strings_list = []

    # Pass every word in the wordlist through the encoding function
    for word in tqdm(real_words_list, desc=encoding):
        modified_strings_list.append(function(word))

    # Write the new list to drive
    write_list_to_file(
        modified_strings_list, os.path.join(
            TRAINING_DATA_PATH, encoding, 'data.txt')
    )
