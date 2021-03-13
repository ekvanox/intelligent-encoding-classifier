from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
import base64
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
import os
from sys import path


# Define functions
def guess_encoding(s: str) -> str:
    driver.get(
        f"https://gchq.github.io/CyberChef/#recipe=Magic(3,false,false,'')&input={base64.b64encode(s.encode('ascii')).decode('ascii').replace('=','')}")

    while 1:
        try:
            root_element = driver.find_element_by_id('output-html')
            analysed_text = root_element.find_element_by_tag_name(
                'tbody').find_elements_by_tag_name('tr')[-1].find_elements_by_tag_name('td')[1]
            assert analysed_text.text == s
            break
        except:
            pass
    possibly_tested_guess_box = root_element.find_element_by_tag_name(
        'tbody').find_elements_by_tag_name('tr')[1].find_elements_by_tag_name('td')[0]
    guess_box = root_element.find_element_by_tag_name(
        'tbody').find_elements_by_tag_name('tr')[-1].find_elements_by_tag_name('td')[2]

    first_guess = possibly_tested_guess_box.text+guess_box.text.split(',')[0]

    if 'Possible languages' in first_guess:
        return 'plain'
    elif 'Hex' in first_guess:
        return 'hex'
    elif 'Base64' in first_guess:
        return 'base64'
    else:
        return 'randomized'


# Set working dir to actual file dir
os.chdir(path[0])

# Start driver
driver = webdriver.Chrome()

# Define constants
TEST_DATA_PATH: str = os.path.join("datasets", "test_data")
ENCODINGS: list = ['randomized', 'plain', 'hex', 'base64']

# Iterate through encodings
for encoding in ENCODINGS:
    # Load test data
    with open(os.path.join(TEST_DATA_PATH, encoding, 'data.txt'), 'r') as f:
        test_strings = f.read().split()[:100_000]
        correct_guesses = 0

        for test_string in tqdm(test_strings):
            correct_guesses += guess_encoding(test_string) == encoding

        print(encoding, (correct_guesses/len(test_strings))*100, '%')
