# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Extrahering

import os
import xml.etree.ElementTree as ET
import json


# Extract reviews from files in directory and return them in a list.
def extract(directory):
    reviews = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            with open(os.path.join(root, file), "r") as review_file:
                for line in review_file.readlines():
                    reviews.append(line)

    return reviews

# Extract reviews from files in directory and return them in a list. Polarities are returned in parallel list.
def xml_extract(directory):
    reviews = []
    polarities = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            print('Extracting reviews from ' + file)
            try:
             tree = ET.parse(os.path.join(root, file))
             for sentence in tree.getroot().findall('sentence'):
                 polarity = sentence.find('polarity').text
                 text = sentence.find('text').text
                 reviews.append(text)
                 polarities.append(polarity)
            except ET.ParseError:
                pass
    return [reviews, polarities]

def json_extract(directory):
    reviews = []
    polarities = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            print('Extracting reviews from ' + file)
            loaded_file = json.load(open(os.path.join(root, file),'r'))
            for review in loaded_file:
                polarity = review['grade']
                text = review['comment']
                reviews.append(text)
                polarities.append(polarity)
    return [reviews, polarities]