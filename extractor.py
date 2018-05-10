# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Extrahering

import os
import xml.etree.ElementTree as ET
import random

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
                 if random.randint(1, 247) <= 1:
                    polarity = sentence.find('polarity').text
                    text = sentence.find('text').text
                    reviews.append(text.lower())
                    if polarity.upper() == 'POSITIVE':
                        polarities.append(1)
                    else:
                        polarities.append(0)
            except ET.ParseError:
                pass

    return [reviews,polarities]
