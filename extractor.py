# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Extrahering

import os
import json
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
def json_extract(directory):
    reviews = []
    polarities = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            print('Extracting reviews from ' + file)
            loaded_file = json.load(open(os.path.join(root, file),'r'))
            for review in loaded_file:
                if random.randint(1,4) <= 3:
                    polarity = review['grade']
                    text = review['comment']
                    if int(polarity) < 5:
                        polarities.append(0)
                        reviews.append(text)
                    elif int(polarity) > 7:
                        if random.randint(1, 4) == 1:  # get 1/4 of positive reviews to balance
                            polarities.append(1)
                            reviews.append(text)
                    else:
                        pass # ignore neutral polarities

    return [reviews, polarities]
