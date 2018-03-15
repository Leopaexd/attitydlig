# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Extrahering

import os

# Extract reviews from files in directory and return them in a list.
def extract(directory):
    reviews = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            with open(os.path.join(root, file), "r") as review_file:
                for line in review_file.readlines():
                    reviews.append(line)

    return reviews

