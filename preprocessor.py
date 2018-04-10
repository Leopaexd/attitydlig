# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Förprocessning

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import time

# Take a list of unprocessed reviews and return a list of preprocessed reviews.
def preprocess(unprocessed_reviews):
    start = time.time()
    # stemmer = SnowballStemmer('swedish')
    preprocessed_reviews = []
    print ("Preprocessing...")
    number_of_reviews = 0
    for review in unprocessed_reviews:
        tokenized_review = []
        for word in word_tokenize(review):
            if word not in ["." , ","]:
                # tokenized_review.append(stemmer.stem(word).lower())
                tokenized_review.append(word.lower())
        preprocessed_reviews.append(tokenized_review)
        number_of_reviews += 1
    time_elapsed = time.time() - start

    print("Preprocessed " + str(number_of_reviews) + " reviews in ", ("%.2f" % time_elapsed), "seconds")
    return preprocessed_reviews