# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Förprocessning

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

# Take a list of unprocessed reviews and return a list of preprocessed reviews.
def preprocess(unprocessed_reviews):
    stemmer = SnowballStemmer('english')
    preprocessed_reviews = []
    for review in unprocessed_reviews:
        tokenized_review = []
        for word in word_tokenize(review):
            if word not in ["." , ","]:
                tokenized_review.append(stemmer.stem(word).upper())

        preprocessed_reviews.append(tokenized_review)
        print(tokenized_review)


    return preprocessed_reviews
