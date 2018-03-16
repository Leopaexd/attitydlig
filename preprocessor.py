# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Förprocessning


# Take a list of unprocessed reviews and return a list of preprocessed reviews.
def preprocess(unprocessed_reviews):
    preprocessed_reviews = []
    for review in unprocessed_reviews:
        tokenized_review = tokenize(review)
        preprocessed_reviews.append(tokenized_review)


    return preprocessed_reviews

def tokenize(review):
    return review.split()