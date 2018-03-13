# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import classifier
import dictionary
import extractor
import preprocessor
import vectorizer

def main():
    extracted_data = extractor.extract('inputfile')
    preprocessed_data = preprocessor.preprocess(extracted_data)
    vectorized_data = vectorizer.vectorize_data(preprocessed_data)
    pass

main()
