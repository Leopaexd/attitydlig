# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import classifier
import dictionary as dict
import extractor
import preprocessor
import vectorizer

def main():
    extracted_data = extractor.extract('inputfile')
    preprocessed_data = preprocessor.preprocess(extracted_data)
    dictionary = dict.Dictionary(preprocessed_data)
    vectorized_data = vectorizer.vectorize_data(preprocessed_data,dictionary,300) # 300 word maximum length

main()
