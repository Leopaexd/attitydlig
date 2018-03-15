# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import classifier
import dictionary as dict
import extractor
import preprocessor
import vectorizer

def main():
    directory = 'C:\\Users\\olive\\Desktop\\Datamängder för uppsats\\ESCW 2016'

    extracted_data = extractor.extract(directory)

    for line in extracted_data:
        print(line)
    preprocessed_data = preprocessor.preprocess(extracted_data)
    dictionary = dict.Dictionary(preprocessed_data)
    vectorized_data = vectorizer.vectorize_data(preprocessed_data,dictionary,300) # 300 word maximum length

main()
