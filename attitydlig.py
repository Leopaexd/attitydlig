# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import classifier
import dictionary as dict
import extractor
import preprocessor
import vectorizer

def main():
    directory = 'C:\\Users\\olive\\Desktop\\Datamängder för uppsats\\ESCW 2016\\Software'

    extracted_data = extractor.xml_extract(directory)
    extracted_reviews = extracted_data[0]
    polarities = extracted_data[1]

    #for line in extracted_reviews:
       # print(line)

    preprocessed_reviews = preprocessor.preprocess(extracted_reviews)
    dictionary = dict.Dictionary(preprocessed_reviews)
    vectorized_data = vectorizer.vectorize_data(preprocessed_reviews,dictionary,300) # 300 word maximum length
    tpot = classifier.Classifier()
    tpot.fit(vectorized_data,polarities)

main()
