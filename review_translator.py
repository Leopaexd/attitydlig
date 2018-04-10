# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Översättning av produktomdömen

import azure_translator
import googletrans
import random

def translate_reviews(swedish_reviews, polarities):
    review_file = open('translated_reviews.txt', 'w')
    polarity_file = open('translated_polarities.txt', 'w')
    untranslated_file = open('untranslated_reviews.txt', 'w')
    local_polarities = []
    for polarity in polarities:
        local_polarities.append(polarity)

    english_reviews = []
    # translator = azure_translator.Translator('4be37246201f4a2a80830e97c0384d72')
    translator = googletrans.Translator()
    i = -1
    for review in swedish_reviews:
        i += 1
        if random.randint(1,25) == 1: # get 4% of reviews
            polarity_file.write(str(local_polarities[i]) + '\n')
        # english_reviews.append(translator.translate(review, source_language='sw', to = 'en'))
            translated = ''
            untranslated = ''
            for word in review:
                translated = translated + ' ' + str(word)
                untranslated = translated

            # english_reviews.append(translated)
            translated = translator.translate(translated,dest='en', src='sv').text
            try:
                review_file.write(translated + '\n')
                untranslated_file.write(untranslated + '\n')
            except UnicodeError:
                review_file.write('error')
    review_file.close()
    polarity_file.close()


