# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Lexikon


import time


class Dictionary(object):

    def __init__(self, preprocessed_data):
        self.dictionary = dict()
        start = time.time()
        print("Indexing words")
        for post in preprocessed_data:
            for word in post:
                if word not in self.dictionary:
                    self.dictionary[word] = len(self.dictionary)
        time_elapsed = time.time() - start
        print("Indexed " + str(len(self.dictionary)) + " words in ", ("%.2f" % time_elapsed), "seconds")

    def clear(self):
        self.dictionary.clear()