# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Lexikon


import time


class Dictionary(object):

    def __init__(self, post_array):
        self.dictionary = dict()
        start = time.time()
        print("Indexing words")
        for post in post_array:
            for word in post:
                if word not in self.dictionary:
                    self.dictionary[word] = len(self.dictionary)
        time_elapsed = time.time() - start
        print("Indexing completed in ", ("%.2f" % time_elapsed), "seconds")

    def clear(self):
        self.dictionary.clear()