# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Klassificering


from tpot import TPOTClassifier

class Classifier:
    def __init__(self):
        self.tpot = TPOTClassifier(generations=20, population_size=10, verbosity=3, max_eval_time_mins=1,
                                   periodic_checkpoint_folder='my_checkpoints')

    def fit(self,reviews,polarities):
        self.tpot.fit(reviews, polarities)
        self.tpot.export('tpot_exported_pipeline.py')