import os
import random
from os import listdir
from os.path import isfile, join
from DataSetBase import DataSetBase

class ClickStreamShopping(DataSetBase):

    def load_data(self):
        name = "ClickStreamShopping"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00553/e-shop%20data%20and%20description.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            print("Done. Cleaning up")
            filenames = [f for f in listdir('data/') if isfile(join('data/', f))]

            for f in filenames:
                if f.split('.')[-1] != 'csv':
                    os.remove('data/' + f)

            print("Done.")

        self.data = pd.read_csv("data/"+name+".csv", sep=';')
        self.columns = data.columns

    # override
    def mutate(self, data, percent_mutation):
        # randomly select data to mutate
        sample = data.sample(round(len(data)*percent_mutation))

        # make a backup of the sampled data
        sample_original = data.loc[sample.index].copy()

        # Mutate the sample. In this case we are dealing with categoric data.
        for c in sample.columns:
            vals = np.unique(sample[c])

            if len(vals) < (len(data)*0.5):
                sample[c] = random.choice(vals)
            else:
                sample[c] = 1.2*sample[c]*np.random.randn()