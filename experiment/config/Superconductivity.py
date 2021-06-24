import pandas as pd
import os
import zipfile
from config.DataSetBase import DataSetBase

class Superconductivity(DataSetBase):

    def load_data(self):
        name = "Superconductivity"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            os.rename('data/train.csv', 'data/' + name + '.csv')
            print("Done. Cleaning up")

            filenames = ['unique_m.csv', 'superconduct.zip']

            for f in filenames:
                if os.path.exists('data/' + f):
                    os.remove('data/' + f)

        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0)
        self.columns = self.data.columns