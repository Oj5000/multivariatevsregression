import pandas as pd
import os
import zipfile
import numpy as np
from config.DataSetBase import DataSetBase

class CNNPred(DataSetBase):

    def load_data(self):
        name = "CNNPred"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00554/CNNpred.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            os.rename('data/Processed_NASDAQ.csv', 'data/' + name + '.csv')
            print("Done. Cleaning up")

            fname = url.split('/')[-1]
            filenames = zipfile.ZipFile("data/"+fname, 'r').namelist()

            for f in filenames:
                if os.path.exists('data/' + f):
                    os.remove('data/' + f)

            os.remove('data/' + fname)

        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0, na_values='?')
        self.data.drop(columns=['Date', 'Name'], inplace=True)

        self.columns = self.data.columns
        self.mutation_cols = list(self.columns)