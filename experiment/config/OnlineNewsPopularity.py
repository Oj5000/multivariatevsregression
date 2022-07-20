import pandas as pd
import os
import zipfile
import numpy as np
from config.DataSetBase import DataSetBase

class OnlineNewsPopularity(DataSetBase):

    def load_data(self):
        name = "OnlineNewsPopularity"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            print("Done. Cleaning up")

            fname = url.split('/')[-1]
            os.remove('data/' + fname) # remove zip file
            os.rmdir('data/' + fname.split('.zip')[0]) # remove zip folder

        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0)
        self.data.drop(columns=['url'], inplace=True)

        cols = []

        for c in self.data.columns:
            cols.append(c.replace(' ', ''))

        self.columns = cols
        self.data.columns = cols

        self.mutation_cols = list(self.columns)