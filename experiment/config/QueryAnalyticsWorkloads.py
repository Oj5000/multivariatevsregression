import pandas as pd
import os
import zipfile
import numpy as np
from config.DataSetBase import DataSetBase

class QueryAnalyticsWorkloads(DataSetBase):

    def load_data(self):
        name = "QueryAnalyticsWorkloads"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00493/datasets.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            os.rename('data/Radius-Queries-Count.csv', 'data/' + name + '.csv')
            print("Done. Cleaning up")

            fname = url.split('/')[-1]
            filenames = zipfile.ZipFile("data/"+fname, 'r').namelist()

            for f in filenames:
                if os.path.exists('data/' + f):
                    os.remove('data/' + f)
                elif os.path.exists('data/' + f.split('/')[-1]):
                    os.remove('data/' + f.split('/')[-1])

            os.rmdir('data/' + filenames[0].split('/')[0])
            os.remove('data/datasets.zip')

        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0)

        self.columns = self.data.columns
        self.mutation_cols = list(self.columns)