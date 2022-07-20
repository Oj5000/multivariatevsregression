import pandas as pd
import os
import zipfile
import numpy as np
from config.DataSetBase import DataSetBase

class WaveEnergyConverters(DataSetBase):

    def load_data(self):
        name = "WaveEnergyConverters"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00494/WECs_DataSet.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            os.rename('data/Perth_Data.csv', 'data/' + name + '.csv')
            print("Done. Cleaning up")

            fname = url.split('/')[-1]
            filenames = zipfile.ZipFile("data/"+fname, 'r').namelist()

            for f in filenames:
                if os.path.exists('data/' + f.split('/')[-1]):
                    if len(f.split('.csv')) > 1:
                        os.remove('data/' + f.split('/')[-1])

            os.rmdir('data/' + filenames[0].split('/')[0])
            os.remove('data/WECs_DataSet.zip')

        cols = []

        for p in ['X', 'Y', 'P']:
            for i in range(1, 17):
                cols.append("%s%i" % (p, i))
            
        cols.append("Powerall")

        self.data = pd.read_csv("data/"+name+".csv", sep=',', names=cols)

        self.columns = self.data.columns
        self.mutation_cols = list(self.columns)