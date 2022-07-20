import pandas as pd
import os
import zipfile
import numpy as np
from config.DataSetBase import DataSetBase

class Accelerometer(DataSetBase):

    def load_data(self):
        name = "Accelerometer"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00611/accelerometer.csv'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            os.rename('data/accelerometer.csv', 'data/' + name + '.csv')
            print("Done. Cleaning up")


        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0)

        self.columns = self.data.columns
        self.mutation_cols = ['x','y','z']