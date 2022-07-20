import pandas as pd
import os
import zipfile
import numpy as np
from config.DataSetBase import DataSetBase

class HouseholdPowerConsumption(DataSetBase):

    def load_data(self):
        name = "HouseholdPowerConsumption"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name, extension="txt")
            print("Done. Cleaning up")

            fname = url.split('/')[-1]
            filenames = zipfile.ZipFile("data/"+fname, 'r').namelist()

            for f in filenames:
                if os.path.exists('data/' + f):
                    os.remove('data/' + f)

            os.remove('data/' + fname)
            os.remove('data/HouseholdPowerConsumption.txt')

        self.data = pd.read_csv("data/"+name+".csv", sep=';', header=0, na_values='?')

        self.data.drop(columns=['Date','Time'], inplace=True)
        self.columns = self.data.columns
        self.mutation_cols = ['Global_reactive_power','Voltage','Global_intensity']