import pandas as pd
import os
import zipfile
import numpy as np
import copy
from config.DataSetBase import DataSetBase

class BiasCorrection(DataSetBase):

    def load_data(self):
        name = "BiasCorrection"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            os.rename('data/Bias_correction_ucl.csv', 'data/' + name + '.csv')
            print("Done. Cleaning up")

        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0)
        self.data.drop(columns=['Date'], inplace=True)

        columns = []

        for c in self.data.columns:
            if c == 'Solar radiation':
                columns.append('Solar_radiation')
            else:
                columns.append(c)

        self.data.columns = columns
        self.columns = columns

        self.mutation_cols = ["Present_Tmax","Present_Tmin","LDAPS_RHmin","LDAPS_RHmax","LDAPS_Tmax_lapse","LDAPS_Tmin_lapse","LDAPS_WS","LDAPS_LH","LDAPS_CC1","LDAPS_CC2","LDAPS_CC3","LDAPS_CC4","LDAPS_PPT1","LDAPS_PPT2","LDAPS_PPT3","LDAPS_PPT4","Solar_radiation","Next_Tmax","Next_Tmin"]