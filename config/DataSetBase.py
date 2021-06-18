import os

import requests
import zipfile
import math

import pandas as pd
import numpy as np

class DataSetBase:

    def __init__(self):
        self.data = None
        self.columns = None
        self.sample_original = None

    def load_data(self):
        pass

    def download_data(self, url, name):
        fname = url.split('/')[-1]

        if not os.path.exists("data/"+fname):
            print(fname + " does not exist. Downloading")
            r = requests.get(url)
            
            with open("data/"+fname, 'wb') as f:
                f.write(r.content)
            print("Done")

        if fname.split('.')[-1] == 'zip':
            print("Detected zip file, extracting")

            with zipfile.ZipFile("data/"+fname, 'r') as zip_ref:
                zip_filenames = zip_ref.namelist()
                zip_ref.extractall("data/")
            print("Done")

            csvs = list()

            for f in zip_filenames:
                if f.split('.')[-1] == 'csv':
                    csvs.append(f)

            if len(csvs) == 1:
                os.rename("data/"+csvs[0], "data/"+name+'.csv')

    def preprocess(self):
        # Simple pre-processing
        columns = list(self.data.columns.copy())
        
        # del singular valued cols
        del_fields = []

        for col in columns:
            if len(pd.unique(self.data[col])) == 1:
                del_fields.append(col)

        if len(del_fields) > 0:
            print("Removing fields %s - values all the same" % del_fields)

        self.data = self.data.drop(columns = del_fields)

        for d in del_fields:
            columns.remove(d)

        # We need to impute values for the N/A's - use median value
        for col in columns:
            self.data[col] = self.data[col].replace(['na'], math.nan)
            self.data[col] = pd.to_numeric(self.data[col])
            i_val = np.median(self.data[col].dropna())
            self.data[col].fillna(i_val, inplace=True)

        # check again for singular valued cols
        del_fields = []

        for col in columns:
            if len(pd.unique(self.data[col])) == 1:
                del_fields.append(col)

        if len(del_fields) > 0:
            print("Removing fields %s - values all the same" % del_fields)

        self.data = self.data.drop(columns = del_fields)

        for d in del_fields:
            columns.remove(d)

        # Normalise data
        self.data = self.data - np.mean(self.data)
        self.columns = columns

    def mutate(self, percent_mutation):
        # randomly select data to mutate
        sample = self.data.sample(round(len(self.data)*percent_mutation))
        
        # make a backup of the sampled data
        self.sample_original = self.data.loc[sample.index].copy()

        # Mutate the sample   pd.Index(
        for i in range(len(sample)):
            for j in range(len(sample.columns)):
                sample.iloc[i, j] = 1.2*np.std(sample[sample.columns[j]])*np.random.randn()

        self.data.loc[sample.index] = sample

        return sample.index, self.data

    def restore(self):
        self.data.loc[self.sample_original.index] = self.sample_original