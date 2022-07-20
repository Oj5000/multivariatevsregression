import os

import requests
import zipfile
import math
import copy
from numpy import random

import pandas as pd
import numpy as np

class DataSetBase:

    def __init__(self):
        self.data = None
        self.columns = None
        self.sample_original = None

    def load_data(self):
        pass

    def download_data(self, url, name, extension="csv"):
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
                if f.split('.')[-1] == extension:
                    csvs.append(f)
                else:
                    try:
                        os.remove("data/"+f)
                    except:
                        pass

            if len(csvs) == 1:
                os.rename("data/"+csvs[0], "data/"+name+".csv")
            else:
                for csv in range(len(csvs)):
                    os.rename("data/"+csvs[csv], "data/"+csvs[csv].split('/')[-1])

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
            
            if d in self.mutation_cols:
                self.mutation_cols.remove(d)

        # We need to impute values for the N/A's - use median value
        for col in columns:
            self.data[col] = self.data[col].replace(['na'], math.nan)
            self.data[col] = pd.to_numeric(self.data[col])
            i_val = np.median(self.data[col].dropna())
            self.data[col].fillna(i_val, inplace=True)

        # check again for singular valued cols
        del_fields = []

        for col in columns:
            uniques = pd.unique(self.data[col])
            
            # Remove columns with less than 1% variation
            if len(uniques) < (len(self.data) * 0.01):
                del_fields.append(col)

        if len(del_fields) > 0:
            print("Removing fields %s - not enough unique values" % del_fields)

        self.data = self.data.drop(columns = del_fields)

        for d in del_fields:
            columns.remove(d)

            if d in self.mutation_cols:
                self.mutation_cols.remove(d)

        # Normalise data
        if self.data[self.data < 0].any().sum() > 0:
            self.data = self.data + abs(np.min(self.data))

        self.data = (self.data / np.max(self.data)) # normalise the data to stop it flowing out of bounds

        self.columns = columns

    def mutate(self, percent_mutation, mutation_amplitude):
        # randomly select data to mutate
        sample = self.data.sample(round(len(self.data)*percent_mutation))
        
        # make a backup of the sampled data
        self.sample_original = copy.deepcopy(sample)

        # Only mutate specific columns - Remember to put these in!!
        for i in range(len(self.mutation_cols)):
            j = sample.columns.get_loc(self.mutation_cols[i])
            mu = np.mean(sample[sample.columns[j]])
            std_1 = np.std(sample[sample.columns[j]])
            std = mutation_amplitude*np.std(sample[sample.columns[j]])

            #rands = random.uniform(mu-std, mu+std, len(sample))
            rands_1 = random.uniform(mu-std, mu-std_1, len(sample))
            rands_2 = random.uniform(mu+std_1, mu+std, len(sample))
            rands = np.concatenate((rands_1, rands_2), axis=0)
            rands = random.choice(rands, size=len(sample))
            
#            for r in range(len(rands)):
#                if (rands[r] > mu-std_1) & (rands[r] <= mu):
#                    rands[r] = mu-std_1
#                elif (rands[r] > mu) & (rands[r] <= mu+std_1):
#                    rands[r] = mu+std_1

            sample.iloc[:, j] = rands

        self.data.loc[sample.index] = sample

        return sample.index, self.data

    def restore(self):
        self.data.loc[self.sample_original.index] = self.sample_original