import pandas as pd
import shutil
import os
import zipfile
from os import listdir
from os.path import isfile, join
from config.DataSetBase import DataSetBase

class GasEmission(DataSetBase):

    def load_data(self):
        name = "GasEmission"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00551/pp_gas_emission.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            # Data downloads and extracts to its own dir. We want to combine all into one file
            print("Combining data")
            
            fname = url.split('/')[-1]

            filenames = zipfile.ZipFile("data/"+fname, 'r').namelist()
            print(filenames)

            first = True
            with open('data/' + name + '.csv', 'w') as outfile:
                for fname in filenames:
                    with open('data/' + fname) as infile:
                        firstline = True
                        for line in infile:
                            if firstline and first:
                                outfile.write(line)
                                firstline = False
                                first = False
                            elif firstline:
                                firstline = False
                                continue
                            else:
                                outfile.write(line)
                            
            print("Done. Cleaning up")

            for f in filenames:
                os.remove('data/' + f)

            os.remove('data/' + fname)

        self.data = pd.read_csv("data/"+name+".csv", sep=',')
        self.columns = list(range(len(self.data.columns)))

        self.data.columns = self.columns