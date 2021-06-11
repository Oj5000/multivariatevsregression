import pandas as pd
import shutil
import os
from os import listdir
from os.path import isfile, join
from DataSetBase import DataSetBase

class WaveEnergyConverters(DataSetBase):

    def load_data(self):
        name = "WaveEnergyConverters"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00494/WECs_DataSet.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)
            
            # Data downloads and extracts to its own dir. We want to combine all into one file
            print("Combining data")
            
            datapath = 'data/'+url.split('/')[-1].split('.')[0]
            filenames = [f for f in listdir(datapath) if isfile(join(datapath, f))]
            with open('data/' + name + '.csv', 'w') as outfile:
                for fname in filenames:
                    with open(datapath + '/' + fname) as infile:
                        for line in infile:
                            outfile.write(line)
                            
            print("Done. Cleaning up")

            # Delete zip and extracted files
            if os.path.exists("data/"+url.split('/')[-1]):
                os.remove("data/"+url.split('/')[-1])

            if os.path.exists("data/"+url.split('/')[-1].split('.')[0]):
                shutil.rmtree("data/"+url.split('/')[-1].split('.')[0])
            
            print("Done")

        self.data = pd.read_csv("data/"+name+".csv")
        self.columns = self.data.columns