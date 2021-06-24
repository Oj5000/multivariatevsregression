import pandas as pd
import os
import zipfile
from config.DataSetBase import DataSetBase

class AirQuality(DataSetBase):

    def load_data(self):
        name = "AirQuality"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)

            fname = url.split('/')[-1]

            filenames = zipfile.ZipFile("data/"+fname, 'r').namelist()

            for f in filenames:
                if f.split('.')[1] == 'csv':
                    print("Renaming data/" + f + " to data/"+name+".csv")
                    try:
                        os.rename('data/' + f, 'data/' + name + '.csv')
                    except:
                        pass
                            
            print("Done. Cleaning up")

            for f in filenames:
                if os.path.exists('data/' + f):
                    os.remove('data/' + f)

            os.remove('data/' + fname)

        self.data = pd.read_csv("data/"+name+".csv", sep=';', skip_blank_lines=True)

        # drop date and time columns
        self.data.drop(columns=["Date", "Time", 'Unnamed: 15', 'Unnamed: 16'], inplace=True)
        self.columns = self.data.columns

        for c in self.columns:
            try:
                self.data[c] = self.data[c].str.replace(',', '.')
                self.data[c] = pd.to_numeric(self.data[c])
            except:
                pass