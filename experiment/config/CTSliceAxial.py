import pandas as pd
import os
import zipfile
from config.DataSetBase import DataSetBase

class CTSliceAxial(DataSetBase):

    def load_data(self):
        name = "CTSliceAxial"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip'

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

        self.data = pd.read_csv("data/"+name+".csv", sep=',', header=0)

        # drop date and time columns
        self.data.drop(columns=["patientId", "reference"], inplace=True)
        self.columns = self.data.columns

        for c in self.columns:
            try:
                self.data[c] = self.data[c].str.replace(',', '.')
                self.data[c] = pd.to_numeric(self.data[c])
            except:
                pass