import pandas as pd
import os.path
import os
from config.DataSetBase import DataSetBase

class QSARFishToxicity(DataSetBase):

    def load_data(self):
        name = "QSARFishToxicity"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)
            os.rename("data/qsar_fish_toxicity.csv", "data/"+name+".csv")
            print("Done.")

        self.data = pd.read_csv("data/"+name+".csv", sep=';')
        self.columns = list(range(len(self.data.columns)))

        self.data.columns = self.columns
        self.mutation_cols = list(self.columns)