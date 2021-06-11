import pandas as pd
import os.path
from DataSetBase import DataSetBase

class QSARFishToxicity(DataSetBase):

    def load_data(self):
        name = "QSARFishToxicity"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)
            print("Done.")

        self.data = pd.read_csv("data/"+name+".csv")
        self.columns = data.columns