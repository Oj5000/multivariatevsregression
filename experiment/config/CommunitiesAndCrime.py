import pandas as pd
import os.path
import os
from config.DataSetBase import DataSetBase

class CommunitiesAndCrime(DataSetBase):

    def load_data(self):
        name = "CommunitiesAndCrime"
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'

        if not os.path.exists("data/"+name+".csv"):
            self.download_data(url, name)
            os.rename("data/communities.data", "data/"+name+".csv")
            print("Done.")

        self.data = pd.read_csv("data/"+name+".csv", sep=',', na_values='?')
        self.columns = list(map(str, list(range(len(self.data.columns)))))

        self.data.columns = self.columns

        self.data.drop(columns=['3'], inplace=True)