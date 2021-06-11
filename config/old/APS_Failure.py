import os.path
import pandas as pd
import requests

class APS_Failure:

    def get_data(self):
        if not os.path.exists("data/"+type(self).__name__+".csv"):
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv'
            r = requests.get(url)

            with open("data/"+type(self).__name__+".csv", 'wb') as f:
                f.write(r.content)

        data = pd.read_csv("data/"+type(self).__name__+".csv", skiprows=range(0,20,1))

        # Use pos as our outlier class
        data['class'] = data['class'].replace(['neg'], 0)
        data['class'] = data['class'].replace(['pos'], 1)

        columns = list(data.columns[1:len(data.columns)])

        return data, columns