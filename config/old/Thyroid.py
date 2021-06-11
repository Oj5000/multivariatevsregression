import pandas as pd

class Thyroid:

    def get_data(self):
        header = ['class','T3resin','Thyroxin','Triiodothyronine','Thyroidstimulating','TSH_value']
        data = pd.read_csv("data/"+type(self).__name__+".csv", header=0, names=header)
        
        # Move target column and re-label classes 1 to 0, and 2 and 3 to a 1
        data['class'] = data['class'].replace([1], 0)
        data['class'] = data['class'].replace([2,3], 1)

        columns = list(data.columns[1:6])

        return data, columns