import pandas as pd

class Epilepsy:

    def get_data(self):
        data = pd.read_csv("data/"+type(self).__name__+".csv")

        # Use 1 as our outlier class
        # Mark remainder of classes as inliers
        data['y'] = data['y'].replace([2, 3, 4, 5], 0)
        data = data.rename(columns={"y" : "class"})

        columns = list(data.columns[1:len(data.columns)-1])
        data = data.iloc[1:,1:len(data.columns)]

        return data, columns