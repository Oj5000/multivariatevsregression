import pandas as pd

class Yeast:

	def get_data(self):
		data = pd.read_csv("data/"+type(self).__name__+".csv")
		columns = list(data.columns[0:8])

		return data, columns