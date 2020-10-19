import pandas as pd

class Wisconsin:

	def get_data(self):
		print(type(self).__name__)
		data = pd.read_csv("data/"+type(self).__name__+".csv")
		columns = list(data.columns[0:9])

		return data, columns