import pandas as pd

class Banknote:

	def get_data(self):
		data = pd.read_csv("data/"+type(self).__name__+".csv")

		columns = list(data.columns[0:len(data.columns)-1])

		return data, columns