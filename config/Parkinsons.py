import pandas as pd

class Parkinsons:

	def get_data(self):
		data = pd.read_csv("data/"+type(self).__name__+".csv")

		data = data.drop(columns='id')

		# invert classes
		data['class'] = data['class'].replace(1, 2)
		data['class'] = data['class'].replace(0, 1)
		data['class'] = data['class'].replace(2, 0)

		columns = list(data.columns[0:len(data.columns)-1])

		return data, columns