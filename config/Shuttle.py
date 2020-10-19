import pandas as pd

class Shuttle:

	def get_data(self):
		print(type(self).__name__)
		data = pd.read_csv("data/"+type(self).__name__+".csv")

		# Class = 1 constitutes 80% class mix
		# Mark remainder of classes as outliers
		data['class'] = data['class'].replace(1, 0)
		data['class'] = data['class'].replace([2, 3, 4, 5, 6, 7], 1)

		columns = list(data.columns[0:len(data.columns)-1])

		return data, columns