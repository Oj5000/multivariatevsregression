import pandas as pd

class Secom:

	def get_data(self):
		print(type(self).__name__)
		data = pd.read_csv("data/"+type(self).__name__+".csv")
		labels = pd.read_csv("data/"+type(self).__name__+"_labels.csv")

		labels = labels.drop(columns='date')

		# Update class labels
		labels['class'] = labels['class'].replace(-1, 0)

		# Merge data sets
		data['class'] = labels['class']

		columns = list(data.columns[0:len(data.columns)-1])

		return data, columns