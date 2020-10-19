import pandas as pd

class Ecoli:

	def get_data(self):
		print(type(self).__name__)
		data = pd.read_csv("data/"+type(self).__name__+".csv")

		# Use imL, imS and omL as our outlier class
		# Mark remainder of classes as inliers and merge data sets
		data['class'] = data['class'].replace(['imL', 'imS', 'omL'], 1)
		data['class'] = data['class'].replace(['cp', 'im', 'imU', 'om', 'pp' ], 0)

		columns = list(data.columns[1:8])
		data = data.iloc[:,1:9]

		return data, columns