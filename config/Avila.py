import pandas as pd

class Avila:

	def get_data(self):
		data = pd.read_csv("data/"+type(self).__name__+".csv")

		# Use W as our outlier class
		# Mark remainder of classes as inliers and merge data sets
		data['class'] = data['class'].replace(['W'], 1)
		data['class'] = data['class'].replace(['A','B','C','D','E','F','G','H','I','X','Y'], 0)

		columns = list(data.columns[0:len(data.columns)-1])

		return data, columns