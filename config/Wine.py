import pandas as pd
import numpy as np

class Wine:

	def get_data(self):
		data = pd.read_csv("data/"+type(self).__name__+".csv")

		# Cleaning
		data = data.rename(index=str, columns={"quality": "class"})
		data = data.drop(columns='type')

		# Remove class from analysis
		columns = list(data.columns[0:len(data.columns)-1])

		# Update class labels, quality <=4 and >= 8 classed as outliers
		data['class'] = np.where(((data['class'] <= 4) | (data['class'] >= 8)), 1, data['class']) #data['class'].loc[(data['class'] <= 4) | (data['class'] >= 8)] = 1
		data['class'] = np.where((data['class'] > 1), 0, data['class']) #data['class'].loc[(data['class'] > 1) ] = 0

		return data, columns