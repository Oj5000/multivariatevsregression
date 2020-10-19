import pandas as pd

class Fertility:

	def get_data(self):
		print(type(self).__name__)
		data = pd.read_csv("data/"+type(self).__name__+".csv")
		
		data = data.rename(index=str, columns={"diagnosis": "class"})
		
		# Update class labels
		data['class'] = data['class'].replace('N', 0)
		data['class'] = data['class'].replace('O', 1)

		columns = list(data.columns[0:len(data.columns)-1])

		return data, columns