import numpy as np
import pandas as pd

def load_auto():

	# import data
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

	# Extract relevant data features
	X_train = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']].values
	Y_train = Auto[['mpg']].values

	return X_train, Y_train
