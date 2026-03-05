from sklearn import preprocessing
import numpy as np
from pandas import DataFrame
import pandas as pd

def normalization(data):
	shape = data.shape
	aux = data[:]
	originalValues = []
	band = False
	for i in range(shape[1]):
		vals = data[:,i]
		if isinstance(vals[0], str):
			vals, relations = StringToNumber(vals)
			aux = DataFrame(aux)
			aux[i] = vals
			aux = np.array(aux)
			band = True
		minValue = min(vals.tolist())
		maxValue = max(vals.tolist())
		if band:
			originalValues.append((minValue, maxValue, relations))
			band = False
		else:
			originalValues.append((minValue, maxValue))
		normalized_info = preprocessing.minmax_scale(aux, feature_range=(0,1))
		return normalized_info, originalValues

def StringToNumber(data):
	vals = list(set(data))
	aux = data[:]
	relations = []
	for i in range(len(vals)):
		relations.append((vals[i], i))
		aux = [x if x != vals[i] else i for x in aux]
	return np.array(aux), relations

def numberToString(data, relations):
	new_data = data[:]
	for relation in relations:
		val = relation[0]
		number = relation[1]
		new_data = [x if x != number else val for x in new_data]
	return new_data

def SingleDenormalization(data, ranges):
	denorm = preprocessing.minmax_scale(data, feature_range=(ranges[0], ranges[1]))
	try:
		int_denorm = denorm.astype(int)
		denorm = numberToString(int_denorm, ranges[2])
	except:
		denorm = np.round(denorm, 2)
	return denorm

def checkNormalization(data, ranges, org):
	denorm = preprocessing.minmax_scale(data, feature_range=(ranges[0], ranges[1]))
	try:
		int_denorm = denorm.astype(int)
		denorm = numberToString(int_denorm, ranges[2])
	except:
		denorm = np.round(denorm, 2)
	for i in range(len(data)):
		if denorm[i] != org[i]:
			print("F. Valor denorm:", denorm[i], "Valor original:", org[i])