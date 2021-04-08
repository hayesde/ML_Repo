import numpy as np
from joblib import load


with open('fetal_health.pkl', 'rb') as file:
	tree = load(file)



class_names = np.array(['N/A','healthy', 'suspect', 'pathological'])

#print(class_names[1])

def prediction(patient_id):
	data = np.array(patient_id)		
	data = data.reshape(1,-1)	
	prediction = tree.predict(data)
	index = prediction.astype(int)
	final = class_names[index]
	print(final[0])
	return final[0]	

#predict(240.0,0.000,0.000,0.000,0.000,0.0,0.0,73.0,0.5,43.0,2.4,64.0,62.0,126.0,2.0,0.0,120.0,137.0,121.0,73.0,1.0) 
