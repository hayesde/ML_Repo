import numpy as np
from joblib import load

tree = load('fetal_health.pkl')

class_names = np.array(['N/A', 'healthy', 'suspect', 'pathological'])

print(class_names[1])

def predict(*patient_id):
	data = np.array(patient_id)
		
	#print(data)
	data = data.reshape(1,-1)	
	prediction = tree.predict(data)
	index = prediction.astype(int)
	return class_names[index]	

	#print(class_names[index])


predict(120.0,0.000,0.000,0.000,0.000,0.0,0.0,73.0,0.5,43.0,2.4,64.0,62.0,126.0,2.0,0.0,120.0,137.0,121.0,73.0,1.0) 
