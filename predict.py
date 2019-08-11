from tensorflow.keras.models import model_from_json
import numpy as np
import sys

def read_model(json_name, weight_name):
	json_name = "cache/" + json_name
	weight_name = "cache/" +  weight_name
	model = model_from_json(open(json_name).read())
	model.load_weights(weight_name)
	
	return model


if __name__=='__main__':

	#Read model to predict
	model = read_model(json_name = "architecture_1500_first commit.json", weight_name = "model_weights_1500_first commit.h5")

	# read input vetor 18 dimension
	# input type: a Sring, 
		# for example: 
		# 1.3788,0.8985,0.8985,1.5141,1.0626,1.0786,50.7792,34.4852,30.2808,2.539,1.7243,1.514,0.2931,0.3054,0.2682,7.5649,4.7938,1.0861
 
	vector = sys.argv[1]
	# convert to numpy array
	vector = np.fromstring(vector, dtype=float, sep=',')
	vector = np.expand_dims(vector, axis = 1)
	vector = np.transpose(vector,(1,0))

	# predict: 1: vestibular, 0: normal
	y = model.predict(vector)
	print(np.round(y)[0][0])