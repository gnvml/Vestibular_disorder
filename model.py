# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


if __name__=="__main__":
	
	# preparing data
	df = pd.read_csv("suffle_file.csv")
	data = df.values

	print(type(data))
	print(data.shape)

	label = data[:,-1]
	label[label == -1] = 0

	x_train, y_train = data[:100,:-1], label[:100]
	x_test, y_test = data[100:,:-1], label[100:]
	print(x_train.shape, y_train.shape)

	# Build model

	model = tf.keras.models.Sequential([
			#Input layer
			tf.keras.layers.Dense(16, activation="relu", input_shape=(12,)),
	
			#Hidden layer
			tf.keras.layers.Dense(12, activation="relu"),
			tf.keras.layers.Dense(12, activation="relu"),
			tf.keras.layers.Dense(12, activation="relu"),
			tf.keras.layers.Dense(8, activation="relu"),
			
			#ouput layer
			tf.keras.layers.Dense(1, activation="sigmoid")
			])

	# model.summary()

	model.compile(optimizer=tf.keras.optimizers.Adam(),
						loss='binary_crossentropy',
						metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=2000)

	model.evaluate(x_test, y_test)

	suffix = "first commit"
	# save_model(model, "{epoch}_{suffix}".format(epoch=1500, suffix=suffix))
