p# Deep Learning
'''
Keras

Input mapped to Output
'''

if 0:  # Basic Tensorflow
	# Import
	import numpy as np
	import matplotlib.pyplot as plt
	import tensorflow as tf
	print(tf.__version__)

	# Data
	mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# print(np.shape(x_train[0]))  # 28x28 array
	# plt.imshow(x_train[0], cmap='gray')  # looks like a 5
	# plt.show()

	x_train = tf.keras.utils.normalize(x_train, axis=1)
	x_test = tf.keras.utils.normalize(x_test, axis=1)

	# Model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # inital layer - rectify linear
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # second layer
	model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=3)

	val_loss, val_acc = model.evaluate(x_test, y_test)
	print(val_loss, val_acc)

	# Store
	#model.save('epic_num_reader.model')
	#new_model = tf.keras.models.load_model('epic_num_reader.model')
	predictions = model.predict([x_test])

	print(np.shape(predictions))

	plt.figure(1)
	for i in range(len(predictions)):
		plt.clf()
		plt.imshow(x_test[i], cmap='gray')
		plt.title(np.argmax(predictions[i]))
		plt.pause(1)


if 1:  # Data Cleaning
	# Import
	import numpy as np
	import matplotlib.pyplot as plt

	# Data
	def gen_data(IMG_SIZE):
		import os
		import cv2
		DATADIR = 'C:/Users/adam/Downloads/PetImages'
		CATEGORIES = ['Dog', 'Cat']
		data = []
		for category in CATEGORIES:
			path = os.path.join(DATADIR, category)
			class_num = CATEGORIES.index(category)
			for img in os.listdir(path):
				try:
					img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
					new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
					# plt.figure(1)
					# plt.imshow(img_array, cmap='gray')
					# plt.figure(2)
					# plt.imshow(new_array, cmap='gray')
					# plt.show()
					data.append([new_array, class_num])
				except Exception as img_cv2_error:
					pass
		np.random.shuffle(data)
		return data

	try:
		print('Loading Data')
		data = np.load('data.npy')
	except Exception as img_not_yet_saved_error:
		print('Creating Data, please wait')
		data = gen_data(IMG_SIZE=100)
		np.save('data', data)
	print('Data ready')


if 1:  # Modelling
	# Import
	import tensorflow as tf
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
	from tensorflow.keras.utils import normalize

	# Data
	print('Normalising Data')
	X = normalize(data[:, 0])
	Y = data[:, 1]

	print('Creating Model')
	model = Sequential()
	# 1x64 Conv Network
	model.add(Conv2D(64, (3, 3), input_shape=X.shape[0]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	# 2x64 Conv Network
	model.add(Conv2D(64), (3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	# 3rd Layer: Organise Layer
	model.add(Dense(64))
	# Output Layer
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	# Run Model
	print('Compiling Model')
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Fitting Model')
	model.fit(X, Y, batch_size=32, epochs=1, validation_split=.1)
