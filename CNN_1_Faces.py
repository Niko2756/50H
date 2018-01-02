from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPool2D, AvgPool2D
from keras.layers import Flatten, Activation, Dropout
from keras.utils.data_utils import get_file
import pathlib

def main():
	generator = ImageDataGenerator(rescale = 1./255,
								   width_shift_range=0.1,
								   height_shift_range=0.1,
								   rotation_range = 180,
								   shear_range = 0.3,
								   zoom_range = 4.3,
								   horizontal_flip = True)

	train = generator.flow_from_directory('/Users/Niko/Desktop/50H_project_faceEmotion/learnfaces',
										  target_size = (256, 256),
										  batch_size = 10,
										  class_mode = 'categorical',
										  save_to_dir = pathlib.Path("/Users/Niko/Desktop/50H_project_faceEmotion/examples_of_pictures").mkdir(parents=True, exist_ok=True),
										  save_format = "jpeg")
	print(train.class_indices)
	whatByWhatStridesBox = 3
	numOfSteps = 2000
	numOfEpochs = 20000
	#TF_WEIGHTS_PATH = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'

	model = Sequential()

	model.add(Conv2D(32, (whatByWhatStridesBox, whatByWhatStridesBox), activation='relu', input_shape=(256, 256, 3), padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (whatByWhatStridesBox, whatByWhatStridesBox), activation='relu', input_shape=(256, 256, 3), padding = 'same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (whatByWhatStridesBox, whatByWhatStridesBox), activation='relu', input_shape=(256, 256, 3),  padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (whatByWhatStridesBox, whatByWhatStridesBox), activation='relu', input_shape=(256, 256, 3),  padding = 'same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(len(train.class_indices)))
	model.add(Activation('softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	#weights_path = get_file('inception-v4_weights_th_dim_ordering_th_kernels.h5', TF_WEIGHTS_PATH, cache_subdir='models')
	#model.load_weights(weights_path, by_name=True)

	model.fit_generator(generator = train,
		steps_per_epoch=numOfSteps, epochs=numOfEpochs, verbose=1, use_multiprocessing = True)
		#batch_size=20
		#validation_data = train,

	#model.fit(X_train, y_train, batch_size=20,
		  #epochs=2, verbose=1, validation_split=0.3)

	score = model.evaluate_generator(steps = numOfSteps, generator = train, use_multiprocessing = True)
	print(f'Test loss: {score[0]}')
	print(f'Test accuracy: {score[1]}')

	model.save_weights("my_model_weights.h5")
	model.save("faces_MLmodel.h5")

	print("\n\ngot to final piece of the puzzle models and weights have been saved-- DONE\n")
main()