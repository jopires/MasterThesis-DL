import tensorflow as tf
tf.python.control_flow_ops = tf

import time
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 2017
np.random.seed(seed)

# load data
# X = features; y = labels
(X_train, y_train), (X_test, y_test) = mnist.load_data()
_, img_rows, img_cols =  X_train.shape
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
num_classes = len(np.unique(y_train))
num_input_nodes = img_rows*img_cols
print "Number of training samples: %d"%X_train.shape[0]
print "Number of test samples: %d"%X_test.shape[0]
print "Image rows: %d"%X_train.shape[1]
print "Image columns: %d"%X_train.shape[2]
print "Number of classes: %d"%num_classes

# Preprocessing
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
print('train_features shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Function to plot model accuracy and loss
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
# Funtion to compute test accuracy
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

def larger_model():
	# create model
	model = Sequential()
	#model1 = Sequential()
	#model1.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28))) 
	#model1.add(Activation("relu"))
	#model1.add(MaxPooling2D(pool_size=(2, 2)))
	#model1.add(Flatten())
	#model1.add(Dense(num_classes))
	#model1.add(Activation("softmax"))
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Convolution2D(15, 3, 3, activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.2))
	model.add(Flatten())
	#model.add(Dense(128, activation='relu'))
	#model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()

# Train the model
start = time.time()
model_info = model.fit(X_train, y_train, batch_size=100, \
                         nb_epoch=100, verbose=1, validation_split=0.2)
end = time.time()

# plot model history
plot_model_history(model_info)
print "Model took %0.2f seconds to train"%(end - start)


# Fit the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=100, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# build the model
#model = larger_model()
# Fit the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))



# compute test accuracy
print "Accuracy on test data is: %0.2f"%accuracy(X_test, y_test, model)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

