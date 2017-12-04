from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
 
model = Sequential()
model.add(Dense(input_dim=784, output_dim=10))
model.add(Activation("softmax"))
 
sgd = SGD(lr=0.5, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
model.fit(mnist.train.images, mnist.train.labels, nb_epoch=1, batch_size=100)
 
loss_and_metrics = model.evaluate(mnist.test.images, mnist.test.labels)
print(loss_and_metrics[1]) # Accuracy
