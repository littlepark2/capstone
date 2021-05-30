import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels = len(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_size = image_size * image_size

x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

batch_size = 128
hidden_units = 256
dropout = 0.2

model = Sequential()
model.add(Dense(hidden_units, input_dim = input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))

model.add(Activation('softmax'))
model.summary()
plot_model(model, 
		   to_file = 'mlp-mnist.png', 
		   show_shapes = True)

model.compile(loss = 'categorical_crossentropy', 
			  optimizer = 'adam', 
			  metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 10, batch_size = batch_size)

loss, acc = model.evaluate(x_test, y_test, batch_size = batch_size)
print("\nTest accuracy: %.1f%%" %(100.0*acc))
