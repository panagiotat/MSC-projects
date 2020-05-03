import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import keras

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer=keras.optimizers.Adam(),
  loss='hinge',
  metrics=['accuracy'],
  
)

# Train the model.
model_history = model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=150,
  batch_size=32,
  validation_data=(test_images, to_categorical(test_labels))
)

figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
plt.plot(a, np.reshape (model_history.history['val_acc'], (150,1) ) , color= "red" )
plt.plot(a, np.reshape (model_history.history['acc'], (150,1) ) , color= "blue")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()