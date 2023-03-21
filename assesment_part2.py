import zipfile

with zipfile.ZipFile('car_bike_data.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import random
import tensorflow as tf

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'car_bike_data/train'
test_dir = 'car_bike_data/test'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=4,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=4,
        class_mode='binary')

model = Sequential()

model.add(Conv2D(10, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(10, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), padding='valid'))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=20,  
      epochs=5,
      validation_data=test_generator,
      validation_steps=5,
      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

test_loss, test_acc = model.evaluate(test_generator, steps=5)
print('Test accuracy:', test_acc)
