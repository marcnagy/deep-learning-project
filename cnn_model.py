# Part 1 - Building the CNN
#importing the Keras libraries and packages

from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout
from keras.models import Sequential

# Initialing the CNN
model = Sequential()

# Step 1 - Convolution Layer
model.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3),padding="same", activation ='relu'))
model.add(Conv2D(32, 3, 3,padding="same", activation ='relu'))

#step 2 - Pooling
model.add(MaxPool2D(pool_size =(2, 2),padding="same"))

# Adding second convolution layer
model.add(Conv2D(32, 3, 3, padding="same", activation ='relu'))
model.add(Conv2D(32, 3, 3, padding="same", activation ='relu'))
model.add(MaxPool2D(pool_size =(2, 2), padding="same"))

#Adding 3rd Concolution Layer
model.add(Conv2D(64, 3, 3,padding="same", activation ='relu'))
model.add(Conv2D(64, 3, 3,padding="same", activation ='relu'))
model.add(MaxPool2D(pool_size =(2, 2),padding="same"))


#Step 3 - Flattening
model.add(Flatten())

#Step 4 - Full Connection
model.add(Dense(256, activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation ='softmax'))

#Compiling The CNN
model.compile(
              optimizer = optimizers.SGD(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

val_set = val_datagen.flow_from_directory(
        'mydata/validation_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

model_his = model.fit(
        training_set,
        epochs=100,
        validation_data=val_set
 )
model.predict(test_set)
#Saving the model
model.save("Trained_model.h5")

print(model_his.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model_his.history['accuracy'])
plt.plot(model_his.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model_his.history['loss'])
plt.plot(model_his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








