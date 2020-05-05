import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras import layers, models, optimizers
from keras import backend as K
from keras.models import load_model
import helper
from keras.models import Sequential
from keras.datasets import cifar10 #data set
import matplotlib
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint




(img_train, label_train), (img_test, label_test) = cifar10.load_data() #load data
num_classes = 10

label_train = keras.utils.to_categorical(label_train, num_classes)
label_test = keras.utils.to_categorical(label_test, num_classes)
#categoreis = get_class_names()
#print(categories)

print("Training size: ", len(img_train)) #training size
print("Testing size: ", len(img_test))  #test size

input_shape = (32,32,3) #this is given, size of image and rgb channels

def CNNmodel():
    model = Sequential() #linear stack of layers

    model.add(Conv2D(32, (3,3), activation='relu', padding = 'same', input_shape = input_shape)) #number of filters, kernal size (2d convolutional matrix),padding helps accuracy.rows columns channels
    model.add(MaxPooling2D(pool_size=(2,2))) #downsample to detect features
    model.add(Dropout(0.25)) #prevent overfittting

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) #more conv layers
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) #prepare for dense network

    model.add(Dense(512, activation='relu')) #classic ANN
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax')) #could also use relu? ,10 classes think

    model.summary()

    return model
model = CNNmodel()



checkpoint = ModelCheckpoint('best_model.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only= True,
                             mode='auto') \

model.compile(loss='categorical_crossentropy', #reputible loss function
              optimizer=Adam(lr=1.0e-4), #Adaptive momentum estimater
              metrics = ['accuracy'])

history = model.fit(img_train, label_train,
                    batch_size = 128, #ratio with learning rate
                    epochs = 50, #numnber of times run through the training data
                    validation_data= (img_test, label_test),
                    callbacks=[checkpoint],
                    verbose=1)


print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
