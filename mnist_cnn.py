

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras import backend as K
import pickle
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
# Where to save the model

dir1 = "/Users/Dhanush/Documents/Deeplearn/MathematicalSymbolRecognition/Models/"

batch_size = 128
nb_classes = 82
nb_epoch = 25

# input image dimensions
#img_rows, img_cols = 28, 28
img_rows, img_cols = 45, 45
# number of convolutional filters to use
nb_filters = 48
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print (X_train.shape)
f = open("MathSymbols_train_test", "rb")
(X_train, y_train, X_test, y_test,X_cv,y_cv)= pickle.load(f)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_cv = X_cv.reshape(X_cv.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_cv = X_cv.reshape(X_cv.shape[0], 1, img_rows, img_cols)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_cv = X_cv.astype('float32')

X_train /= 255
X_cv/=255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_cv.shape[0], 'cv samples')
print(X_train[0].shape)
# convert class vectors to binary class matrices

#Convert the labels to number labels before Binarizing.

le = preprocessing.LabelEncoder()
y_train =le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_cv = le.fit_transform(y_cv)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_cv = np_utils.to_categorical(y_cv, nb_classes)
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
#model.add(Activation('relu'))
model.add(ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              #optimizer='adadelta',
              optimizer='adam',
              metrics=['accuracy'])

#Save the model after each epoch
filepath = dir1+'ModelSave-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=2, min_lr=0.001)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_cv, Y_cv),callbacks=[checkpointer,reduce_lr])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])