from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.layers import concatenate

# Load arrays
RCA_input=np.load('PATH/input_RCA.npy')
output=np.load('PATH/labels.npy')

categories, inverse = np.unique(output, return_inverse=True)
# Create the one-hot encoded matrix
one_hot = np.zeros((output.size, categories.size))
one_hot[np.arange(output.size), inverse] = 1
output=one_hot

# Train/test split
x_train_RCA, x_test_RCA, y_train_RCA, y_test_RCA = train_test_split(RCA_input, output, test_size=0.33, random_state=137)

#Full images
input_RCA = Input(shape=(19,))
x = Dense(19, activation="relu")(input_RCA)
x = Dense(100, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(75, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(50, activation="relu")(x)
x = Dense(80, activation="relu")(x)
x = Dense(80, activation="relu")(x)
x = Dense(20, activation="relu")(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=input_RCA, outputs=x)


model.compile(loss='binary_crossentropy',metrics=['acc'], optimizer='adam')

model.fit(
	x=x_train_RCA, y=y_train_RCA,
	validation_data=(x_test_RCA, y_test_RCA),
	epochs=200, batch_size=16)


