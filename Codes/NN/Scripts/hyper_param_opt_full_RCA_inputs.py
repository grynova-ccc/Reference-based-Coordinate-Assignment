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
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from keras.layers import concatenate

dictionary_NN={}
i=0
for n1l1 in [300,400,500]:
    for n1l2 in [300,400,500]:
        for n1l3 in [20,50,70]:
            for n2l1 in [100,150,200]:
                for n2l2 in [75,100,125]:
                    for n2l3 in [20,50,70]:
                        for n1 in [50,80,110]:
                            for opti in ['adam','SGD']:
                                dictionary_NN[i]=[n1l1,n1l2,n1l3,n2l1,n2l2,n2l3,n1,opti]
                                i=i+1
dictionary_NN=dict(list(dictionary_NN.items())[1458+1458: 1458+1458+1458])

i=0
new_tab={}
for q in range(0,30):
    new_tab[i]=dict(list(dictionary_NN.items())[49*q: 49+49*q])
    i=i+1

def deep_NN_combined_full_RCA (n1l1,n1l2,n1l3,n2l1,n2l2,n2l3,n1,opti):
    # Load arrays
    full_input=np.load('PATH/input_full.npy')
    RCA_input=np.load('PATH/input_RCA.npy')
    output=np.load('PATH/labels.npy')

    categories, inverse = np.unique(output, return_inverse=True)
    # Create the one-hot encoded matrix
    one_hot = np.zeros((output.size, categories.size))
    one_hot[np.arange(output.size), inverse] = 1
    output=one_hot

    x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(full_input, output, test_size=0.33, random_state=137)
    x_train_RCA, x_test_RCA, y_train_RCA, y_test_RCA = train_test_split(RCA_input, output, test_size=0.33, random_state=137)

    #Full images
    input_full = Input(shape=(784,))
    x = Dense(784, activation="relu")(input_full)
    x = Dense(n1l1, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n1l2, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n1l3, activation="relu")(x)
    x = Model(inputs=input_full, outputs=x)

    #RCA reduced images
    input_RCA = Input(shape=(19,))
    y = Dense(19, activation="relu")(input_RCA)
    y = Dense(n2l1, activation="relu")(y)
    y = Dropout(0.5)(y)
    y = Dense(n2l2, activation="relu")(y)
    y = Dropout(0.5)(y)
    y = Dense(n2l3, activation="relu")(y)
    y = Model(inputs=input_RCA, outputs=y)

    combined = concatenate([x.output, y.output])

    z = Dense(n1, activation="relu")(combined)
    z = Dense(n1, activation="relu")(z)
    z = Dense(n1, activation="relu")(z)
    z = Dense(10, activation='softmax')(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss='binary_crossentropy',metrics=['acc'], optimizer=opti)

    model.fit(x=[x_train_full[:2000], x_train_RCA[:2000]], y=y_train_full[:2000],validation_data=([x_test_full[:2000], x_test_RCA[:2000]], y_test_full[:2000]),epochs=10, batch_size=16)

    return ()

def run_everything_in_parallel(dic_no=None):
    for i in new_tab[dic_no]:
        ele_dic=new_tab[dic_no][i]
        deep_NN_combined_full_RCA(ele_dic[0],ele_dic[1],ele_dic[2],ele_dic[3],ele_dic[4],ele_dic[5],ele_dic[6],ele_dic[7])

results = Parallel(n_jobs=30)(delayed(run_everything_in_parallel)(q) for q in [m for m in range(0,30)])
