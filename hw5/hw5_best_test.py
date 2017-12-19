import math
import pandas as pd
import keras
import csv
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Flatten, Merge, Dropout, Dense, Dot, Input, Add, Concatenate, Activation
from keras.models import model_from_json
from keras.models import Sequential, Model
import numpy as np
import sys

def getModel(n_users, m_items, k_factors): #feature dim = 50
    input_users = Input(shape=[1])
    embed_users = Embedding(n_users, k_factors)(input_users)
    flat_users = Flatten()(embed_users)
    # flat_users = Dropout(0.2)(flat_users)

    input_items = Input(shape=[1])
    embed_items = Embedding(m_items, k_factors)(input_items)
    flat_items = Flatten()(embed_items)
    # flat_items = Dropout(0.2)(flat_items)

    input_feature = Input(shape=[48])
    #f_nn = Dense(1024, activation='relu')(input_feature)
    f_nn = Dense(k_factors, activation='relu')(input_feature)


    concat_result = Concatenate()([flat_users, flat_items, f_nn])
    nn = Dense(1024)(concat_result)
    nn = Activation('relu')(nn)
    nn = Dropout(0.5)(nn)
    # nn = Dense(1024)(nn)
    # nn = Activation('relu')(nn)
    # nn = Dropout(0.5)(nn)
    nn = Dense(1)(nn)
    nn_out = Activation('relu')(nn)

    model = Model(inputs=[input_users, input_items, input_feature], outputs=nn_out)

    return model


def writePredict(predictValue,outputFilename):
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['TestDataID','Rating'])
    for n in range(predictValue.shape[0]):
        tmp = predictValue[n,0]
        result_csv.writerow( [str(n+1), str(tmp)] )
    return predictValue


testDataFilename = sys.argv[1]
outputFilename = sys.argv[2]
moviefilename = sys.argv[3]
userfilename = sys.argv[4]

K_FACTORS = 128

max_userid = 6040
max_movieid = 3952

all_user_f = np.loadtxt('all_user_f.txt')
movie_cat_f = np.loadtxt('movie_cat_f.txt')

#######################################################
testData = pd.read_csv(testDataFilename,
                      sep=',',
                      encoding='latin-1',
                      usecols=['TestDataID', 'UserID', 'MovieID'])

Users_test = testData['UserID'].values

Movies_test = testData['MovieID'].values


###################################
x_f_test = np.zeros((Users_test.shape[0],48)) #18+30
for n in range(Users_test.shape[0]): #num of train data
    x_f_test[n,:30] = all_user_f[ Users_test[n],:]
    x_f_test[n,30:48] = movie_cat_f[ Movies_test[n],:]

model = getModel(max_userid, max_movieid, K_FACTORS)
model.load_weights('best_model.h5')

my_pred_value = model.predict([Users_test, Movies_test, x_f_test],batch_size=256)

print('my_pred_value shape', my_pred_value.shape)

writePredict(my_pred_value,outputFilename)
print('finished prediction')
