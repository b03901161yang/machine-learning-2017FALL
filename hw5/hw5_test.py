import math
import pandas as pd
import keras
import csv
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Flatten, Merge, Dropout, Dense, Dot, Input, Add
from keras.models import model_from_json
from keras.models import Sequential, Model
import sys



def get_CFModel(n_users, m_items, k_factors):
    input_users = Input(shape = [1])
    embed_users = Embedding(n_users, k_factors, input_length=1,embeddings_initializer = 'random_normal')(input_users)
    flat_users = Flatten()(embed_users)

    input_items = Input(shape = [1])
    embed_items = Embedding(m_items, k_factors, input_length=1,embeddings_initializer = 'random_normal')(input_items)
    flat_items = Flatten()(embed_items)

    dot_result = Dot(axes = 1, normalize=False)([flat_users,flat_items])
    #################add bias
    bias_users = Embedding(n_users, 1, input_length=1, embeddings_initializer='random_normal')(input_users)
    flat_bias_users = Flatten()(bias_users)
    bias_items = Embedding(m_items, 1 , input_length=1, embeddings_initializer='random_normal')(input_items)
    flat_bias_items = Flatten()(bias_items)

    dot_result = Add()([dot_result, flat_bias_users, flat_bias_items])
    #################
    model = Model(inputs=[input_users, input_items], outputs=dot_result)

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

userfilename = sys.argv[3]
moviefilename = sys.argv[4]

K_FACTORS = 128

testData = pd.read_csv(testDataFilename,
                      sep=',',
                      encoding='latin-1',
                      usecols=['TestDataID', 'UserID', 'MovieID'])
###from training
max_userid = 6040
max_movieid = 3952
print( len(testData), 'test data loaded.')


Users = testData['UserID'].values
print('Users:', Users, ', shape =', Users.shape)
Movies = testData['MovieID'].values
print('Movies:', Movies, ', shape =', Movies.shape)

###################################


model = get_CFModel(max_userid, max_movieid, K_FACTORS)
model.load_weights('mf_model.h5')

my_pred_value = model.predict([Users, Movies],batch_size=256)
'''
mu_ratings =  3.58171208604
sig_ratings = 1.11689766115

my_pred_value = my_pred_value*sig_ratings
my_pred_value = my_pred_value + mu_ratings
'''
print('my_pred_value shape', my_pred_value.shape)

writePredict(my_pred_value,outputFilename)
print('finished prediction')
