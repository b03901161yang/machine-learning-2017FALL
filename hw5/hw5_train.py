import math
import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Embedding, Flatten, Merge, Dropout, Dense, Dot, Input, Add
from keras.models import model_from_json
from keras.models import Sequential, Model
import sys


def get_CFModel(n_users, m_items, k_factors):
    input_users = Input(shape = [1])
    embed_users = Embedding(n_users, k_factors)(input_users)
    flat_users = Flatten()(embed_users)
    flat_users = Dropout(0.0)(flat_users)    

    input_items = Input(shape = [1])
    embed_items = Embedding(m_items, k_factors)(input_items)
    flat_items = Flatten()(embed_items)
    flat_items = Dropout(0.0)(flat_items)    

    dot_result = Dot(axes = 1, normalize=False)([flat_users,flat_items])
    #################add bias
    bias_users = Embedding(n_users, 1)(input_users)
    flat_bias_users = Flatten()(bias_users)
    bias_items = Embedding(m_items, 1)(input_items)
    flat_bias_items = Flatten()(bias_items)

    dot_result = Add()([dot_result, flat_bias_users, flat_bias_items])
    #################
    model = Model(inputs=[input_users, input_items], outputs=dot_result)

    return model

trainDataFilename = 'train.csv'
testDataFilename = sys.argv[1]
outputFilename = sys.argv[2]

moviefilename = sys.argv[3]
userfilename = sys.argv[4]

K_FACTORS = 128

trainData = pd.read_csv(trainDataFilename,
                      sep=',',
                      encoding='latin-1',
                      usecols=['TrainDataID', 'UserID', 'MovieID', 'Rating'])
max_userid = trainData['UserID'].drop_duplicates().max()
max_movieid = trainData['MovieID'].drop_duplicates().max()
print( len(trainData), 'train data loaded.')
print('max_userid',max_userid) #6040
print('max_movieid',max_movieid) #3952
#print(trainData)


Users = trainData['UserID'].values
#print('Users:', Users, ', shape =', Users.shape)
Movies = trainData['MovieID'].values
#print('Movies:', Movies, ', shape =', Movies.shape)
Ratings = trainData['Rating'].values
#print('Ratings:', Ratings, ', shape =', Ratings.shape)
########################ratings normalization
'''
mu_ratings = np.mean(Ratings)
sig_ratings = np.std(Ratings)
print('mu: ',mu_ratings)
print('sigma: ',sig_ratings)
Ratings = (Ratings - mu_ratings)/sig_ratings
print('Ratings:', Ratings, ', shape =', Ratings.shape)
'''
###################################
np.random.seed(0)
cv_idx_tmp = np.random.permutation(899873)


cv_fold_1 = cv_idx_tmp[0:800000]
cv_fold_2 = cv_idx_tmp[800000:899873]

x_train = [Users, Movies]
y_train = Ratings

x1_l = Users[cv_fold_1[:,],]
x1_r = Movies[cv_fold_1[:,],]
x_train1 = [x1_l, x1_r]

y_train1 = Ratings[cv_fold_1[:,],]

x2_l = Users[cv_fold_2[:,],]
x2_r = Movies[cv_fold_2[:,],]
x_train2 = [x2_l, x2_r]

y_train2 = Ratings[cv_fold_2[:,],]
###################################
#opt = keras.optimizers.Adam(lr = 0.0001, decay = 1e-6)

epochs = 15
batch_size = 512
model = get_CFModel(max_userid, max_movieid, K_FACTORS)

opt = keras.optimizers.Adamax(lr = 0.002, decay = 1e-6)
model.compile(loss='mse', optimizer=opt)


#model.load_weights(str(K_FACTORS)+'cf.h5')

callbacks = CSVLogger(str(K_FACTORS) +'_training.log')
history = model.fit(x_train1,
                    y_train1,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_data= (x_train2, y_train2),
                    verbose=1,
                    callbacks=[callbacks])


################################
model_json = model.to_json()
with open( str(K_FACTORS)+'mf.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights( str(K_FACTORS) + 'mf.h5')
print('mf Model saved')
################################

