import math
import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Embedding, Flatten, Merge, Dropout, Dense, Dot, Input, Add, Activation, Concatenate
from keras.models import model_from_json
from keras.models import Sequential, Model
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
#print( len(trainData), 'train data loaded.')
print('max_userid',max_userid) #6040
print('max_movieid',max_movieid) #3952
#print(trainData)


Users = trainData['UserID'].values
#print('Users:', Users, ', shape =', Users.shape)
Movies = trainData['MovieID'].values
#print('Movies:', Movies, ', shape =', Movies.shape)
Ratings = trainData['Rating'].values
#print('Ratings:', Ratings, ', shape =', Ratings.shape)
########################read movie category




Movie_cat = pd.read_csv(moviefilename,
                      sep='::',
                      engine='python',
                      encoding='latin-1',
                      usecols=['movieID', 'Title', 'Genres'])

movie_id_list = Movie_cat['movieID']
movie_cat_list = Movie_cat['Genres']

#print(movie_id_list)
#print('movie_id_list',len(movie_id_list))

print(movie_cat_list)
print('movie_cat_list',len(movie_cat_list))

movie_category_dict = {}
movie_count_dict = {}
user_gender_dict = {}
user_age_dict = {}
user_occupation_dict = {}
cat_count = 0

for n in range(len(movie_id_list)):
    tmp_id = movie_id_list[n] # is n th movie id
    movie_cat_split = movie_cat_list[n].split('|')

    movie_category_dict[tmp_id] = movie_cat_split[0]  # only use the first one
    if not(movie_category_dict[tmp_id] in movie_count_dict.keys()):
        movie_count_dict[movie_category_dict[tmp_id]] = cat_count
        cat_count += 1

print('cat count',cat_count) # 7
movie_cat_f = np.zeros((max_movieid+1, cat_count))
for key, value in movie_category_dict.items():
    movie_cat_f[key,movie_count_dict[value]] = 1

print(movie_cat_f)
print(movie_cat_f.shape)
# print('movie category dict',movie_category_dict)
########################################################################
user_info = pd.read_csv(userfilename,
                      sep='::',
                      engine='python',
                      encoding='latin-1',
                      usecols=['UserID', 'Gender', 'Age','Occupation','Zip-code'])

user_id_list = user_info['UserID']
user_gender_list = user_info['Gender']
user_age_list = user_info['Age']
user_occupation_list = user_info['Occupation']

# gender_count == 2

age_count = 0
occupation_count = 0
age_count_dict = {}
occupation_count_dict = {}

for n in range(len(user_id_list)):
    tmp_user_id = user_id_list[n]  # is n th movie id
    user_gender_dict[tmp_user_id] = user_gender_list[n]
    user_age_dict[tmp_user_id] = user_age_list[n]
    user_occupation_dict[tmp_user_id] = user_occupation_list[n]
    if not(user_age_list[n] in age_count_dict.keys()):
        age_count_dict[user_age_list[n]] = age_count
        age_count += 1
    if not(user_occupation_list[n] in occupation_count_dict.keys()):
        occupation_count_dict[user_occupation_list[n]] = occupation_count
        occupation_count += 1
print('age count',age_count) # 7
print('occupation count',occupation_count) # 21
###################################


# movie_feature = np.zeros(())
gender_f  = np.zeros((max_userid+1,2))  # id start from 1
age_f = np.zeros((max_userid+1,age_count))
occupation_f = np.zeros((max_userid+1,occupation_count))

for key in user_gender_dict: # gender
    if user_gender_dict[key] == 'M':
        gender_f[key,0] = 1
    else:
        gender_f[key,1] = 1

for key, value in user_age_dict.items(): # key is user id, value is
    age_f[key, age_count_dict[value]] = 1

for key, value in user_occupation_dict.items():
    occupation_f[key,occupation_count_dict[value]] = 1

#print(gender_f.shape)
#print(age_f.shape)
#print(occupation_f.shape)

all_user_f = np.concatenate((gender_f, age_f, occupation_f),axis = 1)
'''
for n in range(10):
    print(np.count_nonzero(all_user_f[n]))
'''

print(Users.shape)
print(Movies.shape)

'''
for n in range(3):
    print(Users[n])
    print(Movies[n])
'''
x_f = np.zeros((Users.shape[0],48)) #18+30
for n in range(Users.shape[0]): #num of train data
    x_f[n,:30] = all_user_f[ Users[n],:]
    x_f[n,30:48] = movie_cat_f[ Movies[n],:]


np.savetxt('all_user_f.txt',all_user_f)
np.savetxt('movie_cat_f.txt',movie_cat_f)
###################################
np.random.seed(0)
cv_idx_tmp = np.random.permutation(899873)

x_train = [Users, Movies, x_f]

cv_fold_1 = cv_idx_tmp[0:800000]
cv_fold_2 = cv_idx_tmp[800000:899873]

y_train = Ratings

x_train1 = [Users[cv_fold_1[:,],], Movies[cv_fold_1[:,],], x_f[cv_fold_1[:,],:]]
x_train2 = [Users[cv_fold_2[:,],], Movies[cv_fold_2[:,],], x_f[cv_fold_2[:,],:]]

y_train1 = Ratings[cv_fold_1[:,],]
y_train2 = Ratings[cv_fold_2[:,],]
###################################

epochs = 15
batch_size = 512
'''
model = Sequential()
#model.add(Embedding(input_dim= 48,output_dim=10))
model.add(Dense(1024, input_dim= 50,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(1, activation='relu'))
'''
model = getModel(max_userid, max_movieid, K_FACTORS)

model.summary()

opt = keras.optimizers.Adamax(lr = 0.002, decay = 0)
#opt = keras.optimizers.Adam()
model.compile(loss='mse', optimizer=opt)


#model.load_weights(str(K_FACTORS)+'cf.h5')

callbacks = CSVLogger(str(K_FACTORS) +'_training.log')
history = model.fit(x_train1,
                    y_train1,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose=1,
                    validation_data=(x_train2,y_train2),
                    callbacks=[callbacks])

model_json = model.to_json()
with open( str(K_FACTORS)+'with_feature.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights( str(K_FACTORS) + 'with_feature.h5')

#model.save('plot_tsne.h5')
print('feature Model saved')