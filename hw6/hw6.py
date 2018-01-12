import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Embedding, Flatten, Merge, Dropout, Dense, Dot, Input, Add
from keras.models import model_from_json, load_model
from keras.models import Sequential, Model
import scipy
import csv
import time
from sklearn.cluster import KMeans
from keras.callbacks import CSVLogger
import sys



testDataFilename = sys.argv[2]
outputFilename = sys.argv[3]

##############
testData = pd.read_csv(testDataFilename,
                      sep=',',
                      usecols=['ID', 'image1_index', 'image2_index'])

print( len(testData), 'test data loaded.')

ID = testData['ID'].values
im1_index = testData['image1_index'].values
im2_index = testData['image2_index'].values

##################


def writePredict(predictValue,outputFilename):
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['ID','Ans'])
    for n in range(predictValue.shape[0]):
        tmp = int(predictValue[n,])
        result_csv.writerow( [str(n), str(tmp)] )
    return predictValue

image_npy_name = sys.argv[1]
x_train = np.load(image_npy_name) #image set

x_train = x_train/255.0

num_of_example = 10
example_input = x_train[0:num_of_example,:]


print('x train shape',x_train.shape) # (140000, 784), 784 = 28*28
#print( 'first image value',example_input[0].reshape(28, 28))

'''
plt.figure(figsize=(20, 4))
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
'''


#np.random.seed() #change here to see different validation

cv_idx_tmp = np.random.permutation(140000)


cv_fold_1 = cv_idx_tmp[0:130000]
cv_fold_2 = cv_idx_tmp[130000:140000]

x_train1 = x_train[cv_fold_1[:,],:]
x_train2 = x_train[cv_fold_2[:,],:]

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(64, activation='relu',name='encode_out')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adamax', loss='mse')

autoencoder.summary()

autoencoder = load_model('32_deep_autoencoder.h5')



'''
csv_logger = CSVLogger('32_autoencoder_training.log')
autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                callbacks=[csv_logger],
                validation_data=(x_train2, x_train2))

autoencoder.save('32_deep_autoencoder.h5')

'''

#decoded_imgs = autoencoder.predict(example_input)

example_test = x_train[0:num_of_example,:]
decoded_imgs = autoencoder.predict(example_test)




layer_name = 'encode_out'
encoder = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)

encoder_output = encoder.predict(x_train, batch_size=512)

print('encoder output shape',encoder_output.shape)


my_cluster = KMeans(n_clusters=2, random_state = 10).fit(encoder_output)
print(my_cluster)

print(my_cluster.cluster_centers_.shape)
print(my_cluster.labels_)
print(my_cluster.labels_.shape)
print(np.sum(my_cluster.labels_))

predictValue = np.zeros((len(testData),))

for i in range(len(testData)):
    if i %100000 == 0:
        print('progressing:',i)
    if my_cluster.labels_[im1_index[i],] == my_cluster.labels_[im2_index[i],]:
        predictValue[i,] = 1

print('predict value shape',predictValue.shape)
writePredict(predictValue, outputFilename)
