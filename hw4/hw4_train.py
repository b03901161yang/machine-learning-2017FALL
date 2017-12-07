import numpy as np
import sys
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU,  Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import keras
from gensim.models import Word2Vec
from keras.callbacks import CSVLogger

def readTrainDataLabel(dataFilename):
    dataFile = open(dataFilename,'r', errors='ignore')
    raw_data = []
    my_label = []
    for row in csv.reader(dataFile,delimiter='\n'):
        islabel = 0#reset islabel
        ismark = 0
        #print(row)
        #print(len(row))
        tmp_list=row[0].split('+++$+++')
        #print(tmp_list)
        mylist = []
        my_label.append(int(tmp_list[0]))
        raw_data.append(tmp_list[1])
    my_label = np.array(my_label)
    raw_data = np.array(raw_data)
    return my_label, raw_data

def readTestData(dataFilename):
    dataFile = open(dataFilename,'r', errors='ignore')
    raw_data = []
    my_split = dataFile.read().split('\n')
    for row in my_split[:-1]:
        raw_data.append(row)
    raw_data.pop(0)
    raw_data = np.array(raw_data)
    return raw_data



def writePredict(predictLabel,outputFilename):
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['id','label'])
    for n in range(predictLabel.shape[0]):
        tmp = int(predictLabel[n])
        result_csv.writerow( [str(n), str(tmp)] )
    return predictLabel
############################################################
batch_size = 1024
num_classes = 2
epochs = 3
########################################################
trainDataFilename = sys.argv[1] #'training_label.txt'

nolabelDataFilename = sys.argv[2] #'testing_data.txt'

y_train, x_train = readTrainDataLabel(trainDataFilename)
y_train = keras.utils.to_categorical(y_train, num_classes)

################################################################################
model_embed = Word2Vec.load('model_word2vec.bin')

x_train_vec = []
print('x_train.shape[0] ',x_train.shape[0])

for n in range(x_train.shape[0]):
    tmp_sentence = text_to_word_sequence(x_train[n])
    x_train_vec.append(tmp_sentence)

max_article_length = 40
print('max article length:',max_article_length)

x_train_encode = np.zeros(shape = (len(x_train_vec),max_article_length, 50))

for n in range(len(x_train_vec)):
    x_train_encode[n][-len(x_train_vec[n]):][:] = model_embed[x_train_vec[n]]

x_train_encode = np.array(x_train_encode).astype('float32')

print('x_train_encode shape',x_train_encode.shape)
#############################################################
np.random.seed(0)
cv_idx_tmp = np.random.permutation(200000)


cv_fold_1 = cv_idx_tmp[0:190000]
cv_fold_2 = cv_idx_tmp[190000:200000]

x_train1 = x_train_encode[cv_fold_1[:,],:]
y_train1 = y_train[cv_fold_1[:,],:]

x_train2 = x_train_encode[cv_fold_2[:,],:]
y_train2 = y_train[cv_fold_2[:,],:]

##############################################################
opt = keras.optimizers.Adam()
model = Sequential()
model.add(GRU(128, input_shape = (max_article_length, 50), dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences = True))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(2, activation ='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy , optimizer=opt, metrics = ['accuracy'])

csv_logger = CSVLogger('gru_training.log')
model.fit(x_train1,
          y_train1,
          validation_data = (x_train2, y_train2),
          batch_size=batch_size,
          callbacks=[csv_logger],
          epochs=epochs)

model_json = model.to_json()
with open('model_gru.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model_gru.h5')
print('Model saved')



