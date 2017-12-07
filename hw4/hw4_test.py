import numpy as np
import pandas as pd
import csv
import sys
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
from keras.models import model_from_json
from keras.models import Model


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
epochs = 80
########################################################
#trainDataFilename = 'training_label.txt'

testDataFilename = sys.argv[1] #'testing_data.txt'
outputFilename = sys.argv[2] #'result_ensemble.csv'


#y_train, x_train = readTrainDataLabel(trainDataFilename)
x_test = readTestData(testDataFilename)
#y_train = keras.utils.to_categorical(y_train, num_classes)

################################################################################
model_embed = Word2Vec.load('model_word2vec.bin')

x_test_vec = []

for n in range(x_test.shape[0]):
    tmp_sentence = text_to_word_sequence(x_test[n])
    x_test_vec.append(tmp_sentence)


max_article_length = 40
print('max article length:',max_article_length)

#############################################################
np.random.seed(0)

x_test_vec = []

for n in range(x_test.shape[0]):
    tmp_sentence = text_to_word_sequence(x_test[n])
    x_test_vec.append(tmp_sentence)

x_test_encode = np.zeros(shape = (len(x_test_vec),max_article_length, 50))

for n in range(len(x_test_vec)):
    x_test_encode[n][-len(x_test_vec[n]):][:] = model_embed[x_test_vec[n]]


x_test_encode = np.array(x_test_encode).astype('float32')
####################################


##############################################################
model_file_names = ['model_8179','model_816','model_cnn','model_cnn2','model_gru','model_gru2','model_gru0']
num_models = len(model_file_names)
models = []
predicts_test = []
for n in range(num_models):
    print('open model :',n)
    print('model name:',model_file_names[n])
    json_file = open(model_file_names[n]+'.json', 'r',errors='ignore')
    model_json = json_file.read()
    json_file.close()
    sub_model = model_from_json(model_json)
    sub_model.load_weights(model_file_names[n]+'.h5')
    sub_model.summary()
    tmp_test = sub_model.predict(x_test_encode,batch_size=2048)
    predicts_test.append(tmp_test[:,0])
    models.append(sub_model)
##############################################################
predicts_test = np.array(predicts_test)

x_test_encode_new = np.zeros((x_test_encode.shape[0],num_models))
#for n in range(x_train1.shape[0]):
for m in range(num_models):
    x_test_encode_new[:,m] = predicts_test[m,:]

opt = keras.optimizers.Adam()
model = Sequential()
model.add(Dense(512, input_dim = num_models, activation = 'relu'))
model.add(Dense(2, activation ='softmax'))
model.summary()

model.load_weights('model_ensemble.h5')

print('ensemble model loaded')

my_pred_value = model.predict(x_test_encode_new,batch_size = 2048)
my_pred_label = np.argmax(my_pred_value,axis = 1)

writePredict(my_pred_label,outputFilename)
print('finished prediction')

