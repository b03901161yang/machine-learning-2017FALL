import sys
import numpy as np
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def readTrainDataLabel(dataFilename):
    dataFile = open(dataFilename,'r', errors='ignore')
    raw_data = []
    my_label = []
    counter_first = 0
    for row in csv.reader(dataFile,delimiter=','):
        islabel = 0#reset islabel
        if(counter_first == 0):#skip for the first row(just title)
            counter_first += 1
            continue
        for n in range(len(row)):#1 label + 48*48 = 2304
            if(islabel == 0):
                islabel = 1
                my_label.append(int(row[n]))
            else:
                tmp_list=row[n].split(' ')#tmp is list
                #print(len(tmp_list))
                tmp_2d_arr = np.reshape(np.array(tmp_list),(48,48,1))#channel last
                raw_data.append(tmp_2d_arr)
    my_label = np.array(my_label)
    raw_data = np.array(raw_data)
    return my_label, raw_data

def readTestData(dataFilename):
    dataFile = open(dataFilename,'r', errors='ignore')
    raw_data = []
    my_id = []
    counter_first = 0
    for row in csv.reader(dataFile,delimiter=','):
        is_id = 0#reset is_id
        if(counter_first == 0):#skip for the first row(just title)
            counter_first += 1
            continue
        for n in range(len(row)):#1 label + 48*48 = 2304
            if(is_id == 0):
                is_id = 1
                my_id.append(int(row[n]))
            else:
                tmp_list=row[n].split(' ')#tmp is list
                tmp_2d_arr = np.reshape(np.array(tmp_list),(48,48,1))
                raw_data.append(tmp_2d_arr)
    raw_data = np.array(raw_data)
    return my_id, raw_data

def writePredict(predictLabel,outputFilename):
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['id','label'])
    for n in range(predictLabel.shape[0]):
        tmp = int(predictLabel[n])
        result_csv.writerow( [str(n), str(tmp)] )
    return predictLabel

    
########################    
#trainDataFilename = 'train.csv'
testDataFilename = sys.argv[1]#'test.csv'
outputFilename = sys.argv[2]#'MYRESULT_BEST.csv'

#y_train, x_train = readTrainDataLabel(trainDataFilename)
test_id, x_test = readTestData(testDataFilename)

##################################################
num_classes = 7
epochs = 250
validation_part = 0.1
input_shape = (48, 48, 1)

print('K.image_data_format() =',K.image_data_format() ) #should use channel last

#x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
x_test /= 255

########################################################
########################################################
# convert class vectors to binary class matrices

#y_train = keras.utils.to_categorical(y_train, num_classes)

#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same'))
model.add(Dropout(0.1)) #dropout
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Dropout(0.15)) #dropout
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(Dropout(0.2)) #dropout
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(Dropout(0.2)) #dropout
model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.3)) #dropout
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4)) #dropout
model.add(Dense(num_classes, activation='softmax'))

#####pretrained model, to init
model.load_weights('bestMODEL.h5')

##skip compile and fit

predictions = model.predict(x_test)
#print('predictions shape :',predictions.shape)
predictions_label = np.argmax(predictions,axis = 1)
#print('predictions label shape :',predictions_label.shape)

writePredict(predictions_label,outputFilename)
