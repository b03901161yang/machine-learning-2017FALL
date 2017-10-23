from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import numpy as np
import csv

def readTrainData(dataFilename, chosen_features):
    dataFile = open(dataFilename,'r', errors='ignore')
    raw_data = []
    counter = 0
    for row in csv.reader(dataFile,delimiter=','):
        if(counter == 0):#skip for the first row(just title)
            counter+=1
            continue
        mylist = []#reset mylist here
        for n in range(len(row)):#106 features
            if(n %106 in chosen_features):#only use chosen features
                if(row[n] == '0'):
                    mylist.append(float(0.00001))#to prevent gradient descent fail?
                else:
                    mylist.append(float(row[n]))
        raw_data.append(mylist)
    raw_data = np.array(raw_data)
    return raw_data


def normalizeData(data1,data2):#can be training data or testing data, feature scaling
    num_of_data = len(data1)+len(data2)
    dim_of_data = len(data1[0])#dimension of data, include 2d size
    #print(num_of_data)
    #print(dim_of_data)
    mean_arr  = np.zeros( (dim_of_data,1) )
    sigma_arr = np.zeros( (dim_of_data,1) )
    for n in range(dim_of_data):
        for m in range(len(data1)):
            mean_arr[n] += data1[m,n]
        for m2 in range(len(data2)):
            mean_arr[n] += data2[m2,n]
    mean_arr = mean_arr/num_of_data
    for n in range(dim_of_data):
        for m in range(len(data1)):
            sigma_arr[n] += (data1[m,n]-mean_arr[n])**2
        for m2 in range(len(data2)):
            sigma_arr[n] += (data2[m2,n]-mean_arr[n])**2
    
    sigma_arr = np.sqrt(sigma_arr/num_of_data)
    for n in range(dim_of_data):
        for m in range(len(data1)):
            data1[m,n] = (data1[m,n]-mean_arr[n])/sigma_arr[n]
        for m2 in range(len(data2)):
            data2[m2,n] =(data2[m2,n]-mean_arr[n])/sigma_arr[n]
    return data1, data2

def readTrainLabel(labelFilename):
    labelFile = open(labelFilename,'r', errors='ignore')
    raw_label = []#construct raw_data as 2d array
    counter = 0
    for row in csv.reader(labelFile,delimiter=','):
        if(counter == 0):#skip for the first row(just title)
            counter+=1
            continue
        mylist = []#reset mylist here
        for n in range(len(row)):#106 features
            mylist.append(float(row[n]))
        raw_label.append(mylist)
        #print(mylist)
    raw_label = np.array(raw_label)
    num_of_data = raw_label.shape[0]
    new_label = np.zeros((num_of_data,2))#binary classification here
    for m in range(num_of_data):
        if(raw_label[m] == 1):
            new_label[m,1] = 1
        else:
            new_label[m,0] = 1
        
    return new_label, raw_label

def writePredict(predictLabel,outputFilename):
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['id','label'])
    for n in range(predictLabel.shape[0]):
        tmp =  int(predictLabel[n])
        result_csv.writerow( [str(n+1), str(tmp)] )
    return predictLabel

    
########################    
rawdatafilename = sys.argv[1] #no use
rawtestfilename = sys.argv[2] #no use

trainDataFilename = sys.argv[3]#'X_train'
trainLabelFilename = sys.argv[4]#'Y_train'
testDataFilename = sys.argv[5]#'X_test'
outputFilename = sys.argv[6]#result.csv

chosen_features = []
for n in range(106):
    chosen_features.append(n)

x_train = readTrainData(trainDataFilename, chosen_features)
x_test = readTrainData(testDataFilename, chosen_features)
x_train, x_test = normalizeData(x_train, x_test)
y_train, raw_label = readTrainLabel(trainLabelFilename)

np.random.seed(0)
cv_idx_tmp = np.random.permutation(32561)
#for 4 fold-cross validation, 1400 1400 1400 1440
#cv_fold_1 = cv_idx_tmp[0:16000]
#cv_fold_2 = cv_idx_tmp[16000:32561]

cv_fold_1 = cv_idx_tmp[0:16000]
cv_fold_2 = cv_idx_tmp[16000:32561]
#print(cv_fold_1[:,].shape)

x_train1 = x_train[cv_fold_1[:,] ]
y_train1 = y_train[cv_fold_1[:,] ]

x_train2 = x_train[cv_fold_2[:,] ]
y_train2 = y_train[cv_fold_2[:,] ]

y_train_raw1 = raw_label[cv_fold_1[:,] ]
y_train_raw2 = raw_label[cv_fold_2[:,] ]

y_train_raw1 = np.reshape(y_train_raw1,(-1,))
y_train_raw2 = np.reshape(y_train_raw2,(-1,))


print('x train1 shape ',x_train1.shape)
print('y train1 raw shape ',y_train_raw1.shape)

print('x train shape[1] ',x_train.shape[1])#106 here
print('y train shape[1] ',y_train.shape[1])#2

n_features = x_train.shape[1]
n_labels = y_train.shape[1]

#writePredict(x_test ,w_vec ,outputFilename)
##################################################
# fit model no training data
model = XGBClassifier(
 learning_rate =0.2,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

#model.fit(x_train2, y_train_raw2.ravel())
model.fit(x_train, raw_label.ravel())

# make predictions for test data
#y_pred = model.predict(x_train1)
y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions

#accuracy = accuracy_score(y_train_raw1, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

predictions = np.array(predictions)
#print("predictions shape",predictions.shape)

writePredict(predictions,outputFilename)
