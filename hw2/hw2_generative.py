import sys
import numpy as np
import csv
from numpy.linalg import inv

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
    dim_of_data = len(data1[0])
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
    raw_label = []#construct raw_data as 2d array for raw data, len(features)*20days*12months , 24hr
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



    
def mySigmoid(z):
    sigmoid_out = 1/(1+np.exp(-z))
    return sigmoid_out

def checkAccu(predictLabel,testLabel):
    num_of_data = len(testLabel)
    #print('len test label',num_of_data)
    correct_count = 0
    for n in range(num_of_data):
        if(predictLabel[n] == testLabel[n]):
            correct_count += 1
    my_acc = correct_count / num_of_data
    return my_acc

def divTrainClass(trainData,trainLabel):
    num_of_label = len(trainLabel)
    class0 = []
    class1 = []
    for n in range (num_of_label):
        if(trainLabel[n] == 0):
            class0.append( trainData[n] )
        else:
            class1.append( trainData[n] )
    class0 = np.array(class0)
    class1 = np.array(class1)
    return class0, class1

def getClassProb(trainLabel):
    num_of_label = len(trainLabel)
    C0_count = 0
    for n in range (num_of_label):
        if(trainLabel[n] == 0):
            C0_count +=1
    p0 = C0_count / num_of_label
    p1 = 1-p0
    return p0, p1
    
def calMeanCov(classData): #calculate mean and covariance
    num_of_data = len(classData)
    dim_of_data = len(classData[0])
    class_mean = np.mean(classData,axis = 0)
    class_cov = np.zeros((dim_of_data,dim_of_data))    
    mean_col = np.transpose(class_mean)
    mean_col= np.reshape(mean_col, (dim_of_data,1) )
    
    for n in range(num_of_data):
         data_col = np.transpose(classData[n,:])       
         #print('data_col shape',data_col.shape)
         data_col = np.reshape(data_col, (dim_of_data,1) )
         class_cov += np.dot( (data_col-mean_col) , np.transpose(data_col-mean_col) )

    class_cov = class_cov / num_of_data
    return class_mean, class_cov
    
def getPredictLabel(testData,P0,P1,mean0,mean1, share_cov):
    num_of_data = len(testData)
    dim_of_data = len(testData[0])
    predictLabel = np.zeros((num_of_data,))
    mean0_col = np.transpose(mean0)
    mean0_col= np.reshape(mean0_col, (dim_of_data,1) )
    mean1_col = np.transpose(mean1)
    mean1_col= np.reshape(mean1_col, (dim_of_data,1) )
    #inv_share_cov = inv(share_cov)
    #print('det value = ',np.linalg.det(share_cov)) #almost 0!!!
    inv_share_cov = np.linalg.pinv(share_cov) #use pinv instead
    
    w = np.transpose( np.dot( np.transpose(mean0_col-mean1_col), inv_share_cov  ) )
    w = np.reshape(w,(dim_of_data,))
    num0 = np.dot( np.dot( np.transpose(mean0_col), inv_share_cov), mean0_col)
    num1 = np.dot( np.dot( np.transpose(mean1_col), inv_share_cov), mean1_col)
    b = np.log(P0/P1) - num0/2 + num1/2
    for n in range(num_of_data):
        z = b + np.dot( w , testData[n,:])
        PC0_givenX = mySigmoid(z)
        #print('PC0_givenX ',PC0_givenX)
        if(PC0_givenX > 0.5):
            predictLabel[n] = 0
        else:
            predictLabel[n] = 1
    return predictLabel

def writePredict(testLabel,outputFilename):
    num_of_data = len(testLabel)
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['id','label'])
    for n in range(num_of_data):
        tmp =  int(testLabel[n])
        result_csv.writerow( [str(n+1), str(tmp)] )
    return testLabel
########################    
rawdatafilename = sys.argv[1] #no use
rawtestfilename = sys.argv[2] #no use

trainDataFilename = sys.argv[3]#'X_train'
trainLabelFilename = sys.argv[4]#'Y_train'
testDataFilename = sys.argv[5]#'X_test'
outputFilename = sys.argv[6]#'result.csv'

chosen_features = []
for n in range(106):
    chosen_features.append(n)
    #if(64 <= n and n < 106):
    #    chosen_features.append(n)

x_train = readTrainData(trainDataFilename, chosen_features)
x_test = readTrainData(testDataFilename, chosen_features)
#x_train, x_test = normalizeData(x_train, x_test) #not sure if generative needs normalization

y_train, raw_label = readTrainLabel(trainLabelFilename)

np.random.seed(0)
cv_idx_tmp = np.random.permutation(32561)
#for 4 fold-cross validation, 1400 1400 1400 1440

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

trainC0, trainC1 = divTrainClass(x_train, raw_label)


print('x train1 shape ',x_train1.shape)
print('y train1 shape ',y_train1.shape)

print('x train shape[1] ',x_train.shape[1])#106 here
print('y train shape[1] ',y_train.shape[1])#2

#P0, P1 = getClassProb(y_train_raw1)
P0, P1 = getClassProb(raw_label)
class0, class1 = divTrainClass(x_train, raw_label)
#class0, class1 = divTrainClass(x_train1, y_train_raw1)

class0_mean, class0_cov = calMeanCov(class0)
class1_mean, class1_cov = calMeanCov(class1)

share_cov = P0*class0_cov + P1*class1_cov

predict_label = getPredictLabel(x_test,P0,P1,class0_mean,class1_mean,share_cov)
#predict_label = getPredictLabel(x_train2,P0,P1,class0_mean,class1_mean,share_cov)

writePredict(predict_label,outputFilename)
