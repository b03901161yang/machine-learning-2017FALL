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

def mySigmoid(z):
    sigmoid_out = 1/(1+np.exp(-z))
    return sigmoid_out

def getLoss(trainData, trainLabel, w_vec):#define loss as cross entropy
    num_of_data = len(trainData)
    dim_of_data = len(trainData[0])#dimension of data, not 2d size, should be number of features(1st order)
    b = w_vec[0]
    w1 = w_vec[1:1+dim_of_data]
    w2 = w_vec[1+dim_of_data:1+2*dim_of_data]
    w1_2d = np.reshape(w1, (dim_of_data,1))
    w2_2d = np.reshape(w2, (dim_of_data,1))
    predictValue = mySigmoid( b + np.dot( np.transpose(w1_2d) ,np.transpose(trainData)) + np.dot( np.transpose(w2_2d), np.transpose((trainData)**2)) )
    #print('predict val shape',predictValue.shape)
    cross_entro = 0
    for n in range(num_of_data):
        if(trainLabel[n] == 1):
            if(predictValue [0,n]== 0):
                cross_entro -= -100000
            else:
                cross_entro -= 1*np.log(predictValue[0,n])
        else:
            if(predictValue[0,n] == 1):
                cross_entro -= -100000 #like inf
            else:
                cross_entro -= 1*np.log(1-predictValue[0,n])
    cross_entro = cross_entro/ num_of_data
    return cross_entro
    
def trainModelLinear(trainData, trainLabel, validationData, validationLabel):#use adagrad here, 2d
    #y = w2x^2 +w1x + b
    #initialize w b as 0
    num_of_data = len(trainData)
    dim_of_data = len( trainData[0])
    
    w2 = np.zeros( (dim_of_data,) )#w for weighting
    w1 = np.zeros( (dim_of_data,) )
    b = 0 #b for bias
    w_vec = np.zeros( (2*dim_of_data+1,))
    init_lr =  0.01#learning rate, initial
    
    sum_g_w2 = 0
    sum_g_w1 = 0
    sum_g_b = 0
    for t in range(20000):#500 iteration for easy demo
        if(t%100 == 0):
            print('t = ',t)
            w_vec[0] = b
            w_vec[1:1+dim_of_data] = w1
            w_vec[1+dim_of_data:1+2*dim_of_data] = w2
            tmp_loss = getLoss(trainData, trainLabel,w_vec)
            print('current Loss = ', tmp_loss)
            my_acc = checkAccu(validationData, validationLabel, w_vec)
            print('current validation accuracy = ',my_acc)
        g_w2_t = 0
        g_w1_t = 0 #gradient of w at t-th iteration
        g_b_t = 0
        w2_2d = np.reshape(w2, (dim_of_data,1))
        w1_2d = np.reshape(w1, (dim_of_data,1))
        del_y_arr = np.transpose(trainLabel) - mySigmoid( b + np.dot( np.transpose(w1_2d) ,np.transpose(trainData)) + np.dot( np.transpose(w2_2d), np.transpose((trainData)**2) )  )
        #print('del_y_arr', del_y_arr.shape)
        
        g_w2_t = (-2)*sum( np.dot(del_y_arr,(trainData)**2))
        g_w1_t = (-2)*sum( np.dot(del_y_arr, trainData) )
        g_b_t = (-2)*sum(np.transpose(del_y_arr) )
        
        sum_g_w2 = sum_g_w2 + (g_w2_t)**2
        sum_g_w1 = sum_g_w1 + (g_w1_t)**2
        sum_g_b = sum_g_b + (g_b_t)**2
        
        sqrt_sum_g_w2 = np.sqrt(sum_g_w2)
        sqrt_sum_g_w1 = np.sqrt(sum_g_w1)
        sqrt_sum_g_b = np.sqrt(sum_g_b)
        
        #update w and b
        w2 = w2 - (init_lr*g_w2_t)/sqrt_sum_g_w2
        w1 = w1 - (init_lr*g_w1_t)/sqrt_sum_g_w1
        b = b - (init_lr*g_b_t)/sqrt_sum_g_b
        #print('shape sum_g_w', sum_g_w.shape)
    w_vec[0] = b
    w_vec[1:1+dim_of_data] = w1
    w_vec[1+dim_of_data:1+2*dim_of_data] = w2     
    total_loss = getLoss(trainData, trainLabel, w_vec)
    print('Final Loss = ',total_loss)
    return w_vec

def writePredict(testData,w_vec,outputFilename):
    dim_of_data = len(testData[0])
    num_of_data = len(testData)
    b = w_vec[0]
    w1 = w_vec[1:1+dim_of_data]
    w2 = w_vec[1+dim_of_data:1+2*dim_of_data]
    predictLabel = np.zeros((num_of_data,))
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['id','label'])
    for n in range(num_of_data):
        predictVal = mySigmoid( b + np.dot(w1 , testData[n]) + np.dot(w2,testData[n]**2) )
        if(predictVal < 0.5):
    	    predictLabel[n] = 0
        else:
            predictLabel[n] = 1
        tmp =  int(predictLabel[n])
        result_csv.writerow( [str(n+1), str(tmp)] )
    return predictLabel

def checkAccu(testData,testLabel,w_vec):#2d here
    dim_of_data = len(testData[0])
    num_of_data = len(testData)
    b = w_vec[0]
    w1 = w_vec[1:1+dim_of_data]
    w2 = w_vec[1+dim_of_data:1+2*dim_of_data]
    predictLabel = np.zeros((num_of_data,))
    correct_count = 0
    for n in range(num_of_data):
        predictVal = mySigmoid( b + np.dot(w1 , testData[n]) + np.dot(w2, testData[n]**2) )
        if(predictVal < 0.5):
    	    predictLabel[n] = 0
        else:
            predictLabel[n] = 1
    for n in range(num_of_data):
        if(predictLabel[n] == testLabel[n]):
            correct_count +=1
    my_acc = correct_count / num_of_data
    return my_acc
    
########################    

rawdatafilename = sys.argv[1] #no use
rawtestfilename = sys.argv[2] #no use

trainDataFilename = sys.argv[3]#'X_train'
trainLabelFilename = sys.argv[4]#'Y_train'
testDataFilename = sys.argv[5]#'X_test'
outputFilename = sys.argv[6]#'second_order_logistic_result.csv'


chosen_features = []
for n in range(106):
    chosen_features.append(n)
    #if(64 <= n and n < 106):
    #    chosen_features.append(n)

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

x_train1 = x_train[cv_fold_1[:,] ]
y_train1 = y_train[cv_fold_1[:,] ]

x_train2 = x_train[cv_fold_2[:,] ]
y_train2 = y_train[cv_fold_2[:,] ]

y_train_raw1 = raw_label[cv_fold_1[:,] ]
y_train_raw2 = raw_label[cv_fold_2[:,] ]

y_train_raw1 = np.reshape(y_train_raw1,(-1,))
y_train_raw2 = np.reshape(y_train_raw2,(-1,))

n_features = x_train.shape[1]
n_labels = y_train.shape[1]

#w_vec = trainModelLinear(x_train2, y_train_raw2, x_train1, y_train_raw1)
#w_vec = trainModelLinear(x_train, raw_label, x_train1, y_train_raw1)
#load model instead of train model here

w_vec = np.loadtxt('logistic_model.txt')
writePredict(x_test ,w_vec ,outputFilename)

#my_loss = getLoss(x_train2, y_train_raw2, w_vec)
#print('my final loss = ',my_loss)

my_acc = checkAccu(x_train1, y_train_raw1, w_vec)
print('my accuracy = ',my_acc)
