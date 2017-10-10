import sys
import numpy as np
import csv
from numpy.linalg import inv

def readTrainData(trainFilename,chosen_features,target_idx):
    originalFile = open(trainFilename,'r',errors='ignore')
    raw_data = []#construct raw_data as 2d array for raw data, len(features)*20days*12months , 24hr
    counter = 0
    #print('chosen_features = ',chosen_features)
    #assert target is in chosen_features
    target_idx_in_chosen = 0
    for n in chosen_features:
        if(n == target_idx):
            break
        target_idx_in_chosen += 1
    for row in csv.reader(originalFile,delimiter=','):
        if(counter == 0):#skip for the first row
            counter+=1
            continue
        del row[0:3]#delete the first three elements, which are not features
        mylist = []#reset mylist here
        if(counter%18 in chosen_features):#total 18 features
            for n in range(len(row)):#24hr
                if row[n] == 'NR':
                    mylist.append(float(0))
                else:
                    mylist.append(float(row[n]))
            raw_data.append(mylist)
            #print(mylist)
        counter+=1
    data_tmp = np.zeros( (12*len(chosen_features), 24*20))
    data_9hr = np.zeros( (len(chosen_features)*9,470*12) )#construct 9 hr consecutive data,20 days per month for training
    for n in range(len(raw_data)):#12 months*20 days* len(chosen_features)
        for m in range(len(raw_data[0])):#24 hr
            which_feature = n%len(chosen_features)#belongs to which feature
            which_month = int( np.floor(n/(len(chosen_features)*20)) )#belongs to which month
            which_day = int( np.floor( (n%(len(chosen_features)*20)) / len(chosen_features)) )
            data_tmp[which_month*len(chosen_features)+which_feature][which_day*24+m] = raw_data[n][m]
    #construct data_9hr
    #construct data_label
    
    data_label = np.zeros( (470*12 , 1) )
    for n in range(0,len(data_tmp),len(chosen_features)):#0,3,6,9...
        for m in range(470):#480-10 = 470 permonth, 10 for last label
            which_month = int( np.floor(n/len(chosen_features)) )
            #print('month =',which_month)
            per_data = []
            for k in range(len(chosen_features)):
                tmp = np.transpose( data_tmp[n+k,m:m+9] )#0-8, 9 consecutive
                per_data.extend(tmp)
            #print('per_data',per_data)
            data_9hr[ :, which_month*470 + m] = per_data
            data_label[which_month*470 + m] = data_tmp[n + target_idx_in_chosen,m+9]#10th target as label
    data_9hr = np.transpose(data_9hr)
    
    return data_9hr, data_label

def normalizeData(data):#can be training data or testing data, feature scaling
    num_of_data = len(data)
    dim_of_data = len(data[0])
    #print(num_of_data)
    #print(dim_of_data)
    mean_arr  = np.zeros( (dim_of_data,1) )
    sigma_arr = np.zeros( (dim_of_data,1) )
    for n in range(dim_of_data):
        mean_arr[n] = np.mean(data[:,n])
        sigma_arr[n] = np.std(data[:,n])
    for n in range(dim_of_data):
        for m in range(num_of_data):
            data[m][n] = (data[m][n]-mean_arr[n])/sigma_arr[n]
    return data

def readTestData(testFilename,chosen_features):
    originalFile = open(testFilename,'r',errors='ignore')
    raw_data = []#construct raw_data
    counter = 1#do not skip first row
    print('chosen_features = ',chosen_features)
    for row in csv.reader(originalFile,delimiter=','):
        del row[0:2]#delete the first two elements, which are not features
        #if(counter < 3):
        #    print('original row = ',row)
        mylist = []#reset mylist here
        if( counter%18 in chosen_features):#total 18 features
            for n in range(len(row)):#9 consecutive hr
                if row[n] == 'NR':
                    mylist.append(float(0))
                else:
                    mylist.append(float(row[n]))
            raw_data.append(mylist)
        counter+=1
    raw_data = np.array(raw_data)
    testdata = np.zeros( (240,len(chosen_features)*9) )#240 testing data
    
    for n in range(0,len(raw_data),len(chosen_features)):#0,3,6,9...
        #print('n = ',n)
        which_data = int( np.floor(n/len(chosen_features)) )
        #print('data =',which_data)
        per_data = []
        for k in range(len(chosen_features)):
            tmp = raw_data[n+k,:]
            #print('tmp = ',tmp)
            per_data.extend(tmp)
            #print('per_data',per_data)
        testdata[which_data,:] = per_data 
    return testdata

def trainModelLinear_1d(trainData, trainLabel):#use adagrad here, has bug now
    #y = wx + b
    #initialize w b as 0
    num_of_data = len(trainData)
    dim_of_data = len(trainData[0])
    w = np.zeros( (dim_of_data,) )#w for weighting
    b = 0 #b for bias
    w_vec = np.zeros( (dim_of_data+1,))
    init_lr =  1#learning rate, initial
    #grad w
    #grad b
    sum_g_w = 0
    sum_g_b = 0
    for t in range(50000):#1000 iteration
        if(t%100 == 0):
            print('t = ',t)
            w_vec[0] = b
            w_vec[1:1+dim_of_data] = w
            tmp_loss = getLoss_higher_d(trainData, trainLabel, w_vec,1)
            print('current Loss = ', tmp_loss)
        g_w_t = 0 #gradient of w at t-th iteration
        g_b_t = 0
        for n in range(num_of_data):
            del_y = trainLabel[n]-( b+np.dot(w,trainData[n]) )
            g_w_t = g_w_t + (-2)*del_y*trainData[n]
            g_b_t = g_b_t + (-2)*del_y
        sum_g_w = sum_g_w + (g_w_t)**2
        sum_g_b = sum_g_b + (g_b_t)**2
        
        sqrt_sum_g_w = np.sqrt(sum_g_w)
        sqrt_sum_g_b = np.sqrt(sum_g_b)
        
        #update w and b
        w = w - (init_lr*g_w_t)/sqrt_sum_g_w
        #print('w = ',w)
        b = b - (init_lr*g_b_t)/sqrt_sum_g_b
        #print('shape sum_g_w', sum_g_w.shape)
    w_vec[0] = b
    w_vec[1:1+dim_of_data] = w
    total_loss = getLoss_higher_d(trainData, trainLabel, w_vec,1)
    print('Final Loss = ',total_loss)
    return w_vec

def trainModelLinear_2d(trainData, trainLabel):#use adagrad here, has bug now
    #y = wx + b
    #initialize w b as 0
    num_of_data = len(trainData)
    dim_of_data = len(trainData[0])
    w2 = np.zeros( (dim_of_data,) )#w for weighting
    w1 = np.zeros( (dim_of_data,) )
    b = 0 #b for bias
    w_vec = np.zeros( (2*dim_of_data+1,))
    init_lr =  1#learning rate, initial
    sum_g_w2 = 0
    sum_g_w1 = 0
    sum_g_b = 0
    for t in range(300):#300 iteration for easy demo
        if(t%100 == 0):
            print('t = ',t)
            w_vec[0] = b
            w_vec[1:1+dim_of_data] = w1
            w_vec[1+dim_of_data:1+2*dim_of_data] = w2
            tmp_loss = getLoss_higher_d(trainData, trainLabel,w_vec,2)
            print('current Loss = ', tmp_loss)
        g_w2_t = 0
        g_w1_t = 0 #gradient of w at t-th iteration
        g_b_t = 0
        for n in range(num_of_data):
            tmp_x = trainData[n]
            tmp_x2 = (tmp_x)**2
            del_y = trainLabel[n]-( b+np.dot(w1,tmp_x) + np.dot(w2,tmp_x2) )
            
            g_w2_t = g_w2_t + (-2)*del_y*tmp_x2
            g_w1_t = g_w1_t + (-2)*del_y*tmp_x
            g_b_t = g_b_t + (-2)*del_y
        
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
    total_loss = getLoss_higher_d(trainData, trainLabel, w_vec,2)

    #print('Final Loss = ',total_loss)
    return w_vec

def getLoss_higher_d(data, label, w_vec,d):
    total_loss = 0
    num_of_data = len(data)
    dim_of_data = len(data[0])
    predictValue = np.zeros( (num_of_data,) )
    for n in range(num_of_data):
        predictValue[n] = predictValue[n] + w_vec[0]#w0
        for m in range(d):
            tmp_w = w_vec[1+dim_of_data*m : 1+dim_of_data*(m+1) ]
            tmp_w = np.reshape(tmp_w,(dim_of_data,) )
            predictValue[n] = predictValue[n] + np.dot( tmp_w ,(data[n])**(m+1))
    for n in range(num_of_data):
        total_loss = total_loss + float( ( label[n]-predictValue[n] )**2)
    total_loss = total_loss / num_of_data
    total_loss = np.sqrt(total_loss)
    return total_loss
        
def writePredict_higher_d(testData,w_vec,d,outputFilename):
    dim_of_data = len(testData[0])
    num_of_data = len(testData)
    predictValue = np.zeros( (num_of_data,) )
    out_file = open(outputFilename,'w',newline='')
    result_csv = csv.writer(out_file)
    result_csv.writerow(['id','value'])
    for n in range(num_of_data):
        predictValue[n] = predictValue[n] + w_vec[0]#w0
        for m in range(d):
            tmp_w = w_vec[1+dim_of_data*m : 1+dim_of_data*(m+1) ]
            tmp_w = np.reshape(tmp_w,(dim_of_data,) )
            predictValue[n] = predictValue[n] + np.dot( tmp_w ,(testData[n])**(m+1))#w_n dot data**n
        result_csv.writerow( [ 'id_'+str(n), str(predictValue[n]) ] )

def check_sol_higher_d(trainData, trainLabel,d):
    y = trainLabel
    num_of_data = len(trainData)
    dim_of_data = len(trainData[0])
    A = np.zeros( (num_of_data, d*dim_of_data+1) )
    for n in range(num_of_data):
        for m in range( d*dim_of_data+1):
            if( m == 0):
                A[n][m] = 1
                continue
            which_order = int( np.floor( (m-1)/dim_of_data) + 1)
            which_dim = (  (m-1)%dim_of_data ) 
            tmp_data = (trainData[n][which_dim])**(which_order)
            A[n][m] = tmp_data
    #print(A)
    w_vec = np.dot( np.dot( inv( np.dot(np.transpose(A),A) ), np.transpose(A)) ,y)
    return w_vec

def readModel_2d(modelFilename, data):
    originalFile = open(modelFilename,'r',errors='ignore')
    dim_of_data = len(data[0])
    w_vec = np.zeros( (2*dim_of_data+1,))
    myflag = 1
    for row in csv.reader(originalFile,delimiter=','):
        if(myflag == 1):
            w_vec[0] = row[0]
            w_vec[1:1+dim_of_data] = row[1:1+dim_of_data]
            w_vec[1+dim_of_data:1+2*dim_of_data] = row[1+dim_of_data:1+2*dim_of_data]
            myflag = 0
    return w_vec
#main script below
np.random.seed(0)
cv_idx_tmp = np.random.permutation(5640)
#for 4 fold-cross validation, 1400 1400 1400 1440
cv_fold_1 = cv_idx_tmp[0:1400]
cv_fold_2 = cv_idx_tmp[1400:2800]
cv_fold_3 = cv_idx_tmp[2800:4200]
cv_fold_4 = cv_idx_tmp[4200:5640]

chosen_features = [3,8,10,15,17]
target_idx = 10#idx of target, that is, pm2.5

trainFilename = 'train.csv'#fixed path
testFilename = sys.argv[1]#'test.csv'

modelFilename = 'model.csv'#fixed path
outputFilename = sys.argv[2]

trainData, trainLabel = readTrainData(trainFilename,chosen_features,target_idx)
#trainDataNormalized = normalizeData(trainData)

testData = readTestData(testFilename,chosen_features)
#testDataNormalized = normalizeData(testData)

#print(trainData[180,:])
#print(trainLabel[180])
#print(trainData[181,:])

#w_vec = trainModelLinear_1d(trainData, trainLabel)
#w_vec = trainModelLinear_2d(trainData, trainLabel)#only for demo of trainging
#w_vec = 0

w_vec = readModel_2d(modelFilename, trainData)#read w_Vec from best model

d = 2

#w_vec = check_sol_higher_d(trainData, trainLabel,d)

writePredict_higher_d(testData,w_vec,d,outputFilename)
myLoss = getLoss_higher_d(trainData, trainLabel, w_vec,d)
print('my training loss from best model = ',myLoss)
