import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

from tensorflow.python.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
ar = np.array
new_train=False

########################################################################## un_occluded videos
# !rm -rf '/content/drive/MyDrive/full_frame/data/conv4/original/'
# !mkdir '/content/drive/MyDrive/full_frame/data/conv4/original/'
ds_addr= '/content/drive/MyDrive/full_frame/cross_subject/original/'
data_addr = '/content/drive/MyDrive/full_frame/data/conv4/original/'
options={'downsample_rate':1 , 'background_sub':False, 'stride':4}


# datasets=[['Home_01','Home_02','Office','Office2','Coffee_room_01','ur_fall','ur_adl','Lecture_room','mine','mine_fall']]
datasets=[['train']]
for j,folder in (enumerate(datasets)):

  for i,name in (enumerate(folder)):
      try:
        fc5_n=np.load(data_addr+"/{}_fc5.npy".format(name))
        labels_n=np.load(data_addr+"/{}_labels.npy".format(name))
        print("{} loaded successfully".format(name))
      except:
        ds=extraction(name, options , name,ds_addr,data_addr)
      
        fc5_n,labels_n=ds.extract()
        fc5_n=ar(fc5_n); labels_n=ar(labels_n)  
      if i==0:
        if len(fc5_n)!=0:
          fc5n_all=fc5_n
          labelsn_all=labels_n
      else:
        if len(fc5_n)!=0:
          fc5n_all=np.concatenate((fc5n_all,fc5_n),axis=0)
          labelsn_all=np.concatenate((labelsn_all,labels_n),axis=0)
        
     
##################################################reshaping fc5 and labels_all for network ###################################

fc5n_all=np.reshape(fc5n_all,(-1,2, 7, 7, 512) ) 

size=np.size(labelsn_all); size=int(size/2)
labelsn_all=ar(labelsn_all)
labelsn_all=np.reshape(labelsn_all,(size,2))
Xntrain=fc5n_all; Yntrain=labelsn_all
del fc5n_all,labelsn_all
# datasets=[['Home_01','Home_02','Office','Office2','Coffee_room_01','ur_fall','ur_adl','Lecture_room','mine','mine_fall']]
datasets=[['test']]
for j,folder in (enumerate(datasets)):

  for i,name in (enumerate(folder)):
      try:
        fc5_n=np.load(data_addr+"/{}_fc5.npy".format(name))
        labels_n=np.load(data_addr+"/{}_labels.npy".format(name))
        print("{} loaded successfully".format(name))
      except:
        ds=extraction(name, options , name,ds_addr,data_addr)
      
        fc5_n,labels_n=ds.extract()
        fc5_n=ar(fc5_n); labels_n=ar(labels_n)  
      if i==0:
        if len(fc5_n)!=0:
          fc5n_all=fc5_n
          labelsn_all=labels_n
      else:
        if len(fc5_n)!=0:
          fc5n_all=np.concatenate((fc5n_all,fc5_n),axis=0)
          labelsn_all=np.concatenate((labelsn_all,labels_n),axis=0)
        
     
##################################################reshaping fc5 and labels_all for network ###################################
fc5n_all=np.reshape(fc5n_all,(-1,2, 7, 7, 512) ) 

size=np.size(labelsn_all); size=int(size/2)
labelsn_all=ar(labelsn_all)
labelsn_all=np.reshape(labelsn_all,(size,2))
Xnvalid=fc5n_all; Ynvalid= labelsn_all
del fc5n_all,labelsn_all

########################################################################## occluded videos

!rm -rf '/content/drive/MyDrive/full_frame/data/conv4/openpose/'
!mkdir '/content/drive/MyDrive/full_frame/data/conv4/openpose/'
ds_addr= '/content/drive/MyDrive/full_frame/cross_subject/openpose/'
data_addr = '/content/drive/MyDrive/full_frame/data/conv4/openpose/'
options={'downsample_rate':1 , 'background_sub':False, 'stride':16}

# datasets=[['Home_01','Home_02','Office','Office2','Coffee_room_01','ur_fall','ur_adl','Lecture_room','mine']]
datasets=[['train']]
for j,folder in (enumerate(datasets)):

  for i,name in (enumerate(folder)):
      try:
        fc5_o=np.load(data_addr+"/{}_fc5.npy".format(name))
        labels_o=np.load(data_addr+"/{}_labels.npy".format(name))
        print("{} loaded successfully".format(name))
      except:
        ds=extraction(name, options , name,ds_addr,data_addr)
      
        fc5_o,labels_o=ds.extract()
        fc5_o=ar(fc5_o);  labels_o=ar(labels_o)     
      if i==0:
        if len(fc5_o!=0):
          fc5o_all=fc5_o
          labelso_all=labels_o

      else:
        if len(fc5_o!=0):
          fc5o_all=np.concatenate((fc5o_all,fc5_o),axis=0)
          labelso_all=np.concatenate((labelso_all,labels_o),axis=0)    
##################################################reshaping fc5 and labels_all for network ###################################

d1,d2,d3=np.shape(fc5o_all)
fc5o_all=np.reshape(fc5o_all,(d1,d3)) 

size=np.size(labelso_all); size=int(size/2)
labelso_all=ar(labelso_all)
labelso_all=np.reshape(labelso_all,(size,2))
Xotrain=fc5o_all; Yotrain= labelso_all
del fc5o_all,labelso_all

datasets=[['test']]
for j,folder in (enumerate(datasets)):

  for i,name in (enumerate(folder)):
      try:
        fc5_o=np.load(data_addr+"/{}_fc5.npy".format(name))
        labels_o=np.load(data_addr+"/{}_labels.npy".format(name))
        print("{} loaded successfully".format(name))
      except:
        ds=extraction(name, options , name,ds_addr,data_addr)
      
        fc5_o,labels_o=ds.extract()
        fc5_o=ar(fc5_o);  labels_o=ar(labels_o)     
      if i==0:
        if len(fc5_o!=0):
          fc5o_all=fc5_o
          labelso_all=labels_o

      else:
        if len(fc5_o!=0):
          fc5o_all=np.concatenate((fc5o_all,fc5_o),axis=0)
          labelso_all=np.concatenate((labelso_all,labels_o),axis=0)
        
     
##################################################reshaping fc5 and labels_all for network ###################################

d1,d2,d3=np.shape(fc5o_all)
fc5o_all=np.reshape(fc5o_all,(d1,d3)) 

size=np.size(labelso_all); size=int(size/2)
labelso_all=ar(labelso_all)
labelso_all=np.reshape(labelso_all,(size,2))

Xovalid=fc5o_all; Yovalid= labelso_all
del fc5o_all,labelso_all
    #####################################################################################################    
# print(np.shape(fc5n_all),np.shape(labelsn_all),np.shape(fc5o_all),np.shape(labelso_all))
# Xn2, Xntest, Yn2,Yntest = train_test_split(fc5n_all, labelsn_all, test_size=0.2, random_state=42 )
# Xntrain, Xnvalid, Yntrain,Ynvalid = train_test_split(Xn2, Yn2, test_size=0.2 , random_state=42) 
# print("Normal Xtrain is:",np.shape(Xntrain),"          Normal Xvalid is:",np.shape(Xnvalid),"          Normal Xtest is:",np.shape(Xntest))   
# print("Normal Ytrain is:",np.shape(Yntrain),"          Normal Yvalid is:",np.shape(Ynvalid),"          Normal Ytest is:",np.shape(Yntest))    
# del fc5n_all,labelsn_all,Xn2,Yn2     
    #####################################################################################################  

# Xo2, Xotest, Yo2,Yotest = train_test_split(fc5o_all, labelso_all, test_size=0.2, random_state=42 )
# Xotrain, Xovalid, Yotrain,Yovalid = train_test_split(Xo2, Yo2, test_size=0.2 , random_state=42) 
# print("Occluded Xtrain is:",np.shape(Xotrain),"          Occluded Xvalid is:",np.shape(Xovalid),"          Occluded Xtest is:",np.shape(Xotest))   
# print("Occluded Ytrain is:",np.shape(Yotrain),"          Occluded Yvalid is:",np.shape(Yovalid),"          Occluded Ytest is:",np.shape(Yotest))  
# del fc5o_all,labelso_all,Xo2,Yo2         
    ##################################################################################################### 

Xtrain=np.concatenate((Xntrain,Xotrain),axis=0);  Ytrain=np.concatenate((Yntrain,Yotrain),axis=0)
maximum=np.amax(Xtrain); Xntrain=Xntrain/maximum; Xotrain=Xotrain/maximum
# Xtest=np.concatenate((Xntest,Xotest),axis=0);     Ytest=np.concatenate((Yntest,Yotest),axis=0)
Xvalid=np.concatenate((Xnvalid,Xovalid),axis=0);     Yvalid=np.concatenate((Ynvalid,Yovalid),axis=0)
maximum=np.amax(Xvalid); Xovalid=Xovalid/maximum; Xnvalid=Xnvalid/maximum; Xvalid=Xvalid/maximum
# Xvalid, Xtest, Yvalid,Ytest = train_test_split(Xvalid, Yvalid, test_size=0.5)
    ##################################################################################################### 

# fc_all=np.concatenate((fc5n_all,fc5o_all),axis=0)
# labels_all=np.concatenate((labelsn_all,labelso_all),axis=0)

# print(np.shape(fc_all),np.shape(labels_all))
# X2, Xtest, Y2,Ytest = train_test_split(fc_all, labels_all, test_size=0.3, random_state=42 )
# Xtrain, Xvalid, Ytrain,Yvalid = train_test_split(X2, Y2, test_size=0.2 , random_state=42) 
#     ##################################################################################################### 

# print("Xtrain is:",np.shape(Xtrain),"          Xvalid is:",np.shape(Xvalid),"          Xtest is:",np.shape(Xtest))   
# print("Ytrain is:",np.shape(Ytrain),"          Yvalid is:",np.shape(Yvalid),"          Ytest is:",np.shape(Ytest))  
#     ##################################################################################################### 

def fcn():             # creating fully connected part of C3D network for training

    model = Sequential()
    input_shape = (2, 7, 7, 512) 
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten(name='flat5'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6',input_dim=input_shape))
    model.add(Dropout(.6))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.6))
    model.add(Dense(487, activation='relu', name='fc7_2'))
    model.add(Dropout(.5))
    model.add(Dense(2, activation='softmax', name='fc8'))
    return model

# fcn_nt=fcn()
# fcn_nt.summary()
######################################################################################################################

from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(Yntrain, axis=1)
class_weights_n = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights_n = dict(enumerate(class_weights_n))
a = d_class_weights_n[0]; a=3*a
d_class_weights_n[0]=a
type(d_class_weights_n[1])
print(d_class_weights_n)

y_integers = np.argmax(Yotrain, axis=1)
class_weights_o = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights_o = dict(enumerate(class_weights_o))
a = d_class_weights_o[0]; a=3*a
d_class_weights_o[0]=a
type(d_class_weights_o[1])
print(d_class_weights_o)
##########################################################################################################
epochs=1
n=len(Xntrain)
o=len(Xotrain)
ratio=(n/o)
lamda=0.805


batch_size=8192
momentom=0.937
i_iteration=50
lr=0.01
##########################################################################################################
output=[]
for gam in range(40,41,1):
    lamda=gam/100
    rrr=lamda*ratio
    sgd = SGD(lr=lr, momentum=momentom, nesterov=True)
    optimizer =Adam(lr=lr,beta_1=momentom) 
    fcn_nt=fcn()
    fcn_nt.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #########################################################################################################################3

    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    checkpoint = ModelCheckpoint("/content/drive/MyDrive/biased_weights_10.hdf5", monitor='val_loss',
                                verbose=1, save_best_only=True, save_weights_only=True,
                                mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0,
                          patience=100, verbose=1, mode='auto',restore_best_weights=True)
    ########################repeat two steps for number of iteration times
    result=[]
    for i in range(i_iteration):
        print("number of iteration is:    ",i)
        ###############################first train the last layer with normal loss
        learning_rate=lr
        for layer in fcn_nt.layers[:]:
            layer.trainable = True
        
        history = fcn_nt.fit(x=Xntrain, y=Yntrain,
                            validation_data=(Xvalid, Yvalid),
                            class_weight = d_class_weights_n,
                            batch_size=batch_size,
                            epochs=epochs,
                             verbose=1,callbacks=[checkpoint,early])     #[checkpoint,early] [early]  
        
        ################################# now train the rest of the network with total loss
        learning_rate=rrr*lr                  #changing lr to reduce the loss of occluded data
        # fcn_nt.layers[-4].trainable = False
        # fcn_nt.layers[-3].trainable = False
        # fcn_nt.layers[-2].trainable = False
        # fcn_nt.layers[-1].trainable = False
        history = fcn_nt.fit(x=Xotrain, y=Yotrain,
                            validation_data=(Xvalid, Yvalid),
                            class_weight = d_class_weights_o,
                            batch_size=batch_size,
                            epochs=epochs,
                             verbose=1,callbacks=[checkpoint,early])     #[checkpoint,early] [early]                      
        accuracy=history.history['accuracy']; val_accuracy=history.history['val_accuracy']
        loss=history.history['loss']; val_loss=history.history['val_loss']
        result.append([accuracy,val_accuracy,loss,val_loss])
    result=np.array(result)
    val_loss=min(result[:,-1])[0]
    output.append([val_loss,lamda])
#########################################################################################here we calculate test loss and accuracy
from sklearn.metrics import classification_report, confusion_matrix
score = fcn_nt.evaluate(Xtest, Ytest, verbose=False) 
fcn_nt.metrics_names
print('Test loss: ', score[0])    #Loss on test
print('Test accuracy: ', score[1])

########################################################here we claculate classification report and confusion matrix of test data
Ytest_pred=ar(fcn_nt.predict(Xtest))
targ=[]
for i in range(len(Ytest_pred)):
  if Ytest_pred[i,0]>Ytest_pred[i,1]:
    targ+=[1]
  else:
    targ+=[0]
Ytest=ar(Ytest)
Ytest2=Ytest[:,0]

print("classification_report:\n",classification_report(Ytest2, targ, labels=[0,1]))
print("confusion_matrix:\n",confusion_matrix(Ytest2, targ))
# list all data in history
# print(result.history.keys())
# summarize history for accuracy
result=np.array(result)
result=np.reshape(result,(-1,4))
import matplotlib.pyplot as plt
plt.plot(result[:,0])
plt.plot(result[:,1])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(result[:,2])
plt.plot(result[:,3])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
