#this code tries to extract features using fcn of C3d and classify them with SVM
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import classification_report, confusion_matrix
ar = np.array
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
# ===========================
new_train=False
# new_train=True
def ext(l,c3d):                 #this function first reshapes l into 16*112*112*3  then gives it to c3d to predict
    frames16 = l[:16].reshape((1,16,112,112,3)).astype('int') -100 # .astype('int')*255
    flt = c3d.predict(frames16)   
    features_4layers=fcn2.predict(flt)
    del flt
    return features_4layers

options={'downsample_rate':1 , 'background_sub':False, 'stride':2}
# datasets = [[['train'],['test']]]
# datasets = [[['mine','mine_fall'],['mine2']]]
datasets=[['Home_01','Home_02','Office','Office2','ur_fall','ur_adl','Lecture_room','Coffee_room_01','mine','mine_fall','mine2']]

layers = [0,1,2]
gridCV = {
    'C': [1,10,100],
    'gamma': [0.00001],
        }
for folder in datasets:
    layerscores = []        
    for layer in layers:
        print('\n Layer {}'.format(layer))
        fcn2=load_fcn(fcn)
        Xntrain=[]; Xntest = []
        Yntrain=[]; Yntest = []
        Xotrain=[]; Xotest = []
        Yotrain=[]; Yotest = []

        options={'downsample_rate':1 , 'background_sub':False, 'stride':2}
        ds_addr= '/content/drive/MyDrive/full_frame/cross_subject/original/'
        data_addr = '/content/drive/MyDrive/full_frame/data/trained/biased_org/'
        
        for i,ds in enumerate(folder):
            ds=extraction(ds, options , ds,ds_addr,data_addr)
            ds.load()
            x,y=ds.load_layer(layer)
            print(x.shape)
            Xntrain+=[x]; Yntrain+=y
        Xntrain = np.concatenate(Xntrain, axis=0)[:,0,:]
        Xntrain, Xnvalid, Yntrain,Ynvalid = train_test_split(Xntrain, Yntrain, test_size=0.15 )
        Xntrain, Xntest, Yntrain,Yntest = train_test_split(Xntrain, Yntrain, test_size=0.2 )

        options={'downsample_rate':1 , 'background_sub':False, 'stride':32}
        ds_addr= '/content/drive/MyDrive/full_frame/cross_subject/openpose/'
        data_addr = '/content/drive/MyDrive/full_frame/data/trained/biased_cnt/'
        
        for i,ds in enumerate(folder):
            ds=extraction(ds, options , ds,ds_addr,data_addr)
            ds.load()
            x,y=ds.load_layer(layer)
            print(x.shape)
            Xotrain+=[x]; Yotrain+=y
        Xotrain = np.concatenate(Xotrain, axis=0)[:,0,:]
        Xotrain, Xovalid, Yotrain,Yovalid = train_test_split(Xotrain, Yotrain, test_size=0.15 )
        Xotrain, Xotest, Yotrain,Yotest = train_test_split(Xntrain, Yntrain, test_size=0.2 )
        ####################################################################

        Xtrain=np.concatenate((Xntrain,Xotrain),axis=0);  Ytrain=np.concatenate((Yntrain,Yotrain),axis=0)
        maximum=np.amax(Xtrain); Xntrain=Xntrain/maximum; Xotrain=Xotrain/maximum
        Xtest=np.concatenate((Xntest,Xotest),axis=0);     Ytest=np.concatenate((Yntest,Yotest),axis=0)
        maximum=np.amax(Xtest); Xntest=Xntest/maximum; Xotest=Xotest/maximum
        Xvalid=np.concatenate((Xnvalid,Xovalid),axis=0);     Yvalid=np.concatenate((Ynvalid,Yovalid),axis=0)
        maximum=np.amax(Xvalid); Xnvalid=Xnvalid/maximum; Xovalid=Xovalid/maximum
        print("Xtrain is:",np.shape(Xtrain),"          Xvalid is:",np.shape(Xvalid),"          Xtest is:",np.shape(Xtest))   
        print("Ytrain is:",np.shape(Ytrain),"          Yvalid is:",np.shape(Yvalid),"          Ytest is:",np.shape(Ytest))  
###################################################################################################
        o=len(Xotrain[:,0]);          n=len(Xntrain[:,0])
        ratio=(n/o)
        lamda=0.5
        rrr=lamda*ratio
        c1=np.ones((n)); c2=rrr*np.ones((o))        #giving each sample a weight to reduce occluded data loss
        ccc=np.concatenate((c1,c2),axis=0)

        from sklearn.utils.class_weight import compute_class_weight        #calculating class weights in train dataset
        class_weights = compute_class_weight('balanced', np.unique(Ytrain), Ytrain)
        d_class_weights = {-1:class_weights[0], 1: (4*class_weights[1])}
        print(d_class_weights)
###################################################################################################
        del ds
        pca=sklearnPCA(n_components=200)    #reducing feature number
        X = np.concatenate([Xtrain, Xtest, Xvalid])
        pca.fit(X)
        Xtrain = pca.transform(Xtrain)
        Xvalid = pca.transform(Xvalid)
        Xtest = pca.transform(Xtest)        
        del X; del pca

        scorel=[]
        param=[]
        for c in gridCV['C']:         #finding the best parameters in SVM
            for g in gridCV['gamma']:
                svclassifier1 = SVC(kernel='rbf', C=c, gamma=g,class_weight=d_class_weights)
                svclassifier1.fit(Xtrain,Ytrain,sample_weight=ccc)
                Ypred = svclassifier1.predict(Xvalid)
                falsealarm=0; total = len(Ypred); miss=0; alarm=0;
                for i in range(len(Ypred)):
                    if Ypred[i]==1 and Yvalid[i]==-1:
                        falsealarm+=1                   
                    if Ypred[i]==-1 and Yvalid[i]==1:
                        miss+=1
                    if Ypred[i]==1 and Yvalid[i]==1:
                        alarm+=1                               
                s = alarm/(miss+alarm)* (1-falsealarm/(total-alarm-miss))#svclassifier1.score(xvalid, yvalid)
                scorel +=[s]
                param  +=[[c,g,s]]

        bestparam= param[np.argmax(scorel)]
        print('best parameters in validation C {}   gamma {} score(sens*falarm {}'.format(bestparam[0], bestparam[1],bestparam[2]))

        # finding the performance of the best parameters with test dataset
        svclassifier1 = SVC(kernel='rbf', C=bestparam[0], gamma=bestparam[1],class_weight=d_class_weights)
        # svclassifier1.fit(Xtrain,Ytrain,sample_weight=c)
        svclassifier1.fit(np.append(Xtrain,Xvalid,axis=0),np.append(Ytrain,Yvalid,axis=0))
        s = svclassifier1.score(Xtest, Ytest)
        Ypred = svclassifier1.predict(Xtest)
        falsealarm=0; total = len(Ypred); miss=0; alarm=0;
        for i in range(len(Ypred)):
            if Ypred[i]==1 and Ytest[i]==-1:
                falsealarm+=1
            if Ypred[i]==-1 and Ytest[i]==1:
                miss+=1
            if Ypred[i]==1 and Ytest[i]==1:
                alarm+=1
        print('\n Layer {}'.format(layer))
        print('tot', total, 'alarm', alarm, 'f alrm', falsealarm,'miss',miss, 'else', total-alarm-falsealarm-miss)
        print('SVMscore{:.3} sens{:.3} falarm{:.3} '.format\
              (s, alarm/(miss+alarm), falsealarm/(total-alarm-miss)), 'c,g ', bestparam[:2])
        layerscores += [[layer , alarm/(miss+alarm)*(1- falsealarm/(total-alarm-miss))]]     

        print("classification_report:\n",classification_report(Ytest, Ypred, labels=[-1,1]))
        print("confusion_matrix:\n",confusion_matrix(Ytest, Ypred))

        new_train=False
    n = np.argmax([l[1] for l in layerscores])            
    print('\nFinal Best Layer {}  sens*falarm {:.3f}'.format(layerscores[n][0], layerscores[n][1]))        