
def fcn():
    """ Creates model object with the sequential API:
    https://keras.io/models/sequential/
    """

    model = Sequential()
    input_shape = 8192

    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6',input_dim=input_shape))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='relu', name='fc7_2'))
    model.add(Dropout(.05))
    model.add(Dense(2, activation='softmax', name='fc8'))

    return model

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from keras.layers.merge import concatenate


def load_fcn(fcn):             # defining the trained FCN 

    model = fcn()
    try:
        model.load_weights("/content/drive/MyDrive/biased_weights_10.hdf5")
    except OSError as err:
        print('Check path to the model weights\' file!\n\n', err)
    except :
        print("errooooor")
    fcn = Model(inputs=model.input,
                      outputs=[ model.get_layer('fc6').output,
                               model.get_layer('fc7').output,
                               model.get_layer('fc7_2').output,
                               model.get_layer('fc8').output])

    # fcn.summary()
    del model
    return fcn

# del Xtrain,Xtest,Xvalid,Ytrain,Ytest,Yvalid
# del Xntrain,Xntest,Xnvalid,Yntrain,Yntest,Ynvalid
# del Xotrain,Xotest,Xovalid,Yotrain,Yotest,Yovalid

import numpy as np
ar = np.array
import cv2
import pickle
from sklearn import svm
from scipy.signal import find_peaks
import os; import glob
from tqdm import tqdm  
import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

c3d_len = 16
             
class extraction():                 # in class ham kollan feature exraction va label gozari anjam mide
    def __init__(self, dataset_name, options, label,ds_addr,data_addr):
        self.ds_addr = ds_addr; 
        self.data_addr = data_addr; 
        self.dataset_name = dataset_name; 
        self.label = label
        self.options = options
        self.dsrate = options['downsample_rate']
        self.stride = options['stride'] #2 # was 2 meaning 8 frame steps. # 2 when stride=4


    def load(self):                                               #this function either load feature-all or extract it
        if new_train==True:
          self.extract()
        else:
          try:
              filename =self.data_addr+ self.label
              infile = open(filename,'rb')
              self.features_all = pickle.load(infile)
              filename =self.data_addr+ self.label+'_labels'
              infile = open(filename,'rb')
              self.labels = pickle.load(infile)            
              infile.close()
              print(self.label, "loaded successfully")
          except:           
            self.extract()
                
    def extract(self):
        print(self.label, "extracting ...")
        if self.dataset_name not in os.listdir(self.ds_addr):
            print("error , folder with dataset_name was not found\n\n")             
        
        CNN_FC5 = load_c3d()                                                                    # loading only the convolution part of the C3D network 
        features_all = {}; labels=[]
        f = open(self.ds_addr+ self.dataset_name+'/'+self.dataset_name+'.txt', "r+")
        annot = f.readlines()                                                                 # reading lines of each text file
        f.close()
        videos = glob.glob(self.ds_addr+ self.dataset_name+'/*.avi')                          #reading videos of the dataset folder
        videos.sort(key=natural_keys)

        for i,video in (enumerate(videos)):                           
            # print(video, annot[i])
            capture = cv2.VideoCapture( video )
            start=int(annot[i].split()[0])
            end=int(annot[i].split()[1])
            f_all=int(annot[i].split()[2])
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if f_all!=num_frame:
                print("error, num frames", num_frame,f_all)
            if not capture.isOpened:
                print('Unable to open: ' + videos)
                exit(0)
            
            frame16= []
            i=0
            for i in range(0, num_frame):
                # print(num_frame)
                ret, frame = capture.read()
                    
                if i% self.dsrate==0:                                     # applying downsampling to the frames
                    frame_resized=frame
                    frame16 += [frame_resized]
                if frame is None:
                    break
        #============================================
            frame16 = ar(frame16)
            c3dlength = 16
            if len(frame16)<16 :
                # print("error... not enough frames")
                continue
            sample_number = (len(frame16)-c3dlength)// self.stride +1       # calculating the number of input segments      
            
            features = []; count=0
            for i in range(sample_number):
                sc=i*self.stride; ec=i*self.stride+16                       # applying stride to the frames
                f = ext(frame16[sc:ec],CNN_FC5)
                features += [f]
                if start==0 and end==0:
                  label=-1
                elif (sc*self.dsrate)>=start and (ec*self.dsrate)<=end:
                  label=1
                  count+=1                 
                else:
                  label=-1     
                labels+=[label]
            # print(count)
            features_all.update({video:features})

        self.labels=(labels)
        self.features_all = features_all               
        filename =self.data_addr+ self.label
        outfile = open(filename,'wb')
        outfile_l=open(filename+'_labels','wb')
        pickle.dump(features_all ,outfile)
        pickle.dump(self.labels ,outfile_l)
        outfile.close()
        del CNN_FC5
        
    def load_layer(self,layer): # loads data of nth layer of C3D with labels.
        layer_features=[];
        for k,v in self.features_all.items():      
            layer_features +=[clipdata[layer] for clipdata in v]
        print(np.shape(np.array(layer_features)))
        return np.array(layer_features),self.labels    

def ext(l,c3d):                 # this function first calculates the output of fc5 and then use it as input of FCN
    # 16 black frames with 3 channels
    frames16 = l[:16].reshape((1,16,112,112,3)).astype('int') -100 # .astype('int')*255
    flt = c3d.predict(frames16)   
    features_4layers=fcn2.predict(flt)
    del flt
    return features_4layers