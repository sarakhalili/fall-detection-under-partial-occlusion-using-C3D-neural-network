from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
import numpy as np
ar = np.array
import cv2
import pickle
from sklearn import svm
from scipy.signal import find_peaks
import os; import glob
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Dense
from keras.constraints import maxnorm
import re
from tqdm import tqdm        

def create_model_sequential():
    """ Creates model object with the sequential API:
    https://keras.io/models/sequential/
    """

    model = Sequential()
    input_shape = (16, 112, 112, 3)

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
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
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    return model


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_c3d():                       # defining C3D network

    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    model = create_model_sequential()
    try:
        model.load_weights('/content/drive/MyDrive/C3D_Sport1M_weights_keras_2.2.4.h5')   #weights of the network should be downloaded 
    except OSError as err:
        print('Check path to the model weights\' file!\n\n', err)
    except :
        print("errooooor")

    C3D = Model(inputs=model.input,
                      outputs=[ model.get_layer('fc6').output,
                               model.get_layer('flat5').output,
                               model.get_layer('fc7').output,
                               model.get_layer('fc8').output])
    CNN_FC5 = models.Model(inputs=C3D.inputs, outputs=C3D.layers[-6].output)          # defining a network only with the convolution part of the C3D network (essential for training the FCN)
    CNN_FC5.summary()
    del model
    return CNN_FC5

CNN_FC5=load_c3d()

       
        
        
# ===========================
def ext(l,c3d):                 # This function gives the result of c3d network for the given input frames
    
    frames16 = l[:16].reshape((1,16,112,112,3)).astype('int') -100 # .astype('int')*255
    features_4layers = c3d.predict(frames16)
    return features_4layers



class extraction():                 # This class returns extracted features and their labels
    def __init__(self, dataset_name, options, label,ds_addr,data_addr):
        self.ds_addr = ds_addr;             # address of the dataset location
        self.data_addr = data_addr;         # address of the extracted features
        self.dataset_name = dataset_name;   # name of the folder
        self.label = label
        self.options = options
        self.dsrate = options['downsample_rate']
        self.stride = options['stride'] 

    def extract(self):
        print(self.label, "extracting ...")
        if self.dataset_name not in os.listdir(self.ds_addr):
            print("error , folder with dataset_name was not found\n\n")
  
            
        f = open(self.ds_addr+ self.dataset_name+'/'+self.dataset_name+'.txt', "r+")
        annot = f.readlines()                                                                 # reading lines of each test file
        
        self.annot=[]
        f.close()
        videos = glob.glob(self.ds_addr+ self.dataset_name+'/*.avi')                          #reading videos of the dataset folder
        videos.sort(key=natural_keys)        
        CNN_FC5=load_c3d()                                                                    # loading only the convolution part of the C3D network

        fc5_n=[]; labels_n=[];
        for j,video in (enumerate(videos)):                                  

            start=int(annot[j].split()[0])
            end=int(annot[j].split()[1])
            f_all=int(annot[j].split()[2])

            capture = cv2.VideoCapture( video )
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if f_all!=num_frame:
                print("error, num frames", num_frame,f_all)
            if not capture.isOpened:
                print('Unable to open: ' + video)
                exit(0)
            
            frame16= []
            i=0
            for i in range(0, num_frame):
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
                print("error... not enough frames")
                continue
            sample_number = (len(frame16)-c3dlength)// self.stride +1       # calculating the number of input segments         
            for k in range(sample_number):
                sc=k*self.stride; ec=k*self.stride+16                       # applying stride to the frames
                xn = ext(frame16[sc:ec],CNN_FC5)
                fc5_n+=[xn]
                if start==0 and end==0:
                  label=[0,1]   
                elif (sc*self.dsrate)>=start and (ec*self.dsrate)<=end:
                  label=[1,0]
                else:
                  label=[0,1]      
                labels_n+=[label]
                

        self.fc5_n=fc5_n
        print(np.shape(fc5_n))
        np.save(self.data_addr +"/{}_fc5.npy".format(self.dataset_name),fc5_n)
        np.save(self.data_addr +"/{}_labels.npy".format(self.dataset_name),labels_n)
        return fc5_n,labels_n
        


