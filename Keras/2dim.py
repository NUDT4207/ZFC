# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:09:48 2020

@author: x1c
"""

import keras
import numpy as np
from keras.models import Sequential
#import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.layers import Dense
#import pywt
#import generategif
import tensorflow as tf
from keras import backend as K
from scipy import misc


import os
import argparse
parser = argparse.ArgumentParser(description = 'lstm of mnist')
parser.add_argument('--gpu', '-g', help='gpu use?', default='0')
parser.add_argument('--w', '-w', help='weight?', default=1e-1)
parser.add_argument('--b', '-b', help='bias?', default=1e-1)
parser.add_argument('--epoch', '-e', help='epoch?', default=10)
parser.add_argument('--item', '-i', help='tiem num?', default=1000)
parser.add_argument('--max', '-m', help='input max num?', default=1)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(args)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)


beta=0.95
num_classes   = 10
delta         = 7
alpha=0.5
num=600
num_fre=40
border=1
params=[2,2000,1000,500,200,1]

def build_model(optimizer=keras.optimizers.adam(),loss='mse',activation='relu',
                w=1e-1,b=1e-1):
    def RandNormal2_w(shape, dtype=None,mean=0,stddev=w):
        if dtype == None:
            dtype = 'float32'
        result_1 = K.random_normal(shape, dtype=dtype,mean=3*stddev, stddev=stddev)
        result_2 = K.random_normal(shape, dtype=dtype,mean=-3*stddev, stddev=stddev)
        return result_1+result_2
    def RandNormal2_b(shape, dtype=None,mean=0,stddev=b):
        if dtype == None:
            dtype = 'float32'
        result_1 = K.random_normal(shape, dtype=dtype,mean=3*stddev, stddev=stddev)
        result_2 = K.random_normal(shape, dtype=dtype,mean=-3*stddev, stddev=stddev)
        return result_1+result_2
    model = Sequential()
    #kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=w, seed=None)
    #kernel_initializer=keras.initializers.RandomUniform(minval=-w, maxval=w, seed=None)
    #kernel_initializer=keras.initializers.lecun_uniform(seed=None)
    kernel_initializer = RandNormal2_w
    #bias_initializer = RandNormal2_b
    bias_initializer=keras.initializers.RandomNormal(mean=0, stddev=b, seed=None)
    #bias_initializer = keras.initializers.RandomUniform(minval=-b, maxval=b, seed=None)
    #bias_initializer=keras.initializers.Constant(value=0)
    #bias_initializer=keras.initializers.lecun_uniform(seed=None)
    #initializer=keras.initializers.orthogonal()
    model.add(Dense(params[1], activation=activation, 
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    input_dim=params[0]))
    for i in range(2,len(params)-1):
        model.add(Dense(params[i], activation=activation, 
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer))
    model.add(Dense(1))
    #optimizer = keras.optimizers.Adagrad(lr=10**(-2), epsilon=None, decay=0.0)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def generate_data(pic_name,max_num=1):
    camera = misc.imread(pic_name)
    #plt.imshow(camera,cmap='gray')
    #plt.axis('off')
    x1_max = camera.shape[0]
    x2_max = camera.shape[1]
    x1 = np.linspace(-1,1,x1_max)*max_num
    x2 = np.linspace(-1,1,x2_max)*max_num
    #x_train = np.dot(x1,x2)
    #前面竖向拓展,后面横向拓展
    x2,x1 = np.meshgrid(x2,x1)
    x_train = np.zeros((x1_max,x2_max,2))
    x_train[:,:,0] = x1
    x_train[:,:,1] = x2
    x_train = x_train.astype('float32')
    x_train = x_train.reshape(-1,2)
    y_train = camera.astype('float32')
    y_train = y_train.reshape(-1,1)
    shape = camera.shape
    return x_train,y_train,shape
    
def im_show(pic,shape):
    #pic = pic.reshape(512,512)
    pic = pic.reshape(shape)
    pic = pic.astype('uint8')
    plt.imshow(pic,cmap='gray')
    plt.axis('off')
    
    
    
if __name__=='__main__':
    pic_name = 'circle.jpg'
    #pic_name = 'camera.png'
    #pic_name = 'gezi.jpg'
    max_num = int(args.max)
    x_train,y_train,shape = generate_data(pic_name,max_num=max_num)
    #im_show(y_train,shape)
    #plt.show()
    
    model = build_model(optimizer=keras.optimizers.adam(),
                        loss='mse',
                        activation='relu',
                        w=float(str(args.w)),b=float(str(args.b)))
    for i in range(int(args.item)):
        model.fit(x_train,y_train,batch_size=256,shuffle=True)
        fig = plt.gcf()
        y_pre = model.predict(x_train)
        im_show(y_pre,shape)
        dirs = './my_circle/max_'+str(max_num)+'_w_'+str(float(str(args.w)))+'_b_'+str(float(str(args.b)))+'/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        fig.savefig(dirs+str(i)+'.png')
        #plt.show()
        plt.cla()
    
    
    









