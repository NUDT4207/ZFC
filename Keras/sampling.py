# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:01:42 2019

@author: x1c
"""

import keras
import numpy as np
from keras.models import Sequential
import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.layers import Dense
import pywt
import generategif
import tensorflow as tf
from keras import backend as K



import os
import argparse
parser = argparse.ArgumentParser(description = 'lstm of mnist')
parser.add_argument('--gpu', '-g', help='gpu use?', default='0')
parser.add_argument('--w', '-w', help='weight?', default=1e-1)
parser.add_argument('--b', '-b', help='bias?', default=1e-1)
parser.add_argument('--epoch', '-e', help='epoch?', default=1000)
parser.add_argument('--item', '-i', help='tiem num?', default=1000)
parser.add_argument('--shuffle', '-s', help='shuffle?', default=1)
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
params=[1,2000,1000,500,200,1]
#params=[1,3000,1]

'''
    def RandNormal_w(shape, dtype=None,mean=0,stddev=w):
        if dtype == None:
            dtype = 'float32'
        return K.random_normal(shape, dtype=dtype,mean=mean, stddev=stddev)
    def RandNormal_b(shape, dtype=None,mean=0,stddev=b):
        if dtype == None:
            dtype = 'float32'
        return K.random_normal(shape, dtype=dtype,mean=mean, stddev=stddev)
'''

    

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

class functions(object):
    def __init__(self,x,n,func):
        self.x=x
        self.n=n
        self.func=func
        
    def y(self):
        func_dict={'sum_sin':self.sum_sin(),'sum_sin_norm':self.sum_sin_norm(),
                   'daub':self.daub(),'binary':self.binary(),'one_negone':self.one_negone(),
                   'one':self.one(),'discrete':self.discrete(),'sin_sin':self.sin_sin()}
        return func_dict[self.func]
    
    def one(self):
        return np.zeros(np.shape(self.x))+1
    
    def one_negone(self):
        y_return=np.zeros(np.shape(self.x))+1
        for i in range(np.int(y_return.shape[0]/6)):
            y_return[i]*=-1
        for i in range(np.int(y_return.shape[0]*2/6),np.int(y_return.shape[0]*3/6)):
            y_return[i]*=-1
        for i in range(np.int(y_return.shape[0]*4/6),np.int(y_return.shape[0]*5/6)):
            y_return[i]*=-1
        return y_return
    
    def sum_sin(self):
        result=np.zeros(self.x.shape)
        for k in range(self.n):   
            result+=np.sin(1.5*2*np.pi*(self.x)*(k+1))
        return result 
    
    def sum_sin_norm(self):
        result=np.zeros(self.x.shape)
        for k in range(self.n):   
            result+=np.sin(1.5*2*np.pi*(self.x)*(k+1))/(k+1)
        return result
    
    def daub(self):
        y_output=[]
        x_hat=(self.x+1)*2*np.pi
        for i in x_hat:
            if i<=2*np.pi/3:
                y_output.append(i/2+np.cos(20*i))
            elif i<=5*np.pi/2/3:
                y_output.append(i/2+np.cos(20*i)+np.cos(4/3*((i-10)**3-(2*np.pi-10)**3)+10*(i-2*np.pi)))
            elif i<=4*np.pi/3:
                y_output.append(np.cos(20*i)+np.cos(4/3*((i-10)**3-(2*np.pi-10)**3)+10*(i-2*np.pi)))
            elif i<=2*np.pi/3+4*np.pi/3:
                i-=4*np.pi/3
                y_output.append(i/2+np.cos(20*i))
            elif i<=5*np.pi/2/3+4*np.pi/3:
                i-=4*np.pi/3
                y_output.append(i/2+np.cos(20*i)+np.cos(4/3*((i-10)**3-(2*np.pi-10)**3)+10*(i-2*np.pi)))
            elif i<=4*np.pi/3+4*np.pi/3:
                i-=4*np.pi/3
                y_output.append(np.cos(20*i)+np.cos(4/3*((i-10)**3-(2*np.pi-10)**3)+10*(i-2*np.pi)))
            elif i<=2*np.pi/3+2*4*np.pi/3:
                i-=2*4*np.pi/3
                y_output.append(i/2+np.cos(20*i))
            elif i<=5*np.pi/2/3+2*4*np.pi/3:
                i-=2*4*np.pi/3
                y_output.append(i/2+np.cos(20*i)+np.cos(4/3*((i-10)**3-(2*np.pi-10)**3)+10*(i-2*np.pi)))
            elif i<=4*np.pi/3+2*4*np.pi/3:
                i-=2*4*np.pi/3
                y_output.append(np.cos(20*i)+np.cos(4/3*((i-10)**3-(2*np.pi-10)**3)+10*(i-2*np.pi)))
        y_output=np.array(y_output)
        return y_output.reshape((-1,1))
    
    def binary(self):
        y_return=np.zeros(np.shape(self.x))+1
        for i in range(np.int(y_return.shape[0]/2)):
            y_return[2*i+1]*=-1
        return y_return
    def discrete(self):
        y=np.sin(self.x*10)+np.sin(2*self.x*10)
        return alpha*np.rint(y/alpha)
    
    def sin_sin(self):
        y_return = np.sin(np.sin(self.x*2*np.pi)*np.pi*25)
        #y_return = np.sin(1000*self.x)
        return y_return
        
class sampling(object):
    def __init__(self,sampling_type,border_min,border_max,
                 num,mean=0,std=1e-2,gauss_num=1):
        self.sampling_type=sampling_type
        self.border_min=border_min
        self.border_max=border_max
        self.num=num
        self.mean = mean
        self.std = std
        self.gauss_num = gauss_num
        
    def sampling(self):
        sampling_dict={'average':self.average(),'chebyshev':self.chebyshev(),
                       'random':self.random(),'gauss':self.gauss()}
        x_train=sampling_dict[self.sampling_type]
        x_train=x_train.reshape((self.num,1))
        return np.sort(x_train,axis=0)
        
    def average(self):
        return np.linspace(self.border_min,self.border_max,self.num)
    
    def chebyshev(self):
        x=np.zeros((self.num,1))
        for i in range(self.num):
            x[i]=np.cos((2*i+1)*np.pi/(2*(self.num+1)))
        return (self.border_max-self.border_min)*x/2+(self.border_max+self.border_min)/2
    
    def random(self):
        return np.random.uniform(self.border_min,self.border_max,self.num)
    
    def gauss(self):
        if self.gauss_num==1:
            x_train = np.random.normal(self.mean,self.std,self.num//self.gauss_num)
            x_train = np.sort(x_train,axis=0)
        else:
            x_train=np.zeros((0,1))
            for i in range(self.gauss_num):
                mean_new = -self.mean+i*2*self.mean/(self.gauss_num-1)
                x_train_new = np.random.normal(mean_new,self.std,self.num//self.gauss_num)
                x_train = np.concatenate([x_train,x_train_new.reshape(-1,1)],axis=0)
                x_train = np.clip(x_train,self.border_min,self.border_max)
            x_train = np.sort(x_train,axis=0)
        return x_train

    
    def auto_sampling(f,epsilon=1e-3,step=50,x_min=-1,x_max=1):
        #this function can auto sampling,given a function f,and the minimum descrete second-order derivative will bigger than epsilon.
        #'step'is the minimum added,every time the second-order is lesser than epsilon.
        #x_min,x_max are the filds of the sampling you need of function f.
        x_train=[]
        num=step
        x=np.linspace(x_min,x_max,num)
        y=f(x)
        y_diff=np.abs(np.diff(np.diff(y)))
        while (np.min(y_diff)>epsilon):
            num+=step
            x=np.linspace(x_min,x_max,num)
            y=f(x)
            y_diff=np.abs(np.diff(np.diff(y)))
        diff_num=np.int(np.sqrt(2*num))
        rand_diff=[]
        for i in range(diff_num):
            rand_diff.append([])
        for i in range(num-2):
            tmp=np.int((diff_num-1)*(y_diff[i]-np.min(y_diff))/(np.max(y_diff)-np.min(y_diff)))
            rand_diff[tmp].append(i)
        for i in range(diff_num):
            probability=1/(len(rand_diff[i])+1e-5)
            for j in range(len(rand_diff[i])):
                for k in range(np.int(i**(1))):
                    if np.random.random()<probability:
                        x_train.append(np.random.uniform(x_min+(rand_diff[i][j]-1)*(x_max-x_min)/num,x_min+(rand_diff[i][j])*(x_max-x_min)/num))
        return np.sort(np.array(x_train))      

class tools(object):
    def __init__(self):
        pass
    
    
    def cal_delta(x,y):
        y_delta=copy.copy(y)
        for i in range(100):
            c=np.sum(np.exp(-np.sum((x-x[i,:])**2,axis=1)/2/delta))
            y_delta[i,:]=np.sum(np.expand_dims(np.exp(-np.sum((x-x[i,:])**2,axis=1)/2/delta),axis=1)*y,axis=0)/c
        return y_delta
        
    def fourier(x,y,fre=None):
        pca=PCA(n_components=1)
        pca.fit_transform(x)
        p1=pca.components_
        im=1j
        if fre==None:
            y_hat=np.zeros((num_fre,1))*im
            for k in range(num_fre):
                y_hat[k]=np.sum(np.exp(-im*2*np.pi*np.dot(x,np.transpose(p1))*k)*y,axis=0)/num
        else:
            y_hat=np.sum(np.exp(-im*2*np.pi*np.dot(x,np.transpose(p1))*fre)*y,axis=0)/num
        return y_hat
    
    def Delta_F(self,x,y,T,fre=None):
        return np.abs(self.fourier(x,y,fre)-self.fourier(x,T,fre))/np.abs(self.fourier(x,y,fre))
    
    def discrete(y_0,alpha):
        if alpha==0:
            return y_0
        else:
            return alpha*np.rint(y_0/alpha) 
    
    def my_fft_ori(data):
        
        datat=np.squeeze(data)
        datat_fft = np.fft.fft(datat) 
        return datat_fft
    
    def mse(y_true,y_pred):
        return np.mean(np.square(y_true-y_pred),axis=0)
    
def activation_plot():
    x=np.linspace(-3,3,100)
    y_softplus=np.log(np.exp(x)+1)
    y_linear=x+0
    y_hardsigmoid=x+0
    for i in range(100):
        if x[i]<-2.5:
            y_hardsigmoid[i]=0
        elif x[i]>=2.5:
            y_hardsigmoid[i]=1
        else:
            y_hardsigmoid[i]=0.2*x[i]+0.5
    y_exponential=np.exp(x)
    fig1=plt.gcf()
    plt.plot(x,y_softplus,label='softplus')
    plt.plot(x,y_linear,label='linear')
    plt.plot(x,y_hardsigmoid,label='hardsigmoid')
    plt.plot(x,y_exponential,label='exponential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.show()
    
    y_sigmoid=np.exp(x)/(np.exp(x)+1)
    y_elu=x+0
    for i in range(100):
        if x[i]<0:
            y_elu[i]=x[i]
        else:
            y_elu[i]=np.exp(x[i])-1
    y_tanh=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    fig2=plt.gcf()
    plt.plot(x,y_sigmoid,label='sigmoid')
    plt.plot(x,y_elu,label='elu')
    plt.plot(x,y_tanh,label='tanh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.show()
    
    y_softsign=x/(np.abs(x)+1)
    y_selu=x+0
    for i in range(100):
        if x[i]<0:
            y_selu[i]=x[i]
        else:
            y_selu[i]=1.6732632423543772848170429916717*(np.exp(x[i])-1)
    y_selu*=1.0507009873554804934193349852946
    y_relu=x+0
    for i in range(100):
        y_relu[i]=np.max(x[i],0)
    fig3=plt.gcf()
    plt.plot(x,y_softsign,label='softsign')
    plt.plot(x,y_selu,label='selu')
    plt.plot(x,y_relu,label='relu')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.show()    

def train_plot(sampling_type='average',border_min=-1,border_max=1,num=600,fre_ord=1,batch_size=600,bias=0,up=0,
               epochs=10,ite_num=100,func_name='sum_sin',verbose=1,
               optimizer=keras.optimizers.adam(),loss='mse',activation='relu',
               cbks_name='cbks.h5',w=1e-1,b=1e-1,shuffle=True,
               mean=0,std=1e-2,gauss_num=1):
    samp=sampling(sampling_type,border_min,border_max,num,
                  mean=mean,std=std,gauss_num=gauss_num)
    x_train=samp.sampling()+bias
    func=functions(x_train,fre_ord,func_name)
    y_train=func.y()+up

        
    x_test=np.linspace(border_min,border_max,int(num*1.1))+bias
    x_test=x_test.reshape((int(num*1.1),1))
    func.x=x_test
    y_test=func.y()+up


    
    # build network
    model = build_model(optimizer,loss,activation,w=w,b=b)
    print(model.summary())
    
    
    #cbks=keras.callbacks.EarlyStopping('loss',min_delta=0.001,mode='min',patience=170)
    #cbks=keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.99, patience=10, min_delta=0.007)
    cbks=keras.callbacks.ModelCheckpoint(filepath='./cbks/'+cbks_name, monitor='val_loss',
                                        verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=1) 
    #teacher
    dirs = './pics/w_'+str(w)+'_b_'+str(b)+'/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    loss_value={'0':[],'1':[],'2':[],'all':[]}
    val_loss_value={'0':[],'1':[],'2':[],'all':[]}

    for i in range(ite_num):
        print('start train ite_num:',ite_num)
        model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[],
                      validation_data=(x_test,y_test),
                      shuffle=shuffle,verbose=0)
        print('finish train ite_num:',ite_num)
        for j in range(3):
            x_pre=np.linspace(border_min+(border_max-border_min)*j/3,border_min+(border_max-border_min)*(j+1)/3,np.int(num/3))+bias
            x_pre=x_pre.reshape((-1,1))
            y_pre=model.predict(x_pre)
            func.x=x_pre
            y_true=func.y()+up
            val_loss_value[str(j)].append(tools.mse(y_pre,y_true))
        val_loss_value['all'].append(tools.mse(y_test,model.predict(x_test)))
        
        for j in range(3):
            x_pre=x_train[np.int(j*num/3):np.int((j+1)*num/3)]
            x_pre=x_pre.reshape((-1,1))
            y_pre=model.predict(x_pre)
            func.x=x_pre
            y_true=func.y()+up
            loss_value[str(j)].append(tools.mse(y_pre,y_true))
        loss_value['all'].append(tools.mse(y_train,model.predict(x_train)))
        
        plt.figure(figsize=(12,5))
        fig = plt.gcf()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=None, hspace=1)
        plt.subplot(311)
        plt.plot(x_train,y_train,c='red',label='train')
        plt.plot(x_test,model.predict(x_test),c='black',label='fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('epoch='+str(i))
        plt.legend(loc='upper right')
        
        '''
        #画loss子图
        plt.subplot(312)
        plt.plot(val_loss_value['0'],label='L')
        plt.plot(val_loss_value['1'],label='M')
        plt.plot(val_loss_value['2'],label='R')
        plt.plot(val_loss_value['all'],label='all')
        plt.xscale('symlog')
        plt.ylabel('val_loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        
        plt.subplot(313)
        plt.plot(loss_value['0'],label='L')
        plt.plot(loss_value['1'],label='M')
        plt.plot(loss_value['2'],label='R')
        plt.plot(loss_value['all'],label='all')
        plt.xscale('symlog')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        '''
               
        plt.subplot(312)
        
        #fft_y=np.abs(tools.my_fft_ori(y_test))
        #x_plot=np.arange(0,fft_y.shape[0],1)
        #plt.plot(x_plot,np.abs(tools.my_fft_ori(y_test)[0:1200]))
        '''
        plt.semilogy(np.abs(tools.my_fft_ori(y_test)[0:num//20]+1e-5),label='train')
        
        plt.semilogy(np.abs(tools.my_fft_ori(model.predict(x_test))[0:num//20]+1e-5),label='fit')
        plt.xlabel('f')
        plt.ylabel('A')
        plt.legend(loc='upper right')
        '''
        ft(np.squeeze(x_test),np.squeeze(y_test),x_test.shape[0])
        
        plt.subplot(313)
        ft(np.squeeze(x_test),np.squeeze(model.predict(x_test)),x_test.shape[0])
        #ft_diff(np.squeeze(x_train),np.squeeze(y_train),np.squeeze(model.predict(x_train)),x_train.shape[0])


        
        fig.savefig(dirs+'xf'+str(i)+'.png', dpi=170)
        
        #plt.show()
        plt.cla()
       
    generategif.gif('sin_sin',ite_num,dirs)

def ft(t,data,sampling_rate):    
    wavename = 'cgau8'
    totalscal = 512
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 10, -10)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"frequency(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)

def ft_diff(t,data_real,data_pre,sampling_rate):    
    wavename = 'cgau8'
    totalscal = 512
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 10, -10)
    [cwtmatr_real, frequencies] = pywt.cwt(data_real, scales, wavename, 1.0 / sampling_rate)
    [cwtmatr_pre, frequencies] = pywt.cwt(data_pre, scales, wavename, 1.0 / sampling_rate)
    cwtmatr_plot = (np.exp(abs(cwtmatr_real))-np.exp(abs(cwtmatr_pre)))/(np.exp(abs(cwtmatr_pre)))
    plt.contourf(t, frequencies, abs(cwtmatr_plot))
    plt.ylabel(u"frequency(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.colorbar()
    #plt.show()
    #print(frequencies)       
    
if __name__ == '__main__':
    '''
    #choose a kind of sampling type from:['average','chebyshev','auto_sampling','random','gauss']
    samp_types=['average','chebyshev','random']
    
    for sampling_type in samp_types:
        train_plot(sampling_type,border_min,border_max,num,fre_ord,batch_size)
    '''
    train_plot(sampling_type='gauss',border_min=-1,border_max=1,num=600,fre_ord=3,batch_size=60,bias=0,up=0,
               epochs=int(args.epoch),ite_num=int(args.item),func_name='sin_sin',verbose=0,
               optimizer=keras.optimizers.adam(),loss='mse',activation='relu',cbks_name='complex.h5',
               w=float(str(args.w)),b=float(str(args.b)),shuffle=int(args.shuffle),
               mean=5e-1,std=1e-1,gauss_num=3)
    '''
    from keras.models import load_model
    model=load_model('./cbks/initial_relu.h5')
    weights=model.get_weights()
    x=np.arange(1,5001)
    for i in range(3):
        plt.hist(np.squeeze(weights[i]),100)
    '''
    #plot different activation functions.
    #activation_plot()
 
