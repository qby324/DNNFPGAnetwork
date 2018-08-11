---------------------------------------------------------------------------------------------------------------------------------
debug=1
if debug:
    print("program run")
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import Variable as V
from numpy import array as ar
from chainer.functions import squared_difference as dist
import pickle
import gc
import time

epoch=1000
LP1=1
LP2=1
fixed=1

if debug:
    print("finished import")

class DPNN(chainer.Chain):#calculate the dp between template and speech
    def __init__(self, dim_data):
        super(DPNN, self).__init__()
        with self.init_scope():
            self.nodes = []
            self.dim = dim_data
            self.b=ar([2],dtype=np.float32)
            self.W=ar([[0.1]],dtype=np.float32)
    def __call__(self, template, speech, length_of_template, length_of_speech):
        self.nodes=[]
        for i in range(length_of_template+1):
            for j in range(length_of_speech+1):
                self.nodes.append(V(ar(0.0,dtype=np.float32)))
        for i in range(length_of_template+1):
            #print("("+str(i)+",1)",end=' ')
            for j in range(length_of_speech+1):
                if(i!=0 and j!=0):
                    self.nodes[i*(length_of_speech+1)+j]=F.min(F.stack([self.nodes[(i-1)*(length_of_speech+1)+j],self.nodes[i*(length_of_speech+1)+j-1],self.nodes[(i-1)*(length_of_speech+1)+j-1]]))+F.sqrt(F.sum(dist(template[i-1],speech[j-1]))+1e-8)
        #print(self.nodes[-length_of_speech-1:])
        result_temp=self.nodes[-length_of_speech:]
        result=[]
        t1=[]
        t2=[]
        t3=[]
        for i in range(len(result_temp)):
            t1.append(F.expand_dims(result_temp[i],axis=0))
            t2.append(F.expand_dims(t1[i],axis=0))
            t3.append(F.linear(t2[i],self.W,self.b))
            result.append(F.sigmoid(t3[i]))
        y = F.hstack(result)
        return y[0]

class NN(chainer.Chain):# the whole network
    def __init__(self, dpnn):
        super(NN, self).__init__()
        with self.init_scope():
            self.rnn=L.NStepLSTM(3,12,20,0.5)#first part is RNN
            self.dpnn=dpnn#second part is the DPNN
    def reset_state():
        self.rnn.reset_state()
    def __call__(self, template, speech, samp_rate = 10):
        hy,cy,speech_feature=self.rnn(None,None,[speech])#go through the rnn
        print("speech_feature="+str(speech_feature))
        hy,cy,template_feature=self.rnn(None,None,[template])
        print("RNN finishes")
        template_length,_=template_feature[0].shape
        speech_length,_=speech_feature[0].shape
        sampled_speech_feature=speech_feature[0][2*samp_rate-1::samp_rate]#subsampled
        sampled_speech_length=len(sampled_speech_feature)
        sampled_template_feature=template_feature[0][2*samp_rate-1::samp_rate]
        sampled_template_length=len(sampled_template_feature)
        y=self.dpnn(sampled_template_feature,sampled_speech_feature,sampled_template_length,sampled_speech_length)#calculate dp
        return y

def calF(y,a):
    for i in range(len(y)):
        if y[i]>threoshold :
            y[i]=1
        else:
            y[i]=0
    for i in range(len(a)):

        



    return F
file = np.load("/work/wj_ous/keyword/data/evldata.npz")#read data
speech_data = file['speech']
alignment_data = file['alignment']
template_data = file['template']
newtemplate_data=[]
newalignment_data=[]
newspeech_data=[]
for i in range(800):#use when debug
    newspeech_data.append(template_data[0])
    newtemplate_data.append(template_data[0])
    newalignment_data.append(ar([0,0,0,1],dtype=np.float32))


if debug:
    print("finished data")
#print(type(template))
#print(template.data)
dpnn=DPNN(20)#initialize the model
nn = NN(dpnn)
F=[]
for i in range(len(alignment_data)):
    y=nn(speech_data[i],template_data[i])
    F.append(y,alignment_data[i])
