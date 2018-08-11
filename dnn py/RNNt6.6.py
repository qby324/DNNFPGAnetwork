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
class DPNN(chainer.Chain):
    def __init__(self, dim_data):
        super(DPNN, self).__init__()
        with self.init_scope():
            self.nodes = []
            self.dim = dim_data
            self.b=ar([2],dtype=np.float32)
            self.W=ar([[0.001]],dtype=np.float32)
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
            t3.append(F.linear(t2[i],W,b))
            result.append(F.sigmoid(t3[i]))
        y = F.hstack(result)
        return y[0]

class NN(chainer.Chain):
    def __init__(self, dpnn):
        super(NN, self).__init__()
        with self.init_scope():
            self.rnn=L.NStepLSTM(3,12,20,0.5)
            self.dpnn=dpnn
    def reset_state():
        self.rnn.reset_state()
    def __call__(self, template, speech, samp_rate = 10):
        hy,cy,speech_feature=self.rnn(None,None,[speech])
        hy,cy,template_feature=self.rnn(None,None,[template])
        print("RNN finishes")
        template_length,_=template_feature[0].shape
        speech_length,_=speech_feature[0].shape
        sampled_speech_feature=speech_feature[0][2*samp_rate-1::samp_rate]
        sampled_speech_length=len(sampled_speech_feature)
        sampled_template_feature=template_feature[0][2*samp_rate-1::samp_rate]
        sampled_template_length=len(sampled_template_feature)
        y=self.dpnn(sampled_template_feature,sampled_speech_feature,sampled_template_length,sampled_speech_length)
        return y

def compute_loss(y, t):
    loss=0.0
    print(y)
    loss+=LP1*F.sum(F.absolute(y-t))
    loss/=len(y)
    loss+=LP2*F.max(F.absolute(y-t))
    return loss

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
    def __call__(self, x1, x2, t):
        loss=[]
        for i in range(len(x1)):
            y = self.predictor(x1[i],x2[i])
            loss.append(compute_loss(y, t[i]))
        loss_n=F.hstack(loss)
        loss_na=F.average(loss_n)
        #accuracy = F.accuracy(y, t)
        print("loss= "+str(loss_na))
        report({'loss': loss_na}, self)
        return loss_na

if debug:
    print("finished model")

file = np.load("./data/80data1_seg_sp20.npz")
speech_data = file['speech']
alignment_data = file['alignment']
template_data = file['template']
newtemplate_data=[]
newalignment_data=[]
newspeech_data=[]
for i in range(800):
    newspeech_data.append(template_data[0])
    newtemplate_data.append(template_data[0])
    newalignment_data.append(ar([0,0,0,1],dtype=np.float32))


if debug:
    print("finished data")
#print(type(template))
#print(template.data)
dpnn=DPNN(20)
nn = NN(dpnn)
model = Classifier(nn)
optimizer = optimizers.Adam(0.005)
optimizer.setup(model)
train=chainer.datasets.TupleDataset(template_data,speech_data,alignment_data)
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
model.cleargrads()

for i in range(epoch):
    it=train_iter.next()
    x1=[]
    x2=[]
    t=[]
    for j in range(len(it)):
        x1.append(it[j][0])
        x2.append(it[j][1])
        t.append(it[j][2])
    loss=model(x1,x2,t)
    loss.backward()
    optimizer.update()
    if fixed:
        model.predictor.dpnn.l.b.data=ar([2],dtype=np.float32)
        model.predictor.dpnn.l.W.data=ar([[0.001]],dtype=np.float32)
    gc.collect()

if debug:
    print("prepare train")

with open('file6.6.pkl','wb') as f:
	pickle.dump(nn.rnn,f)

