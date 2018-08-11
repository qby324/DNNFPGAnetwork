#! /usr/bin/python
# -*- coding: utf-8 -*-

# Thu Sep 18 12:15:26 JST 2014
# Extended to handle DAG of nuclei

# Support Kaldi's Nnet format (text)
# shinot
# Wed Nov 27 18:42:38 JST 2013

import sys
import re
import random
import math

import numpy
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import nnet

debug=0

if len(sys.argv) < 3:
    print >> sys.stderr, "Usage: rbmtrain.py datafile hidnum <opts>"
    print >> sys.stderr, "       datafile: htk or kaldi's scp file"
    print >> sys.stderr, "<opts> = [--of=str]                              # name of output Nnet file of RBM"
    print >> sys.stderr, "         [--feadim=dim]                          # input feature dimension"
    print >> sys.stderr, "         [--mvnf=file]                           # mean and variance file for input featur normalization"
    print >> sys.stderr, "         [--cmvn]                                # compute mean and var, and normalize input data"
    print >> sys.stderr, "         [--xType=(htkscp|arkscp)]               # data type"
    print >> sys.stderr, "         [--bb=int]                              # big batch size (shuffling is applied inside this unit)"
    print >> sys.stderr, "         [--spl=int]                             # splice for feature data. (not applied to categorial data)"
    print >> sys.stderr, "         [--trmspl=(True|False)]                 # trim spliced framas including zero-padding"
    print >> sys.stderr, "         [--sh=(True|False)]                     # shuffle data"
    print >> sys.stderr, "         [--wt=str]                              # name of output Nnet file of transformation file"
    print >> sys.stderr, "         [--at=str]                              # activation type for appended nucleus written in transformation file"
    print >> sys.stderr, "         [--ep=int] [--mb=int]                   # epoch, mini batch size"
    print >> sys.stderr, "         [--lr=float] [--rgl=float] [--mm=float] # learning rate, regulalizer, momentum"
    print >> sys.stderr, "         [--sd=int]                              # random seed"
    print >> sys.stderr, "         [--rt=(bb|gb)]                          # bb:bern-bern, gb:gauss-bern"
    print >> sys.stderr, "         [--ft=nnetfile]                         # nnet file for feature transform"
    print >> sys.stderr, "         [--pn=01vec]                            # binary vector expressing input nuclei"
    sys.exit(1)

datfilename = sys.argv[1]          # 学習用入力データファイル名
hidnum      = int(sys.argv[2])     # 隠れ層のニューロン数

rbmfilename = ""                   # 出力 RBM 定義ファイル名
feadim      = 0
outtrnfname = ""                   # 出力 transformation 定義ファイル名
mvnormf     = ""                   # mean and variance for input normalization
cmvnflg     = False               # compute man and var, and normalize input data
xType       = "htkscp"
bbsize      = 0                    # big batch size (0 means all data)
spl         = 0                    # splice (add left/right spl frames to form a composit frame)
trmspl      = True                # trim frames including zero-padding for splice
shflg       = True                 # shuffle flag for train data
acttype     = "sigmoid"            # 出力 transformation に書き込む活性化関数のタイプ
seed        = 1                    # 乱数シード
epoch       = 5                    # エポック数
mbsize      = 128                  # ミニバッチサイズ
lr          = 1.0                  # 学習係数
mm          = 0.0                  # 慣性項
rgl         = 0.0                  # 正則化係数
rbmtype     = "bb"                 # RBM type
ftfilename  = ""                   # nnet file for feature transform
prevnuclbv  = []                   # binary vector expressing input nuclei

for ag in sys.argv[3:]:
    if re.match("--of=", ag):
        rbmfilename = re.sub("--of=", "", ag)
    elif re.match("--feadim=", ag):
        feadim = int(re.sub("--feadim=", "", ag))
    elif re.match("--mvnf=", ag):
        mvnormf = re.sub("--mvnf=", "", ag)
    elif re.match("--cmvn=", ag):
        if not re.sub("--cmvn=", "", ag) in ["True", "true", "TRUE", "False", "false", "FALSE"]:
            print >> sys.stderr, "Error: invalid arument for cmvn. Expecting True or False."
            sys.exit(1)
        cmvnflg = re.sub("--cmvn=", "", ag) in ["True", "true", "TRUE"]
    elif re.match("--xType=", ag):
        xType = re.sub("--xType=", "", ag)
    elif re.match("--bb=", ag):
        bbsize = int(re.sub("--bb=", "", ag))
    elif re.match("--spl=", ag):
        spl = int(re.sub("--spl=", "", ag))
    elif re.match("--trmspl=", ag):
        trmspl = re.sub("--trmspl=", "", ag) in ["True", "true", "TRUE"]
    elif re.match("--sh=", ag):
        if not re.sub("--sh=", "", ag) in ["True", "true", "TRUE", "False", "false", "FALSE"]:
            print >> sys.stderr, "Error: invalid arument for sh. Expecting True or False."
            sys.exit(1)
        shflg = re.sub("--sh=", "", ag) in ["True", "true", "TRUE"]
    elif re.match("--wt=", ag):
        outtrnfname = re.sub("--wt=", "", ag)
    elif re.match("--at=", ag):
        acttype = re.sub("--at=", "", ag)
    elif re.match("--sd=", ag):
        seed = int(re.sub("--sd=", "", ag))
    elif re.match("--ep=", ag):
        epoch = int(re.sub("--ep=", "", ag))
    elif re.match("--mb=", ag):
        mbsize = int(re.sub("--mb=", "", ag))
    elif re.match("--lr=", ag):
        lr = float(re.sub("--lr=", "", ag))
    elif re.match("--mm=", ag):
        mm = float(re.sub("--mm=", "", ag))
    elif re.match("--rgl=", ag):
        rgl = float(re.sub("--rgl=", "", ag))
    elif re.match("--rt=", ag):
        rbmtype = re.sub("--rt=", "", ag)
    elif re.match("--ft", ag):
        ftfilename = re.sub("--ft=", "", ag)
    elif re.match("--pn", ag):
        prevnuclbv = map(int, re.sub("--pn=", "", ag))
    else:
        print >> sys.stderr, "Error: Unknown option: " + ag
        sys.exit(1)

if (rbmtype != "bb" and rbmtype != "gb"):
    print >> sys.stderr, "Error (rbmtrain): rbm type " + rbmtype + " is not supported  yet"
    sys.exit(1)

if hidnum < 1:
    print >> sys.stderr, "Error (rbmtrain): hidnum must be equal to or larger than 1. hidnum=" + str(hidnum)
    sys.exit(1)
    
# initialize random seed
random.seed(seed)

# prepare training data
if xType in ("htkscp", "arkscp"):
    trainXLst = nnet.loadscpf(datfilename)
    trainXall = numpy.array([], dtype=theano.config.floatX)
else:
    print >> sys.stderr, "Right now, only htkscp and arkscp are supported as xType."
    sys.exit(1)


# feature transform
connectflg=0
x = T.fmatrix("x") # fmatrix is float32
if ftfilename:
    # load kaldi format Nnet
    nn = nnet.loadnnet(ftfilename) 
    if debug:
        nn.debuginfo()
    if prevnuclbv:
        tmp = []
        prevnuclids = []
        visnum=0
        for i in range(len(prevnuclbv)):
            if prevnuclbv[i] == 1 and nn.nuclei[i].connectchk()==1:
                tmp+=[nn.nuclei[i].forwardChain(x)]
                prevnuclids += [i]
                visnum+=nn.nuclei[i].nnNnum()
                connectflg=1
        if connectflg==1:
            nnout = T.concatenate(tmp, axis=1)
    else:
        if 0<len(nn.nuclei):
            i=len(nn.nuclei)-1
            if nn.nuclei[i].connectchk()==1:
                prevnuclids = [i]
                connectflg=1
                nnout = nn.forward(x)
                visnum = nn.nuclei[i].nnNnum()
        else:
            prevnuclids = []
            connectflg=1
            nnout = x
            if feadim==0:
                print >> sys.stderr, "Feature dimension must be specified."
                sys.exit(1)
            visnum = feadim
else:
    prevnuclids = []
    connectflg=1
    nnout = x
    if feadim==0:
        print >> sys.stderr, "Feature dimension must be specified."
        sys.exit(1)
    visnum = feadim

if connectflg!=1:
    print "No connection from input. Skip parameter estimation."
    visnum = 0
    W=theano.shared(
        value=numpy.asarray([[]], dtype=theano.config.floatX),
        name="W")
    HBias=theano.shared(
        value=numpy.zeros(hidnum, dtype=theano.config.floatX),
        name="HBias")
    VBias=theano.shared(
        value=numpy.zeros(visnum, dtype=theano.config.floatX),
        name="VBias")
else:
    print "visnum: " + str(visnum)

    featrans = theano.function(inputs=[x], outputs=nnout)

    # initialize RBM parameters
    W=theano.shared(
        value=numpy.asarray(numpy.random.RandomState(seed).uniform(
            low =-4.0*numpy.sqrt(6.0/(hidnum+visnum)),
            high= 4.0*numpy.sqrt(6.0/(hidnum+visnum)),
            size=(visnum, hidnum)), dtype=theano.config.floatX),
        name="W")

    HBias=theano.shared(
        value=numpy.zeros(hidnum, dtype=theano.config.floatX),
        name="HBias")

    VBias=theano.shared(
        value=numpy.zeros(visnum, dtype=theano.config.floatX),
        name="VBias")

    # RBM parameter update by contrastive divergence
    gibbs_rng=RandomStreams(
        numpy.random.RandomState(seed).randint(2**30))

    v0act=T.fmatrix("v0act")
    h0act=T.nnet.sigmoid(T.dot(v0act, W) + HBias)
    h0smp=gibbs_rng.binomial(
        size=(mbsize,hidnum),n=1,p=h0act,dtype=theano.config.floatX)
    if rbmtype=="gb":
        v1act=T.dot(h0smp, W.T) + VBias
    elif rbmtype=="bb":
        v1act=T.nnet.sigmoid(T.dot(h0smp, W.T) + VBias)
    h1act=T.nnet.sigmoid(T.dot(v1act, W) + HBias)

    grad_W    =(T.dot(v1act.T,h1act)-T.dot(v0act.T,h0act))/mbsize
    grad_HBias=(T.sum(h1act,axis=0) -T.sum(h0act,axis=0) )/mbsize
    grad_VBias=(T.sum(v1act,axis=0) -T.sum(v0act,axis=0) )/mbsize

    # 学習用 Python 関数の作成

    DeltaW=theano.shared(
        value=numpy.zeros(visnum*hidnum,
                      dtype=theano.config.floatX).reshape(visnum, hidnum),
        name="DeltaW"    )

    DeltaHBias=theano.shared(
        value=numpy.zeros(hidnum, dtype=theano.config.floatX),
        name="DeltaHBias")

    DeltaVBias=theano.shared(
        value=numpy.zeros(visnum, dtype=theano.config.floatX),
        name="DeltaVBias")

    updates_diff=[
        (DeltaW    , -lr*grad_W    +mm*DeltaW   -rgl*W),
        (DeltaHBias, -lr*grad_HBias+mm*DeltaHBias     ),
        (DeltaVBias, -lr*grad_VBias+mm*DeltaVBias     )]

    updates_update=[
        (W    , W    +DeltaW    ),
        (HBias, HBias+DeltaHBias),
        (VBias, VBias+DeltaVBias)]

    mse=T.mean((v0act-v1act)**2)

    trainer_diff  =theano.function(inputs=[v0act],
                                   outputs=mse,
                                   updates=updates_diff)

    trainer_update=theano.function(inputs=[],
                                   outputs=None,
                                   updates=updates_update)


    if mvnormf != "":
        print mvnormf
        mnstd=nnet.loadMVFile(mvnormf)
    else:
        mnstd=numpy.array([])

    if cmvnflg:
        if mvnormf != "":
            print >> sys.stderr, "Warn: computing mean and var on training set. Normalization file is ignored : " + mvnormf
        bblot=0
        ntot=0
        while True:
            data = nnet.prepX(trainXLst, trainXall, numpy.array([]), 
                              xType, bbsize, bblot, 0, trmspl, shflg)
            bblot+=1
            if len(data)==0:
                break
            if ntot==0:
                xsum = sum(data)
                xxsum = sum(data**2)
                ntot += len(data)
            else:
                xsum += sum(data)
                xxsum += sum(data**2)
                ntot += len(data)
        mnstd=numpy.array([xsum/ntot, numpy.sqrt(xxsum/ntot-(xsum/ntot)**2)], dtype="float32")

    # パラメータの推定処理
    for e in range(epoch):
        err=0.0
        bblot=0
        totmbnum=0
        while True:
            data = nnet.prepX(trainXLst, trainXall, mnstd, 
                         xType, bbsize, bblot, spl, trmspl, shflg)
            bblot+=1
            if len(data)==0:
                break
            mbnum=data.shape[0]/mbsize
            totmbnum+=mbnum
#            numpy.random.RandomState(seed).shuffle(data)

            # ミニバッチによる推定
            for b in range(mbnum):
                err+=trainer_diff(featrans(data[mbsize*b:mbsize*(b+1)]))
                trainer_update()
        err/=totmbnum
        print err
    
# モデルパラメータの書き出し
if rbmfilename != "":
    f = open(rbmfilename, "w")
    f.write("<Nnet>\n")
    f.write("<rbm> " + str(hidnum) + " " + str(visnum) + "\n")
    if rbmtype=="gb":
        f.write("gauss bern [\n")
    elif rbmtype=="bb":
        f.write("bern bern [\n")
    f.write(re.sub(",", "", re.sub("\]", "", re.sub("\[", " ", re.sub("\],", "\n", str(W.get_value().T.tolist()))))))
    f.write(" ]\n")
    f.write(re.sub("\]", " ]\n", re.sub("\[", " [ ", re.sub(",", "", str(VBias.get_value().tolist())))))
    f.write(re.sub("\]", " ]\n", re.sub("\[", " [ ", re.sub(",", "", str(HBias.get_value().tolist())))))
    f.write("</Nnet>\n")
    f.close()

if outtrnfname != "":
    nn.appendNucleus(nnet.AffineNucleus(len(nn.nuclei), prevnuclids, W.get_value(), HBias.get_value()))
    if acttype=="sigmoid":
        nn.appendNucleus(nnet.SigmoidNucleus(len(nn.nuclei), [len(nn.nuclei)-1], hidnum))
    elif acttype=="softmax":
        nn.appendNucleus(nnet.SoftmaxNucleus(len(nn.nuclei), [len(nn.nuclei)-1], hidnum))
    else:
        print >> sys.stderr, "Error: activation " + acttype + " is not supported"
        sys.exit(1)
    f = open(outtrnfname, "w")
    nn.writeParams(f)
    f.close()
    

