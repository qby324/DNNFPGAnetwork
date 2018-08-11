#! /usr/bin/python
# -*- coding: utf-8 -*-

# Thu Sep 18 12:15:26 JST 2014
# Extended to handle DAG of nuclei

# Mon Nov  4 08:12:04 JST 2013
# Takahiro Shinozaki
# Read Kaldi format Nnet and run back-propagation


import sys
import re
import random

import numpy
import theano
import theano.tensor as T

import nnet

debug=0

# main function
if __name__ == "__main__":

    if len(sys.argv) < 4:
        print >> sys.stderr, "Usage: bakprop Nnet trainX trainY <opts>"
        print >> sys.stderr, "         trainX, trainY: htk or kaldi's scp file, or kaldi's text ark file of int vectors"
        print >> sys.stderr, "<opts> = [--devX=file] [--devY=file]     # development data X and Y"
        print >> sys.stderr, "         [--mvnf=file]                   # mean and variance file for input featur normalization"
        print >> sys.stderr, "         [--cmvn]                        # compute mean and var, and normalize input data"
        print >> sys.stderr, "         [--xType=(htkscp|arkscp)]       # data type of X"
        print >> sys.stderr, "         [--yType=(htkscp|arkscp|arkiv)] # data type of Y"
        print >> sys.stderr, "         [--ofs=str]                     # stem name for output Nnet files"
        print >> sys.stderr, "         [--bb=int]                      # big batch size (shuffling is applied inside this unit)"
        print >> sys.stderr, "         [--ep=int] [--mb=int]           # epoch, mini batch size"
        print >> sys.stderr, "         [--lr=float] [--rgl=float] [--mm=float] # learning rate, regulalizer, momentum"
        print >> sys.stderr, "         [--spl=int]                     # splice for feature data. (not applied to categorial data)"
        print >> sys.stderr, "         [--trmspl=(True|False)]         # trim spliced framas including zero-padding"
        print >> sys.stderr, "         [--sd=int]                      # random seed"
        print >> sys.stderr, "         [--sh=(True|False)]             # shuffle data"
        print >> sys.stderr, "         [--ot=(c|f)]                    # output layer is category or feature vector"
        print >> sys.stderr, "                                         #  ([default] c : arkiv, f: htkscp, arkscp)"
        sys.exit(1)

    nnetfile   = sys.argv[1]
    trainXfile = sys.argv[2]
    trainYfile = sys.argv[3]
    devXfile   = ""
    devYfile   = ""
    mvnormf    = ""
    cmvnflg     = False # compute man and var, and normalize input data
    xType      = "htkscp"
    yType      = "htkscp"
    outfstem   = ""
    bbsize     = 0      # big batch size (0 means all data)
    epoch      = 5      # number of epochs
    mbsize     = 128    # mini batch size
    lr         = 1.0    # learning rate
    mm         = 0.0    # momentum
    rgl        = 0.0    # regulalizer
    spl        = 0      # splice (add left/right spl frames to form a composit frame)
    trmspl     = True  # trim frames including zero-padding for splice
    seed       = 1      # 乱数シード
    shflg      = True   # shuffle flag for train data
    ot         = ""     # output type. c:category, f:feature

    for ag in sys.argv[4:]:
        if re.match("--ofs=", ag):
             outfstem = re.sub("--ofs=", "", ag)
        elif re.match("--devX=", ag):
            devXfile = re.sub("--devX=", "", ag)
        elif re.match("--devY=", ag):
            devYfile = re.sub("--devY=", "", ag)
        elif re.match("--mvnf=", ag):
            mvnormf = re.sub("--mvnf=", "", ag)
        elif re.match("--cmvn=", ag):
            if not re.sub("--cmvn=", "", ag) in ["True", "true", "TRUE", "False", "false", "FALSE"]:
                print >> sys.stderr, "Error: invalid arument for cmvn. Expecting True or False."
                sys.exit(1)
            cmvnflg = re.sub("--cmvn=", "", ag) in ["True", "true", "TRUE"]
        elif re.match("--xType=", ag):
            xType = re.sub("--xType=", "", ag)
        elif re.match("--yType=", ag):
            yType = re.sub("--yType=", "", ag)
        elif re.match("--bb=", ag):
            bbsize = int(re.sub("--bb=", "", ag))
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
        elif re.match("--spl=", ag):
            spl = int(re.sub("--spl=", "", ag))
        elif re.match("--trmspl=", ag):
            trmspl = re.sub("--trmspl=", "", ag) in ["True", "true", "TRUE"]
        elif re.match("--sd=", ag):
            seed = int(re.sub("--sd=", "", ag))
        elif re.match("--sh=", ag):
            shflg = re.sub("--sh=", "", ag) in ["True", "true", "TRUE"]
        elif re.match("--ot", ag):
            ot = re.sub("--ot=", "", ag)
        else:
            print >> sys.stderr, "Error: Unknown option: " + ag
            sys.exit(1)

    if ot=="":
        if yType in ["arkiv"]:
            ot="c"
        else:
            ot="f"

    # check option requirements
    if (devXfile or devYfile) and not (devXfile and devYfile):
        print >> sys.stderr, "Error: devXfile and devYfile must be specified together"
        sys.exit(1)

    # initialize random seed
    random.seed(seed)


    # load kaldi format Nnet
    nn = nnet.loadnnet(nnetfile) 
    nn.connectchk()
    nn.networkeffectsize()
    if debug:
        nn.debuginfo()
        
    # Symbolic definition of transformation by Nnet
    x      = T.fmatrix("x") # fmatrix is float32
    dnum   = T.dscalar("dnum")
    nnout  = nn.forward(x)

    # Cost definition
    if (ot == "f"):
        y      = T.fmatrix("y")
        cost   = T.sum((nnout-y)*(nnout-y))/dnum
    elif (ot == "c"):
        y      = T.ivector("y") # ivector is int32
        cost   = T.sum(T.nnet.categorical_crossentropy(nnout, y))/dnum

    # List of parameter symbols
    params = nn.params()
    diff   = nn.diffParams()
    # Symbolic computation of gradients
    grads  = T.grad(cost, params)

    # Get lists of symbolic equations for parameter update
    updatesDiff = []
    newparams = []
    for i in range(len(grads)):
        # symbolic equation to update diff[i]
        updatesDiff = updatesDiff + [(diff[i], -lr*grads[i] - rgl*params[i] + mm*diff[i])]
        # symbolic equation to update params[i]
        newparams = newparams + [(params[i], params[i] + diff[i])]

    # compile the functions for update
    trainer_diff = theano.function(
        inputs=[x,y,dnum], outputs=None, updates=updatesDiff)
    trainer_update = theano.function(
        inputs=[], outputs=None, updates=newparams)

    # for evaluation
    if (ot == "f"):
        evalscr = theano.function(inputs=[x,y,dnum], outputs=cost)
    elif (ot == "c"):
        err = T.sum(T.neq(T.argmax(nnout,axis=1),y))/dnum
        evalscr = theano.function(inputs=[x,y,dnum], outputs=err*100)  # error rate

    # prepare training data
    if mvnormf != "":
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

    if xType in ("htkscp", "arkscp"):
        trainXLst = nnet.loadscpf(trainXfile)
        trainXall = numpy.array([], dtype=theano.config.floatX)
    else:
        print >> sys.stderr, "Right now, only htkscp and arkscp are supported as xType."
        sys.exit(1)
#        trainXLst = dict()
#        trainXall = nnet.loadCFile(trainXfile, xType)
        
    if yType in ("htkscp", "arkscp"):
        trainYLst = nnet.loadscpf(trainYfile)
        trainYall = numpy.array([], dtype=theano.config.floatX)
    else:
        trainYLst = dict()
        trainYall = nnet.loadCFile(trainYfile, yType)

    # prepare development data
    if devXfile:
        if xType in ("htkscp", "arkscp"):
            devXLst = nnet.loadscpf(devXfile)
            devXall = numpy.array([], dtype=theano.config.floatX)
        else:
            print >> sys.stderr, "Right now, only htkscp and arkscp are supported as xType."
            sys.exit(1)
#            devXLst = dict()
#            devXall = nnet.loadCFile(devXfile, xType)
        
        if yType in ("htkscp", "arkscp"):
            devYLst = nnet.loadscpf(devYfile)
            devYall = numpy.array([], dtype=theano.config.floatX)
        else:
            devYLst = dict()
            devYall = nnet.loadCFile(devYfile, yType)
        (devXDat, devYDat) = nnet.prepXY(devXLst, devYLst, devXall, devYall, mnstd, xType, yType, 0, 0, spl, trmspl, False)

    else:
        devXLst = dict()
        devXall = numpy.array([], dtype=theano.config.floatX)
        devYLst = dict()
        devYall = numpy.array([], dtype=theano.config.floatX)


    sys.stdout.flush()
    print "Start training (minibatch size=" + str(mbsize) + ")"
    print "Epoch Train Devel"

    # Initial evaluation by training set
    trainScr=0.0
    totnumsmp=0
    bblot=0
    rttot=0
    cttot=0
    while True:
        (trainXDat, trainYDat) = nnet.prepXY(trainXLst, trainYLst, trainXall, trainYall, mnstd, xType, yType, bbsize, bblot, spl, trmspl, shflg)
        bblot+=1
        if len(trainXDat)==0:
            break
        totnumsmp+=len(trainXDat)
        (rt, ct) = trainXDat.shape
        rttot += rt
        cttot += ct

        mbnum=len(trainXDat)/mbsize
        if mbnum*mbsize < len(trainXDat):
            mbnum+=1
        for b in range(mbnum):
            segs=mbsize*b
            sege=mbsize*(b+1)
            dlen = len(trainXDat[segs:sege])
            trainScr += evalscr(trainXDat[segs:sege], trainYDat[segs:sege], dlen) * dlen
    trainScr/=totnumsmp
    print "Training data X:  (" + str(rttot) + ", " + str(cttot) + ")"

    # Initial evaluation by development set
    devScr=0.0
    mbnumdev=len(devXDat)/mbsize
    (rdtot, cdtot) = devXDat.shape
    if mbnumdev*mbsize < len(devXDat):
        mbnumdev+=1
    for b in range(mbnumdev):
        dlen = len(devXDat[mbsize*b:mbsize*(b+1)])
        segs=mbsize*b
        sege=mbsize*(b+1)
        devScr += evalscr(devXDat[segs:sege], devYDat[segs:sege], dlen) * dlen
    devScr/=len(devXDat)
    print "Dev data X:  (" + str(rdtot) + ", " + str(cdtot) + ")"

    # Show initial scores
    print str(0) + ": " + str(trainScr) + " " + str(devScr)
    sys.stdout.flush()

    # Loop for update
    for e in range(epoch):
        trainScr=0.0
        totnumsmp=0
        bblot=0
        while True:
            (trainXDat, trainYDat) = nnet.prepXY(trainXLst, trainYLst, trainXall, trainYall, mnstd, 
                                                 xType, yType, bbsize, bblot, spl, trmspl, shflg)
            bblot+=1
            if len(trainXDat)==0:
                break
            totnumsmp+=len(trainXDat)

            # Mini batch update
            mbnum=len(trainXDat)/mbsize
            if mbnum*mbsize < len(trainXDat):
                mbnum+=1
            for b in range(mbnum):
                segs=mbsize*b
                sege=mbsize*(b+1)
                dlen = len(trainXDat[segs:sege])
                trainer_diff(trainXDat[segs:sege], trainYDat[segs:sege], dlen)
                trainer_update()
                trainScr += evalscr(trainXDat[segs:sege], trainYDat[segs:sege], dlen) * dlen
        trainScr/=totnumsmp

        # Evaluation by development set
        devScr = 0.0;
        for b in range(mbnumdev):
            segs=mbsize*b
            sege=mbsize*(b+1)
            dlen = len(devXDat[segs:sege])
            devScr += evalscr(devXDat[segs:sege], devYDat[segs:sege], dlen) * dlen
        devScr/=len(devXDat)

        # Show progress
        print str(e+1) + ": " + str(trainScr) + " " + str(devScr)
        sys.stdout.flush()

        # Output Nnet file
        if outfstem:
            f = open(outfstem + "." + str(e+1), "w")
            nn.writeParams(f)
            f.close()

