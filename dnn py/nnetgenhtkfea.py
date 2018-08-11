#! /usr/bin/python
# -*- coding: utf-8 -*-

# Sun Sep 28 11:45:34 JST 2014
# Takahiro Shinozaki
# Apply forward computation by Nnet and output the results in HTK format



import sys
import re
import struct

import numpy
import theano
import theano.tensor as T

import nnet

debug=0

# main function
if __name__ == "__main__":

    if len(sys.argv) < 4:
        print >> sys.stderr, "Error: nnetgenhtkfea.py Nnet datafile outdir <opts>"
        print >> sys.stderr, "       datafile: htk or kaldi's scp file"
        print >> sys.stderr, "<opts> = [--mb=int]                         # mini batch size"
        print >> sys.stderr, "         [--mvnf=file]                      # mean and variance file for input featur normalization"
        print >> sys.stderr, "         [--spl=int]                        # splice"
        print >> sys.stderr, "         [--trmspl=(True|False)]            # trim spliced framas including zero-padding"
        print >> sys.stderr, "         [--xType=(htkscp|arkscp)]          # data type"
        print >> sys.stderr, "         [--smpprd=int]                     # sample period in 100ns units"
        print >> sys.stderr, "         [--pmkind=int]                     # parameter kind"
        sys.exit(1)

    nnetfile = sys.argv[1]
    datfile  = sys.argv[2]
    outdir   = sys.argv[3]
    mbsize     = 128    # mini batch size
    mvnormf    = ""
    spl        = 0      # splice (add left/right spl frames to form a composit frame)
    trmspl     = True   # trim frames including zero-padding for splice
    xType      = "htkscp"
    smpprd     = 100000 # 10ms
    pmkind     = 9      # in HTK, 9 mean user defined feature

    for ag in sys.argv[4:]:
        if re.match("--mb=", ag):
            mbsize = int(re.sub("--mb=", "", ag))
        elif re.match("--spl=", ag):
            spl = int(re.sub("--spl=", "", ag))
        elif re.match("--mvnf=", ag):
            mvnormf = re.sub("--mvnf=", "", ag)
        elif re.match("--trmspl=", ag):
            trmspl = re.sub("--trmspl=", "", ag) in ["True", "true", "TRUE"]
        elif re.match("--xType=", ag):
            xType = re.sub("--xType=", "", ag)
        elif re.match("--smpprd", ag):
            smpprd = int(re.sub("--smpprd=", "", ag))
        elif re.match("--pmkind", ag):
            pmkind = int(re.sub("--pmkind=", "", ag))
        else:
            print >> sys.stderr, "Error: Unknown option: " + ag
            sys.exit(1)

    # load kaldi format Nnet
    nn = nnet.loadnnet(nnetfile) 
    nn.connectchk()
    if debug:
        nn.debuginfo()
        
    # Symbolic definition of transformation by Nnet
    x      = T.fmatrix("x") # fmatrix is float32
    nnout  = nn.forward(x)
    featrans = theano.function(inputs=[x], outputs=nnout)

    # prepare training data
    if mvnormf != "":
        mnstd=nnet.loadMVFile(mvnormf)
    else:
        mnstd=numpy.array([])

    if xType in ("htkscp", "arkscp"):
        trainXLst = nnet.loadscpf(datfile)
        trainXall = numpy.array([], dtype=theano.config.floatX)
    else:
        print >> sys.stderr, "Right now, only htkscp and arkscp are supported as xType."
        sys.exit(1)

    idlist = nnet.scp2idlist(trainXLst, xType)
    for ut in range(len(idlist)):
        # set bb size to 1 and process one utterance by one utterance.
        # no shuffling.
        data = nnet.prepX(trainXLst, trainXall, mnstd, 
                          xType, 1, ut, spl, trmspl, False)

        mbnum=len(data)/mbsize
        if mbnum*mbsize < len(data):
            mbnum+=1
        outdlst = []
        for b in range(mbnum):
            segs=mbsize*b
            sege=mbsize*(b+1)
            outdlst.append(featrans(data[segs:sege]))
        outdata = numpy.concatenate(outdlst)

        stem=idlist[ut]
        outf=outdir + "/" + stem + ".htk"
        fout = open(outf, "wb")
        fout.write(struct.pack(">i i h h", len(outdata), smpprd, nn.nnNnum()*4, pmkind))
        fout.write(struct.pack(">"+str(outdata.size)+"f", *numpy.concatenate(outdata).tolist()))
        fout.close()


