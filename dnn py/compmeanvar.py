#! /usr/bin/python
# -*- coding: utf-8 -*-

# Compute mean and var for normalization
# shinot
# Sat Jan  3 10:49:55 JST 2015

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
    print >> sys.stderr, "Usage: compmeanvar.py datafile outf <opts>"
    print >> sys.stderr, "       datafile: htk or kaldi's scp file"
    print >> sys.stderr, "<opts> = [--xType=(htkscp|arkscp)]               # data type"
    print >> sys.stderr, "         [--bb=int]                              # big batch size"
    sys.exit(1)

datfilename = sys.argv[1]          # 学習用入力データファイル名
outfname    = sys.argv[2]     # 出力ファイル名

xType       = "htkscp"
bbsize      = 0                    # big batch size (0 means all data)

for ag in sys.argv[3:]:
    if re.match("--xType=", ag):
        xType = re.sub("--xType=", "", ag)
    elif re.match("--bb=", ag):
        bbsize = int(re.sub("--bb=", "", ag))
    else:
        print >> sys.stderr, "Error: Unknown option: " + ag
        sys.exit(1)

# prepare training data
if xType in ("htkscp", "arkscp"):
    trainXLst = nnet.loadscpf(datfilename)      #读取文件内容
    trainXall = numpy.array([], dtype=theano.config.floatX)     #将trainXall变成需求的格式
else:
    print >> sys.stderr, "Right now, only htkscp and arkscp are supported as xType."
    sys.exit(1)

bblot=0
ntot=0
while True:
    data = nnet.prepX(trainXLst, trainXall, numpy.array([]), 
                      xType, bbsize, bblot, 0, 0, False)
    bblot+=1
    if len(data)==0:
        break
    if ntot==0:
        xsum = sum(data)
        xxsum = sum(data**2)
        ntot += len(data)
        dim = data.shape[1]
    else:
        xsum += sum(data)
        xxsum += sum(data**2)
        ntot += len(data)
nmmean = xsum/ntot
nmvar = xxsum/ntot-(nmmean)**2

    
# 正規化定数の書き出し
f = open(outfname, "w")
f.write("<DATA> " + datfilename + "\n")
f.write("<MEAN> " + str(dim) + "\n")
f.write(re.sub(",", "", re.sub("\]", "", re.sub("\[", " ", re.sub("\],", "\n", str(nmmean.tolist()))))))
f.write("\n")
f.write("<VARIANCE> " + str(dim) + "\n")
f.write(re.sub(",", "", re.sub("\]", "", re.sub("\[", " ", re.sub("\],", "\n", str(nmvar.tolist()))))))
f.close()
