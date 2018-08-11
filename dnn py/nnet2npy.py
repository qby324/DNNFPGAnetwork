#! /usr/bin/python
# -*- coding: utf-8 -*-

# Wed Dec 31 10:33:18 JST 2014
# Takahiro Shinozaki
# Read Kaldi (text) format Nnet and export as numpy matrix


import sys
import re
import random

import numpy
import theano
import theano.tensor as T

import nnet

# main function
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print >> sys.stderr, "Error: nnet2npy Nnet outd <opts>"
        print >> sys.stderr, "<opts> = [--T]"                       # transpose matrix
        sys.exit(1)
        
    nnetfile   = sys.argv[1]
    outd       = sys.argv[2]
    mtrnp      = False   # If true, transpose matrices before writing out to files

    for ag in sys.argv[3:]:
        if re.match("--T", ag):
            mtrnp = True
        else:
            print >> sys.stderr, "Error: Unknown option: " + ag
            sys.exit(1)

    # load kaldi (text) format Nnet
    nn = nnet.loadnnet(nnetfile) 
    # save as numpy matrix
    nn.exportParams(outd, mtrnp)
