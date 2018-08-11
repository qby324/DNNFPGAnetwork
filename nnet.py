# Nnet module

# Thu Sep 18 12:15:26 JST 2014
# Extended to handle DAG of nuclei
# First nuclei is assumed to be an input layer.
# Last nuclei is assumed to be an output layer.

# Manipulates kaldi format Nnet
# Takahiro Shinozaki
# Wed Nov 27 13:38:54 JST 2013

import sys
import re
import random
import struct
import math

import numpy
import theano
import theano.tensor as T

debug=0

class Nnet(object):
    def __init__(self, ncl):
        self.nuclei = ncl # list of nucleus

    # input size of the network
    def nnInum(self):
        return self.nuclei[0].nnInum() # inum of the first unit

    # output size of the network
    def nnNnum(self):
        return self.nuclei[-1].nnNnum() # nnum of the last unit

    def writeParams(self, f):
        f.write("<Nnet>\n")
        for i in range(len(self.nuclei)):
            self.nuclei[i].writeParams(f)
        f.write("</Nnet>\n")

    def exportParams(self, dirname, tflg):
        for i in range(len(self.nuclei)):
            self.nuclei[i].exportParams(dirname, tflg)

    # parameters of this network. 
    def params(self):
        if self.connectchk() == 1:
            tmp=[]
            for i in range(len(self.nuclei)):
                tmp += self.nuclei[i].params()
            return tmp
        else:
            return []
        
    def diffParams(self):
        if self.connectchk() == 1:
            tmp=[]
            for i in range(len(self.nuclei)):
                tmp += self.nuclei[i].diffParams()
            return tmp
        else:
            return []

    def forward(self,x):
        return self.nuclei[-1].forwardChain(x)

    def connectchk(self):
        if self.nuclei[-1].connectflg == 0:
            self.nuclei[0].connectflg = 1
            return self.nuclei[-1].connectchk()
        else:
            return self.nuclei[-1].connectflg

    def resetconnectflg(self):
        for i in range(len(self.nuclei)):
            self.nuclei[i].connectflg = 0

    def debuginfo(self):
        print "# --debug info--"
        for i in range(len(self.nuclei)):
            print "# NN: nucleous " + str(i)
            print "#  prevnuclids " + str(self.nuclei[i].prevnuclids)
#            for j in range(len(self.nuclei[i].prevnucl)):
#                print "#  prevnucid " + str(self.nuclei[i].prevnucl[j].nucid)
            print "#  connectflg = " + str(self.nuclei[i].connectflg)
        print "# -----"


    def appendNucleus(self, nuc):
        self.nuclei += [nuc]
        self.nuclei[-1].setPrevNuclei(self.nuclei)

    def networkeffectsize(self):
        if self.connectchk() == 1:
            for i in range(len(self.nuclei)):
                if self.nuclei[i].type() == "affine":
                    if self.nuclei[i].connectflg == 1:
                        print "(" + str(self.nuclei[i].inum) + ", " + str(self.nuclei[i].nnum) + ")"
                    else:
                        print "(,)"
        else:
            print "(" + str(0) + ", " + str(0) + ")"
            
                    
                    


# abstract like Nnet nucleus class
class NnetNucleus(object):
    connectflg = 0

    def __init__(self):
        self.nucid = -1
        self.inum = 0
        self.nnum = 0
        self.prevnuclids = []
        self.prevnucl = []
        return

    # nucleus id
    def nucleusid(self):
        return self.nucid

    # transformation type
    def type(self):
        return "null"

    # input size of the network
    def nnInum(self):
        return self.inum

    # output size of the network
    def nnNnum(self):
        return self.nnum

    # transformation
    def forward(self,x):
        # x is a matrix where a row is a frame and a column is dimension
        return x

    # params of this nucleus
    def params(self):
        return []

    # diff params of this nucleus
    def diffParams(self):
        return []

    # links to previous nuclei
    def setPrevNuclei(self, ncl):
        self.prevnucl = []
        for i in range(len(self.prevnuclids)):
            self.prevnucl += [ncl[self.prevnuclids[i]]]

    # transformation chain
    def forwardChain(self,x):
        if debug:
            print "# forwardchain: " + str(self.nucid)
        if self.prevnucl:
            tmp = []
            for i in range(len(self.prevnucl)):
                tmp+=[self.prevnucl[i].forwardChain(x)]
            return self.forward(T.concatenate(tmp, axis=1))
        else:
            return self.forward(x)

    # The connectflg is 1 if these exist a path from the input nucleus 
    # to this nucleus. It is -1 if there is no path from the input 
    # nucleus to this one. It is 0 if it is not known.
    # If there is no path from the input to this nucleus, then the parameters
    # need to be excluded from training.
    def connectchk(self):
        if self.connectflg == 1 or self.connectflg == -1:
            return self.connectflg
        elif self.prevnucl:
            tmp = 0
            for i in range(len(self.prevnucl)):
                if self.prevnucl[i].connectchk()==1:
                    tmp+=1
            if 0<tmp:
                self.connectflg = 1
            else:
                self.connectflg = -1
            return self.connectflg
        else:
            self.connectflg = -1
            return self.connectflg

    def writeParams(self, f):
        pass

    def exportParams(self, dirname, tflg):
        pass
            

# Dummy nucleus class to express input
# Required when DAG extended network is used
class InputNucleus(NnetNucleus):
    # constructor: 
    def __init__(self, inum):
        self.nucid = 0
        self.inum = inum
        self.nnum = inum
        self.prevnuclids = []
        self.prevnucl = []

    # transformation type
    def type(self):
        return "input"

    def writeParams(self, f):
        f.write("<input> ")
        f.write(str(self.nnum) + " " + str(self.inum) + "\n")

#  Affine nucleus class
class AffineNucleus(NnetNucleus):

    # constructor: weight matrix and bias vector
    def __init__(self, nucid, prevnuclids, W, bias):
#        print W.shape
        self.nucid = nucid
        self.prevnuclids = prevnuclids
        self.prevnucl=[]
        if W.shape[0]==0 or W.shape[1]==0:
            self.inum=0
            self.nnum=bias.shape[0]
        else:
            self.inum=W.shape[0] # Note: W is assumed to be transposed from the definition file
            self.nnum=W.shape[1]
#        print "(" + str(self.inum) + ", " + str(self.nnum) + ")"
        self.W   =theano.shared(value=W   ,name="W")
        self.Bias=theano.shared(value=bias,name="Bias")
        self.DiffW   =theano.shared(
            value=numpy.zeros(self.inum*self.nnum,
                              dtype=theano.config.floatX).reshape(self.inum,
                                                                  self.nnum),
            name="DiffW")
        self.DiffBias=theano.shared(
            value=numpy.zeros(self.nnum,dtype=theano.config.floatX),
            name="DiffBias")
        return

    # transformation type
    def type(self):
        return "affine"

    # transformation
    def forward(self,x):
        return T.dot(x,self.W)+self.Bias

    # params of this nucleus
    def params(self):
        if self.connectflg == 1:
            return [self.W, self.Bias]
        else:
            return []

    # diff params of this nucleus
    def diffParams(self):
        if self.connectflg == 1:
            return [self.DiffW, self.DiffBias]
        else:
            return []

    def writeParams(self, f):
        f.write("<affinetransform> ")
        if self.prevnuclids and not (len(self.prevnuclids)==1 and self.prevnuclids[0] == self.nucid -1):
            f.write("[ ")
            for i in range(len(self.prevnuclids)):
                f.write(str(self.prevnuclids[i]) + " ")
            f.write("] ")
        elif self.prevnuclids == [] and not self.connectflg==1:  # if prevnuclids == [] and self.connectflg=1, it is the first unit 
            f.write("[ ] ")
        f.write(str(self.nnum) + " " + str(self.inum) + "\n")
        f.write(" [ \n")
        f.write(re.sub(",", "", re.sub("\]", "", re.sub("\[", " ", re.sub("\],", "\n", str(self.W.get_value().T.tolist()))))))
        f.write(" ]\n")
        f.write(re.sub("\]", " ]\n", re.sub("\[", " [ ", re.sub(",", "", str(self.Bias.get_value().tolist())))))
#        f.write(re.sub("\]", " \]\n", re.sub("\[", " \[ ", re.sub(",", "", str(self.Bias.get_value().tolist()))))
        # Why max_line_width doesn't work?
        # f.write("[\n " + re.sub("  ", " ", re.sub("\[|\]", "", numpy.array_str(self.W.get_value().T, max_line_width=1000000))) + " ] \n")
        # f.write(re.sub(",", "", numpy.array_str(self.Bias.get_value(), max_line_width=100000))+" \n")

    def exportParams(self, dirname, tflg):
        outfname = dirname + "/W_l" + str(self.nucid) + ".npy"
        if tflg:
            numpy.save(outfname, self.W.get_value().T);
        else:
            numpy.save(outfname, self.W.get_value());

        outfname = dirname + "/bias_l" + str(self.nucid) + ".npy"
        if tflg:
            numpy.save(outfname, numpy.array([self.Bias.get_value()]));
        else:
            numpy.save(outfname, numpy.array([self.Bias.get_value()]).T);

#  Sigmoid nucleus class
class SigmoidNucleus(NnetNucleus):
    # constructor: num neurons
    def __init__(self, nucid, prevnuclids, nnum):
        self.nucid = nucid
        self.inum=nnum
        self.nnum=nnum
        self.prevnuclids = prevnuclids
        self.prevnucl=[]

    # transformation type
    def type(self):
        return "sigmoid"

    # transformation
    def forward(self,x):
        return T.nnet.sigmoid(x)

    def writeParams(self, f):
        f.write("<sigmoid> ")
        if self.prevnuclids and not (len(self.prevnuclids)==1 and self.prevnuclids[0] == self.nucid -1):
            f.write("[ ")
            for i in range(len(self.prevnuclids)):
                f.write(str(self.prevnuclids[i]) + " ")
            f.write("] ")
        f.write(str(self.nnum) + " " + str(self.inum) + "\n")

# Softmax nucleus class
class SoftmaxNucleus(NnetNucleus):
    # constructor: num neurons
    def __init__(self, nucid, prevnuclids, nnum):
        self.nucid = nucid
        self.inum=nnum
        self.nnum=nnum
        self.prevnuclids = prevnuclids
        self.prevnucl=[]

    # transformation type
    def type(self):
        return "softmax"

    # transformation
    def forward(self,x):
        return T.nnet.softmax(x)

    def writeParams(self, f):
        f.write("<softmax> ")
        if self.prevnuclids and not (len(self.prevnuclids)==1 and self.prevnuclids[0] == self.nucid -1):
            f.write("[ ")
            for i in range(len(self.prevnuclids)):
                f.write(str(self.prevnuclids[i]) + " ")
            f.write("] ")
        f.write(str(self.nnum) + " " + str(self.inum) + "\n")

# extraction nucleus
class ExtractionNucleus(NnetNucleus):
    # constructor: 
    def __init__(self, nucid, prevnuclids, nin, nout, spos, epos):
        self.nucid = nucid
        self.inum=nin
        self.nnum=nout
        self.spos=spos
        self.epos=epos
        self.prevnuclids = prevnuclids
        self.prevnucl=[]

    # transformation type
    def type(self):
        return "extraction"

    # transformation
    def forward(self,x):
        return x[:,self.spos:self.epos]

    def writeParams(self, f):
        f.write("<extraction> ")
        if self.prevnuclids and not (len(self.prevnuclids)==1 and self.prevnuclids[0] == self.nucid -1):
            f.write("[ ")
            for i in range(len(self.prevnuclids)):
                f.write(str(self.prevnuclids[i]) + " ")
            f.write("] ")
        f.write(str(self.nnum) + " " + str(self.inum) \
                    + " " + str(self.spos) + " " + str(self.epos)+ "\n")

 
# Get a line skipping empty ones.
# If there is no more line, exit with an error message if msg is given 
# or return an empty result.
def mgetline(f, msg):
    while True:
        line = f.readline()
        if not line and msg:
            print >> sys.stderr, "Error: Unexpected format: " + msg
            sys.exit(1)
        elif line.strip() != "" or not line:
            break
    return line.strip()

# Load Kaldi's Nnet written in text format
# (the format is extended by shinot to support DAG of nuclei)
def loadnnet(fname):
    def getinput(nucid, line):
        if nucid != 0:
            print >> sys.stderr, "Error: nucid of InputNucle must be 0: " + line
            sys.exit(1)
        m=re.match("<input>\s*(?P<nout>\d+)\s*(?P<nin>\d+)\s*$", line)
        if m:
            nout = int(m.group("nout"))
            nin = int(m.group("nin"))
        else:
            print >> sys.stderr, "Error: Unexpected format : " + line
            sys.exit(1)
        if nout != nin:
            print >> sys.stderr, "Error: nout must equal to nin for InputNucle: " + line
            sys.exit(1)
        return InputNucleus(nin)

    def getaffine(nucid, line):
        m=re.match("<affinetransform>(?P<inl>\s*\[\s*((\d+)\s*)*\])?\s*(?P<nout>\d+)\s*(?P<nin>\d+)\s*$", line)
        if m:
            nout = int(m.group("nout"))
            nin = int(m.group("nin"))
            pre=m.group("inl")
            if pre:
                prevnuclids = map(int, re.findall("(\d+)", m.group("inl")))
            elif 0<nucid:
                prevnuclids = [nucid-1]
            else:
                prevnuclids = []
        else:
            print >> sys.stderr, "Error: Unexpected format : " + line
            sys.exit(1)
        nr = 0
        wmat = []
        if 0<nin:
            while nr < nout:
                line = mgetline(f, "Error: File ended in the middle of affine transform definition");
                if re.match("\[$", line):
                    continue
                unit = map(float, line.strip("\]").split())
                wmat = wmat + [unit]
                nr = nr + 1
            wmatnp = numpy.array(wmat, dtype=theano.config.floatX).T # a row is a neuron
        else:
            line = mgetline(f, "Error: File ended in the middle of affine transform definition");
            line = mgetline(f, "Error: File ended in the middle of affine transform definition");
            wmatnp = numpy.array([[]], dtype=theano.config.floatX)
        line = mgetline(f, "Error : File ended in the middle of affine transform const definition");
        bias = map(float, line.strip("\[\]").split())
        biasnp = numpy.array(bias, dtype=theano.config.floatX)
        return AffineNucleus(nucid, prevnuclids, wmatnp, biasnp)

    def getsigmoid(nucid, line):
        m=re.match("<sigmoid>(?P<inl>\s*\[\s*((\d+)\s*)+\])?\s*(?P<nout>\d+)\s*(?P<nin>\d+)\s*$", line)
        if m:
            nout = int(m.group("nout"))
            nin = int(m.group("nin"))
            pre=m.group("inl")
            if pre:
                prevnuclids = map(int, re.findall("(\d+)", m.group("inl")))
            elif 0<nucid:
                prevnuclids = [nucid-1]
            else:
                prevnuclids = []
        else:
            print >> sys.stderr, "Error: Unexpected format : " + line
            sys.exit(1)
        if nout != nin:
            print >> sys.stderr, "Error: nin and nout must be the same: " + line
            sys.exit(1)
        return SigmoidNucleus(nucid, prevnuclids, nout)

    def getsoftmax(nucid, line):
        m=re.match("<softmax>(?P<inl>\s*\[\s*((\d+)\s*)+\])?\s*(?P<nout>\d+)\s*(?P<nin>\d+)\s*$", line)
        if m:
            nout = int(m.group("nout"))
            nin = int(m.group("nin"))
            pre=m.group("inl")
            if pre:
                prevnuclids = map(int, re.findall("(\d+)", m.group("inl")))
            elif 0<nucid:
                prevnuclids = [nucid-1]
            else:
                prevnuclids = []
        else:
            print >> sys.stderr, "Error: Unexpected format : " + line
            sys.exit(1)
        if nout != nin:
            print >> sys.stderr, "Error: nin and nout must be the same: " + line
            sys.exit(1)
        return SoftmaxNucleus(nucid, prevnuclids, nout)
        
    def getextraction(nucid, line):
        m=re.match("<extraction>(?P<inl>\s*\[\s*((\d+)\s*)+\])?\s*(?P<nout>\d+)\s*(?P<nin>\d+)\s*(?P<spos>\d+)\s*(?P<epos>\d+)\s*$", line)
        if m:
            nout = int(m.group("nout"))
            nin = int(m.group("nin"))
            spos = int(m.group("spos"))
            epos = int(m.group("epos"))
            pre=m.group("inl")
            if pre:
                prevnuclids = map(int, re.findall("(\d+)", m.group("inl")))
            elif 0<nucid:
                prevnuclids = [nucid-1]
            else:
                prevnuclids = []
        else:
            print >> sys.stderr, "Error: Unexpected format : " + line
            sys.exit(1)
        return ExtractionNucleus(nucid, prevnuclids, nin, nout, spos, epos)

    nuclei = []
    f = open(fname, "r")
    line = mgetline(f, "<Nnet> expected")
    if line != "<Nnet>":
        print >> sys.stderr, "Error: Not Nnet file"
        sys.exit(1)
    nucid = 0
    while True:
        line = mgetline(f, "");
        if not line:
            print >> sys.stderr, "Error: file ended in the middle of Nnet definition"
            sys.exit(1)
        else:
            if re.match("<input>", line):
                nuclei += [getinput(nucid, line)]
            elif re.match("<affinetransform>", line):
                nuclei += [getaffine(nucid, line)]
            elif re.match("<sigmoid>", line):
                nuclei += [getsigmoid(nucid, line)]
            elif re.match("<softmax>", line):
                nuclei += [getsoftmax(nucid, line)]
            elif re.match("<extraction>", line):
                nuclei += [getextraction(nucid, line)]
            elif re.match("</Nnet>", line):
                f.close()
                break
            else:
                print >> sys.stderr, "Error: Unexpected format or not supported yet : " + line
                sys.exit(1)
            nucid = nucid+1
    for i in range(1,len(nuclei)):
        nuclei[i].setPrevNuclei(nuclei)
        # check for extraction nucleus
        if nuclei[i].type() == "extraction":
            if nuclei[i].epos <= nuclei[i].spos:
                print >> sys.stderr, "Error (nnet.py): spos is equal or larger than epos in an extraction nuclei"
                print >> sys.stderr, "  nuclei(" + str(i) + ") spos= " + str(nuclei[i].spos) + " epos= " + str(nuclei[i].epos)
                print >> sys.stderr, "  prevs =" + str(nuclei[i].prevnuclids)
                sys.exit(1)
            totsize=0
            for j in range(len(nuclei[i].prevnucl)):
                totsize=nuclei[i].prevnucl[j].nnNnum()
            if totsize-1 < nuclei[i].spos or totsize < nuclei[i].epos:
                print >> sys.stderr, "Error (nnet.py): spos or epos of an extraction nuclei is out of range"
                print >> sys.stderr, "  nuclei(" + str(i) + ") spos= " + str(nuclei[i].spos) + " epos= " + str(nuclei[i].epos)
                print >> sys.stderr, "  prevs =" + str(nuclei[i].prevnuclids)
                sys.exit(1)
    if 0<len(nuclei):
        nuclei[0].connectflg = 1
    return Nnet(nuclei)

def loadscpf(fname):
    fp = open(fname)
    scp=[]
    while True:
        line = mgetline(fp, "")
        if not line:
            break;
        scp.append(line)
    return scp

def loadMVFile(fname):
    fp = open(fname)

    mean = []
    std = []
    while True:
        line = mgetline(fp, "")
        if not line:
            break;
        if re.match(r"<MEAN> \d+", line):
            dim = re.match(r"<MEAN> \d+", line).group().split()[1]
            line = mgetline(fp, "mean")
            mean = map(float, line.split())
        if re.match(r"<VARIANCE> \d+", line):
            dim = re.match(r"<VARIANCE> \d+", line).group().split()[1]
            line = mgetline(fp, "var")
            std = map(math.sqrt, map(float, line.split()))

    if mean == []:
        print >> sys.stderr, "Error: mean definition is not found"
        sys.exit(1)
    if std == []:
        print >> sys.stderr, "Error: variance definition is not found"
        sys.exit(1)

    return numpy.array([mean, std], dtype="float32")

def loadHtkFile(fname):
    fp = open(fname)
    nSamples = struct.unpack('>I', fp.read(4))[0]
    sampPeriod = struct.unpack('>I', fp.read(4))[0]
    sampSize = struct.unpack('>H', fp.read(2))[0]
    parmKind = struct.unpack('>H', fp.read(2))[0]

    fea = struct.unpack('>'+str(sampSize/4*nSamples)+'f', fp.read(sampSize*nSamples))
    feaA = numpy.array(fea, dtype="float32").reshape(nSamples, sampSize/4)
    if 1e30 < feaA.max():
        print >> sys.stderr, "WARN:  Huge value (1e30<). Broken data?" + fname
    if -1e30 > feaA.max():
        print >> sys.stderr, "WARN:  Huge value (-1e30>). Broken data?" + fname
    return feaA

def loadArkFile(fp):
    uid=""
    while True:
        a = fp.read(1)
        if a == ' ':
            break
        uid+=a

    a = fp.read(1)
    if a != '\0':
        print >> sys.stderr, "Error: Expecting '\\0'"
        sys.exit(1)

    chk=""
    for i in range(4):
        a = fp.read(1)
        chk+=a
    if chk == "BCM ":
        print >> sys.stderr, "Error: Compressed data is not supported yet."
        sys.exit(1)
    if chk != "BFM ":
        print >> sys.stderr, "Error: Expecting 'BFM'"
        sys.exit(1)

    a = fp.read(1)
    row = struct.unpack('<I', fp.read(4))[0]
    a = fp.read(1)
    col = struct.unpack('<I', fp.read(4))[0]

    # assumes little endian
    fea = struct.unpack('<'+str(col*row)+'f', fp.read(4*col*row))
    feaA = numpy.array(fea, dtype="float32").reshape(row, col)
    if 1e30 < feaA.max():
        print >> sys.stderr, "WARN:  Huge value (1e30<). Broken data?" + fname
    if -1e30 > feaA.max():
        print >> sys.stderr, "WARN:  Huge value (-1e30>). Broken data?" + fname
    return (feaA, uid)


def expandContext(fea,
                  spl):
    (num, dim) = fea.shape
    tmpl = []
    for i in range(spl):
        tmpl.append(numpy.concatenate([numpy.zeros([spl-i, dim], dtype="float32"), fea[0:num-spl+i]], axis=0))
    tmpl.append(fea)
    for i in range(spl):
        tmpl.append(numpy.concatenate([fea[i+1:num], numpy.zeros([i+1, dim], dtype="float32")], axis=0))

    exfea = numpy.concatenate(tmpl, axis=1)
    return exfea
   # 1 1 1
   # 2 2 2
   # 3 3 3
   # 4 4 4
   # 5 5 5 
   # ->
   # 0 0 0  0 0 0  1 1 1  2 2 2  3 3 3  
   # 0 0 0  1 1 1  2 2 2  3 3 3  4 4 4
   # 1 1 1  2 2 2  3 3 3  4 4 4  5 5 5
   # 2 2 2  3 3 3  4 4 4  5 5 5  0 0 0
   # 3 3 3  4 4 4  5 5 5  0 0 0  0 0 0

def loadFFile(flist,
              fmt,
              spl,
              trmspl,
              idlist,
              mnstd):

    prearkf=""
    udlist=[]
    for i in range(len(idlist)):
        if fmt=="htkscp":
            fea = loadHtkFile(flist[i])
        elif fmt=="arkscp":
            uid = flist[i].split(" ")[0]
            arkf = flist[i].split(" ")[1].split(":")[0]
            fpos = int(flist[i].split(" ")[1].split(":")[1])
            if prearkf!=arkf:
                if prearkf!="":
                    fp.close()
                fp = open(arkf)
            fp.seek(fpos-len(uid)-1)
            (fea, uid2) = loadArkFile(fp)
            if uid != uid2:
                print >> sys.stderr, "Error: inconsistent utterance ID. scp:" + uid + " ark:" + uid2
                sys.exit(1)
        else:
            print >> sys.stderr, "Error: unsupported data format : " + fmt
            sys.exit(1)

        if len(mnstd)!=0:
            fea = (fea - mnstd[0])/mnstd[1]
        if 0<spl:
            fea = expandContext(fea, spl)
            if trmspl:
                fea = fea[spl:-spl]

        udlist.append(fea)

    data = numpy.concatenate(udlist)
    return data


def loadCFile(fname,
              fmt):
    ref = dict()
    if fmt=="arkiv":
        fp = open(fname)
        while True:
            line = fp.readline()
            if line == "":
                break;
            elems = line.split()
            uid = elems[0]
            wids = numpy.array(map(int, elems[1:len(elems)]))
            ref[uid] = wids
    else:
        print >> sys.stderr, "Unknown data type. Not supported yet. " + fmt + " " + fname
        sys.exit(1)

    return ref


def extCData(alldata,
             idlist,
             trim):
    # ** Repeating numpy.concatenate is slow due to memory copy of O(n^2) **
    # data=numpy.array([], dtype="int")
    # for uid in idlist:
    #     data=numpy.concatenate([data, alldata[uid]])
    totlen = 0
    for uid in idlist:
        totlen += len(alldata[uid])
    data = numpy.zeros(totlen, dtype="int32")
    st=0
    for uid in idlist:
        ulen=len(alldata[uid])-trim*2
        data[st:st+ulen] = alldata[uid][trim:trim+ulen]
#        ulen=len(alldata[uid])
#        data[st:st+ulen] = alldata[uid]
        st += ulen
    return data[0:st]

def htkBasename(line):
    return re.sub(r'\..*$', '', line.split("/")[-1])
def arkBasename(line):
    return line.split(" ")[0]
    
def scp2idlist(lst,
               scptype):
    if scptype == "htkscp":
        return map(htkBasename, lst)
    elif scptype == "arkscp":
        return map(arkBasename, lst)

def prepX(listX,
           dataXall,
           mnstd,
           typeX,
           bbsize,
           lot,
           spl,
           trmspl,
           shflg):

    if not typeX in ("htkscp", "arkscp"):
        print >> sys.stderr, "Right now, only htkscp and arkscp are supported as xType."
        sys.exit(1)
        # memo: to add support for arkiv, add a code to expand position index to feature vector.
        # ex. 2 -> 0 0 1 0

    # make id list of this lot
    if typeX in ("htkscp", "arkscp"):
        if bbsize==0:
            bbsize = len(listX)
        st = lot*bbsize
        ed = (lot+1)*bbsize
        idlist = scp2idlist(listX[st:ed], typeX)
    else:
        if bbsize==0:
            bbsize = len(dataXall)
        st = lot*bbsize
        ed = (lot+1)*bbsize
        idlist = dataXall.keys()[st:ed]

    if idlist==[]:
        return numpy.array([])

    # prepare data
    if typeX in ("htkscp", "arkscp"):
        dataX = loadFFile(listX[st:ed], typeX, spl, trmspl, idlist, mnstd)
    elif typeX in ("arkiv"):
        if trmspl:
            dataX = extCData(dataXall, idlist, spl)
        else:
            dataX = extCData(dataXall, idlist, 0)

    if shflg:
        rindx=range(dataX.shape[0])
        random.shuffle(rindx)
        dataX=dataX[rindx]

    return dataX

def prepXY(listX,
           listY,
           dataXall,
           dataYall,
           mnstd,
           typeX,
           typeY,
           bbsize,
           lot,
           spl,
           trmspl,
           shflg):

    if not typeX in ("htkscp", "arkscp"):
        print >> sys.stderr, "Right now, only htkscp and arkscp are supported as typeX."
        # memo: to add support for arkiv, add a code to expand position index to feature vector.
        # ex. 2 -> 0 0 1 0
        sys.exit(1)

    # make id list of this lot
    if typeX in ("htkscp", "arkscp"):
        if bbsize==0:
            bbsize = len(listX)
        st = lot*bbsize
        ed = (lot+1)*bbsize
        idlist = scp2idlist(listX[st:ed], typeX)
    elif typeY in ("htkscp", "arkscp"):
        if bbsize==0:
            bbsize = len(listY)
        st = lot*bbsize
        ed = (lot+1)*bbsize
        idlist = scp2idlist(listY[st:ed], typeY)
    else:
        if bbsize==0:
            bbsize = len(dataXall)
        st = lot*bbsize
        ed = (lot+1)*bbsize
        idlist = dataXall.keys()[st:ed]

    if idlist==[]:
        return (numpy.array([]), numpy.array([]))

    # prepare data
    if typeX in ("htkscp", "arkscp"):
        dataX = loadFFile(listX[st:ed], typeX, spl, trmspl, idlist, mnstd)
    elif typeX in ("arkiv"):
        if trmspl:
            dataX = extCData(dataXall, idlist, spl)
        else:
            dataX = extCData(dataXall, idlist, 0)

    if typeY in ("htkscp", "arkscp"):
        dataY = loadFFile(listY[st:ed], typeY, spl, trmspl, idlist, numpy.array([]))
    elif typeY in ("arkiv"):
        if trmspl:
            dataY = extCData(dataYall, idlist, spl)
        else:
            dataY = extCData(dataYall, idlist, 0)

    if shflg:
        rindx=range(dataX.shape[0])
        random.shuffle(rindx)
        dataX=dataX[rindx]
        dataY=dataY[rindx]

    return (dataX, dataY)
             
