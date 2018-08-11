#! /bin/bash -e

# Sat Jan  3 14:29:57 JST 2015
# shinot
export THEANO_FLAGS="floatX=float32,device=gpu"

# HMM alignment file to make train and development data (output side of NN)
alimlf=/net/monshiro/storage1/shinot/slp2013tutorial/data/lab.mlf
# definition file of alignment symbol to ID mapping
st2idmap=/net/monshiro/storage1/shinot/slp2013tutorial/work/prefs/mapfile
# initial file of NN
ftrans0=trans.ini    
# list of training data (input side of NN)
xtrain=ids.train.scp 
# list of development data (input side of NN)
xvalid=ids.valid.scp 

xType=htkscp   # data type (input side of NN)
yType=arkiv    # data type (output side of NN)
ydat=slp.arkiv # file to store train and development data made from alignment file (output side)
ytrain=$ydat   # utterances corresponding to xtrain are automatically extracted from this file
yvalid=$ydat   # utterances corresponding to xvalid are automatically extracted from this file
mvnormf=slp.norm # file to store mean and variance for input normalization
feadim=130     # feature dimension
splice=2       # expand feature vectors appending frames before and after the current frame. (splice + 1 + split)
rbmepoch=5    # number of epochs for RBM pretraining
bpepoch=5     # number of epochs for fine tuning
outdir=out     # directory to store trained MLP etc
nnfeaout=nnfea # directory to store NN outputs as HTK format features

mlf2arkiv=/net/monshiro/storage1/shinot/mysrc/tsDeep/tsdeep.v8.1/mlf2arkiv.pl
compmv=/net/monshiro/storage1/shinot/mysrc/tsDeep/tsdeep.v8.1/compmeanvar.py
rbmtrain=/net/monshiro/storage1/shinot/mysrc/tsDeep/tsdeep.v8.1/rbmtrain.py
backprop=/net/monshiro/storage1/shinot/mysrc/tsDeep/tsdeep.v8.1/backprop.py
nnetgenhtkfea=/net/monshiro/storage1/shinot/mysrc/tsDeep/tsdeep.v8.1/nnetgenhtkfea.py 

############ Makes Y data from alignment file ##########################
$mlf2arkiv $alimlf $st2idmap > $ydat

############ compute mean and variance for input normalization #########
$compmv $xtrain $mvnormf

############ RBM PRETRAINING ###########################################

echo \#\# RBM pretraining
mkdir -p $outdir
cp -fp $ftrans0 $outdir/trans.0


# rbm 1st layer
rbmi=1
outsize=32
prev=$(($rbmi - 1))
$rbmtrain $xtrain $outsize --xType=$xType --spl=$splice --mvnf=$mvnormf --ft=$outdir/trans.${prev} --sd=1234 \
    --rt=gb --lr=0.002 --mm=0.9 --rgl=0.00001  --of=$outdir/rbm.$rbmi --wt=$outdir/trans.${rbmi} --ep=$rbmepoch --trmspl=True

# rbm 2nd and 3rd layers
for rbmi in `seq 2 3`; do
    outsize=32
    prev=$(($rbmi - 1))
    $rbmtrain $xtrain $outsize --xType=$xType --spl=$splice --mvnf=$mvnormf --ft=$outdir/trans.${prev} --sd=1234 \
	--rt=bb --lr=0.005 --mm=0.9 --rgl=0.0001  --of=$outdir/rbm.$rbmi --wt=$outdir/trans.${rbmi} --ep=$rbmepoch --trmspl=True
done

# output layer (random)
outsize=19
rbmi=4
prev=$(($rbmi - 1))
$rbmtrain /dev/null $outsize --xType=$xType --spl=$splice --mvnf=$mvnormf --ft=$outdir/trans.${prev} --sd=1234 \
    --rt=gb --lr=0.001 --mm=0.9 --rgl=0.00001  --of=$outdir/rbm.$rbmi --wt=$outdir/mlp.ini --at=softmax --ep=0 --trmspl=True

############ MLP FINETUNING ############################################
echo \#\# backprop
$backprop $outdir/mlp.ini $xtrain $ytrain --xType=$xType --spl=$splice --mvnf=$mvnormf --yType=$yType \
    --devX=$xvalid --devY=$yvalid --ep=$bpepoch --ot=c --lr=1.0 --mm=0.0 --rgl=0.0  --ofs=$outdir/mlp --trmspl=True
(cd $outdir; rm -f mlp.final; ln -s mlp.$bpepoch mlp.final)


############ Export output of NN as HTK feature file ###################
mkdir -p $nnfeaout
$nnetgenhtkfea $outdir/mlp.final $xvalid $nnfeaout --spl=$splice --mvnf=$mvnormf --trmspl=False
