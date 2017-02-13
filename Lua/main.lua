-----------------------------------------------------------------------------------------------------------------------
-- requirements:
-- data: train/test mat file -- data.v(point feature, with last three features being the xyz coordinates) data.V(LB basis) data.D(LB energy) data.label(point label)
--       jointSpec mat file -- data.V(joint LB basis) data.D(joint LB energy) data.vidx(voxel space index) assume ngrid=20
-- comments: in the aligned setting, we use voxelized R3 space to map individual spec to the jointSpec
-----------------------------------------------------------------------------------------------------------------------
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'MyDataContainer'
require 'MyTrainer'
require 'MyModel'
require 'MyCriterion'
require 'MyCriterionSpecTN'
require 'torchx'
matio = require 'matio'
matio.use_lua_strings = true
tensorType = "torch.CudaTensor"

cmd = torch.CmdLine()
cmd:option('-s','','synset option')
cmd:option('-i',33,'number of input vertex functions')
cmd:option('-o',4,'range of output labels')
cmd:option('-d',15,'number of basis to be synchronized for each shape')
cmd:option('-dj',45,'number of basis for joint laplacian')
cmd:option('-ntr',4,'number of training chunks')
cmd:option('-nte',1,'number of test chunks')
cmd:option('-nval',1,'number of validation chunks')
cmd:option('-e_b1',20,'number of maximum training epochs for SpecTN pre-training')
cmd:option('-b_b1',48,'batch size for SpecTN pre-training')
cmd:option('-e',40,'number of maximum training epochs for main training')
cmd:option('-b',24,'batch size for main training')
cmd:option('-g',0,'gpuIndex')
cmd:option('-l',3e-3,'learning rate')
params = cmd:parse(arg)

synset = params.s
if synset=='' then
    print('please specify a synset!')
    return
end
path = './Model/'..synset
if not paths.dirp(path) then
    paths.mkdir(path)
end
cutorch.setDevice(params.g+1)

datapath = '../Data/Categories/'
jointspecpath = '../Data/JointSpec/'

trainList = {}
for i=1,params.ntr do
    trainList[i] = matio.load(datapath..synset..'/train_'..i..'.mat')
end
testList = {}
for i=1,params.nte do
    testList[i] = matio.load(datapath..synset..'/test_'..i..'.mat')
end
valList = {}
for i=1,params.nval do
    valList[i] = matio.load(datapath..synset..'/val_'..i..'.mat')
end
function trainList:size() 
    return #self
end
function testList:size() 
    return #self
end
function valList:size() 
    return #self
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function pretrainSpecTN(ndim, ndimJoint, batchSize, maxEpoch, lr)
    -- train SpecTN
    model_spectn = SpecTN(ndim,ndimJoint):cuda()
    criterion_spectn = nn.MyCriterionSpecTN():cuda()
    lossHistory_spectn = {}
    state_spectn = {}
    -- train config
    trainer_spectn = MyTrainer(model_spectn, criterion_spectn)
    trainer_spectn.lossHistory = lossHistory_spectn
    trainer_spectn.epoch = 1
    trainer_spectn.maxEpoch = maxEpoch or 30
    trainer_spectn.batchSize = batchSize or 48
    trainer_spectn.isoutput = true
    trainer_spectn.verbose = true
    trainer_spectn.outputeveryNepoch = 5
    trainer_spectn.datagen = gen_batch_input_spectn
    trainer_spectn.config.learningRateDecay = 0
    trainer_spectn.config.weightDecay = 0
    trainer_spectn.config.learningRate = lr or 3e-3
    trainer_spectn.solver = optim.adam
    trainer_spectn.config.beta1 = 0.9
    trainer_spectn.config.beta2 = 0.999
    trainer_spectn.state = state_spectn
    trainer_spectn.tensorType = tensorType --'torch.CudaTensor' or 'torch.DoubleTensor'

    local specParamJoint = matio.load(jointspecpath..synset..'.mat')
    trainer_spectn.specParamJoint = {}
    trainer_spectn.specParamJoint.V = specParamJoint.V:type(tensorType)
    trainer_spectn.specParamJoint.D = specParamJoint.D:type(tensorType)
    trainer_spectn.specParamJoint.vidx = specParamJoint.vidx:view(-1):type(tensorType)
    trainer_spectn.specParamJoint.ngrid = 20
    trainer_spectn.specParamJoint.ndim_sep = ndim
    trainer_spectn.specParamJoint.ndim_joint = ndimJoint

    trainer_spectn.evaluate = 0
    trainer_spectn.outputpath = './Model/'..synset..'/'
    trainer_spectn.outputprefix = 'FuncMap'..synset
    if not paths.dirp(trainer_spectn.outputpath) then
        paths.mkdir(trainer_spectn.outputpath)
    end
    
    -- start train
    trainer_spectn:train(trainList)
end

function trainLearnTrans(ndim, ndimJoint, nin, nout, batchSize, maxEpoch, lr1, lr2, e_b1)
    -- train model learn trans
    spectn = torch.load('./Model/'..synset..'/FuncMap'..synset..'_epoch'..e_b1..'.dat')
    model_spectn = spectn[2]
    state_spectn = spectn[3]
    model = SyncSpecCNN(model_spectn,ndim,ndimJoint,nin,nout):cuda()
    criterion = nn.MyCriterion(ndim):cuda()
    lossHistory = {}
    state = {}
    splitimx = torch.Tensor(#model.modules):fill(0)
    for i=2,#model.modules do
        if model.modules[i].weight or model.modules[i].modules then
            splitimx[i] = model.modules[i]:getParameters():size(1)+splitimx[i-1]
        else
            splitimx[i] = splitimx[i]+splitimx[i-1]
        end
    end

    local transModuleIdx = 6
    nparam = model:getParameters():size(1)
    state.v = torch.Tensor(nparam):type(tensorType):fill(0)
    state.m = torch.Tensor(nparam):type(tensorType):fill(0)
    state.denom = torch.Tensor(nparam):type(tensorType):fill(0)
    state.t = 0
    state.v[{{splitimx[transModuleIdx-1]+1,splitimx[transModuleIdx]}}] = state_spectn.v
    state.m[{{splitimx[transModuleIdx-1]+1,splitimx[transModuleIdx]}}] = state_spectn.m
    state.denom[{{splitimx[transModuleIdx-1]+1,splitimx[transModuleIdx]}}] = state_spectn.denom

    -- train
    trainer = MyTrainer(model, criterion)
    trainer.lossHistory = lossHistory
    trainer.epoch = 1
    trainer.maxEpoch = maxEpoch or 40
    trainer.batchSize = batchSize or 24
    trainer.isoutput = true
    trainer.verbose = true
    trainer.outputeveryNepoch = 2
    trainer.datagen = gen_batch_input
    trainer.config.learningRateDecay = 1e-4
    trainer.config.weightDecay = 0
    local learningRate1 = lr1 or 3e-3
    local learningRate2 = lr2 or 1e-6
    trainer.config.learningRate = torch.Tensor({learningRate1,learningRate2,learningRate1})
    trainer.config.learningRateSplit = torch.Tensor({0,splitimx[transModuleIdx-1],splitimx[transModuleIdx],model:getParameters():size(1)})
    trainer.solver = optim.adammy
    trainer.config.beta1 = 0.9
    trainer.config.beta2 = 0.999
    trainer.state = state
    trainer.tensorType = tensorType --'torch.CudaTensor' or 'torch.DoubleTensor'
    trainer.evaluate = 1
    trainer.outputpath = './Model/'..synset..'/'
    trainer.outputprefix = 'SyncSpecCNN'..synset
    if not paths.dirp(trainer.outputpath) then
        paths.mkdir(trainer.outputpath)
    end
    
    -- start train
    trainer:train(trainList,testList,valList)
end


if not file_exists('./Model/'..synset..'/FuncMap'..synset..'_epoch'..params.e_b1..'.dat') then
    pretrainSpecTN(params.d, params.dj, params.b_b1, params.e_b1, params.l)
end
trainLearnTrans(params.d, params.dj, params.i, params.o, params.b, params.e, params.l, 1e-6, params.e_b1)
