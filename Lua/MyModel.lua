require 'nn'
require 'nngraph'
require 'LaplacianVolReparam'
require 'FunctionalMapReshape'
require 'PD'
require 'DP'
require 'SpecModulation'
require 'SpecSynchronization'
require 'SpecModulationCutDim'

-- SpecTN: spectral transformer network
-- predicts functional map from laplacian eigenbasis
-- nDim: number of low frequency basis to be synchronized from each shape graph
-- nDimSync: number of basis of the synchronized canonical domain
function SpecTN(nDim, nDimSync)
    local inputs = {}
    inputs[1] = nn.Identity()()
    local ndim = nDim or 12
    local ndim_sync = nDimSync or 36
    local ngrid = 20 -- voxelization resolution
    local nchannel = 50 -- base number of signal channels
    local embed = nn.LaplacianVolReparam(ndim,ngrid)(inputs)

    embed = nn.VolumetricConvolution(1,nchannel,3,3,3,1,1,1,1,1,1)(embed)
    embed = nn.ReLU(true)(nn.VolumetricBatchNormalization(nchannel)(embed))
    embed = nn.VolumetricMaxPooling(2,2,2,2,2,2)(embed)
    
    embed = nn.VolumetricConvolution(nchannel,nchannel*2,3,3,3,1,1,1,1,1,1)(embed)
    embed = nn.ReLU(true)(nn.VolumetricBatchNormalization(nchannel*2)(embed))
    embed = nn.VolumetricMaxPooling(2,2,2,2,2,2)(embed)
    
    embed = nn.VolumetricConvolution(nchannel*2,nchannel*4,5,5,5,1,1,1)(embed)
    embed = nn.ReLU(true)(nn.BatchNormalization(nchannel*4)(nn.Squeeze()(embed)))
    embed = nn.Dropout(0.3)(embed)
    
    local trans = nn.Linear(nchannel*4,ndim_sync)(embed)
    local net = nn.gModule(inputs,{trans})
    return net
end

-- One typical block in SyncSpecCNN, used for conducting spectral convolution without SpecTN 
-- inChannel: number of input signal channels
-- outChannel: number of output signal channels
-- gamma: dilation parameter for convolution
-- nParam: number of kernel parameters
function Block(inChannel, outChannel, gamma, nParam)
    local inputs = {}
    inputs[1] = nn.Identity()()
    
    local embed_skip = nn.SpecModulation(inChannel,outChannel,gamma,1)( nn.PD()(inputs) )
    embed_skip = nn.BatchNormalization(outChannel)( nn.DP()(embed_skip) )
    
    local embed = nn.SpecModulation(inChannel,outChannel,gamma,nParam)( nn.PD()(inputs) )
    embed = nn.BatchNormalization(outChannel)( nn.DP()(embed) )
    embed = nn.Dropout(0.2)( nn.ReLU(true)(embed) )
    embed = nn.SpecModulation(outChannel,outChannel,gamma,1)( nn.PD()(embed) )
    embed = nn.BatchNormalization(outChannel)( nn.DP()(embed) )
    
    embed = nn.CAddTable()({embed,embed_skip})
    embed = nn.Dropout(0.2)( nn.ReLU(true)(embed) )
    
    local net = nn.gModule(inputs,{embed})
    return net
end

-- SyncSpecCNN: synchronized spectral CNN
-- specTN: spectral transformer network
-- nDim: number of low frequency basis to be synchronized from each shape graph
-- nDimSync: number of basis of the synchronized canonical domain
-- nIn: #(input vertex functions)
-- nOut: #(output vertex functions)
function SyncSpecCNN(specTN, nDim, nDimSync, nIn, nOut)
    local inputs = {}
    local output = {}
    inputs[1] = nn.Identity()()
    local ndim = nDim or 12
    local ndim_sync = nDimSync or 36
    local nin = nIn
    local nout = nOut
    local nparam = 7 -- number of kernel parameters
    local nchannel = 50 -- base number of signal channels
    
    local trans = specTN(inputs)
    trans = nn.FunctionalMapReshape(ndim)(trans)
    local embed = inputs
    
    -- conv without SpecTN
    embed = Block(nin, nchannel, 4, nparam)(embed)
    embed = Block(nchannel, nchannel, 16, nparam)(embed)
    local embed_skip = Block(nchannel, 2*nchannel, 32, nparam)(embed)
    
    -- conv with SpecTN 
    embed = nn.SpecSynchronization(ndim,0)({nn.PD()(embed_skip),trans})
    embed = nn.SpecModulationCutDim(2*nchannel,2*nchannel,ndim_sync)(embed)
    embed = nn.DP()(nn.SpecSynchronization(ndim,1)({embed,trans}))
    embed = nn.BatchNormalization(2*nchannel)(embed)
    embed = nn.Dropout(0.2)(nn.ReLU(true)(embed))
    
    embed = nn.SpecSynchronization(ndim,0)({nn.PD()(embed),trans})
    embed = nn.SpecModulationCutDim(2*nchannel,2*nchannel,ndim_sync)(embed)
    embed = nn.DP()(nn.SpecSynchronization(ndim,1)({embed,trans}))
    embed = nn.BatchNormalization(2*nchannel)(embed)
    embed = nn.Dropout(0.2)(nn.ReLU(true)(embed))
    
    embed = nn.SpecSynchronization(ndim,0)({nn.PD()(embed),trans})
    embed = nn.SpecModulationCutDim(2*nchannel,2*nchannel,ndim_sync)(embed)
    embed = nn.DP()(nn.SpecSynchronization(ndim,1)({embed,trans}))
    embed = nn.BatchNormalization(2*nchannel)(embed)
    
    embed = nn.CAddTable()({embed,embed_skip})
    
    -- conv without SpecTN
    embed = Block(2*nchannel, 2*nchannel, 4, nparam)(embed)
 
    output = nn.Linear(2*nchannel,nout)(embed)
    local net = nn.gModule(inputs,{output,trans})
    return net
end