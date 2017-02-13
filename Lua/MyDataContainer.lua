require 'torchx'
matio = require 'matio'

-- sample data for SpecTN pretraining
function gen_batch_input_spectn(dataset,miniBatchStart,curBatchSize,shuffledIndices,tensorType,dataAll,specParamJoint)
    local batchInputs = {}
    local batchLabels = {}
    local batchSpecs = {}
    local batchSpecsJoint = {}
    local ndim_sep = specParamJoint.ndim_sep or 12
    local ndim_joint = specParamJoint.ndim_joint or 36
    local nfea = dataAll.v[1]:size(2)
    for i = miniBatchStart, miniBatchStart+curBatchSize-1 do
        batchInputs[i-miniBatchStart+1] = dataAll.v[shuffledIndices[i]]:type(tensorType)
        batchSpecs[i-miniBatchStart+1] = {}
        batchSpecs[i-miniBatchStart+1].V = dataAll.V[shuffledIndices[i]]:type(tensorType)
        batchSpecs[i-miniBatchStart+1].D = dataAll.D[shuffledIndices[i]]:type(tensorType)
        
        v = dataAll.v[shuffledIndices[i]][{{},{nfea-2,nfea}}]:clone():type(tensorType)
        local vmax = torch.abs(v):max(1)[1]
        v[{{},{1}}] = v[{{},{1}}]/vmax[1]
        v[{{},{2}}] = v[{{},{2}}]/vmax[2]
        v[{{},{3}}] = v[{{},{3}}]/vmax[3]
        local ngrid = specParamJoint.ngrid
        local res = 2/(ngrid-1)
        local v_grid = ((v+1+1e-16)/res):ceil()
        
        v_grid = ((v_grid[{{},{3}}]-1)*ngrid*ngrid+(v_grid[{{},{2}}]-1)*ngrid+v_grid[{{},{1}}]):view(-1)
        imx = specParamJoint.vidx:index(1,v_grid:type("torch.LongTensor")):view(-1):type("torch.LongTensor")
        batchSpecsJoint[i-miniBatchStart+1] = {}
        batchSpecsJoint[i-miniBatchStart+1].V = specParamJoint.V[{{},{1,ndim_joint}}]:index(1,imx):type(tensorType) 
        batchSpecsJoint[i-miniBatchStart+1].V:cdiv(torch.sqrt(torch.pow(batchSpecsJoint[i-miniBatchStart+1].V,2):sum(1)):expand(batchSpecsJoint[i-miniBatchStart+1].V:size(1),ndim_joint))
        batchSpecsJoint[i-miniBatchStart+1].D = specParamJoint.D[{{1,ndim_joint},{}}]:type(tensorType)
        
        batchLabels[i-miniBatchStart+1] = torch.mm( batchSpecs[i-miniBatchStart+1].V[{{},{1,ndim_sep}}]:t(), batchSpecsJoint[i-miniBatchStart+1].V ):type(tensorType)
    end
    batchInputs = torch.cat(batchInputs,1)
    batchLabels = torch.cat(batchLabels,1)    
    return batchInputs, batchLabels, batchSpecs
end

-- sample data for main training
function gen_batch_input(dataset,miniBatchStart,curBatchSize,shuffledIndices,tensorType,dataAll)
    local batchInputs = {}
    local batchLabels = {}
    local batchSpecs = {}
    for i = miniBatchStart, miniBatchStart+curBatchSize-1 do
        batchInputs[i-miniBatchStart+1] = dataAll.v[shuffledIndices[i]]:type(tensorType)
        batchSpecs[i-miniBatchStart+1] = {}
        batchSpecs[i-miniBatchStart+1].V = dataAll.V[shuffledIndices[i]]:type(tensorType)
        batchSpecs[i-miniBatchStart+1].D = dataAll.D[shuffledIndices[i]]:type(tensorType)
        batchLabels[i-miniBatchStart+1] = dataAll.label[shuffledIndices[i]]:view(-1):type(tensorType)
    end
    batchInputs = torch.cat(batchInputs,1)
    batchLabels = torch.cat(batchLabels,1):view(-1)
    return batchInputs, batchLabels, batchSpecs
end
