require 'optim'
require 'adammy'
require 'matio'
local MyTrainer = torch.class('MyTrainer')

function MyTrainer:__init(model, criterion)
    self.epoch = 1
    self.maxEpoch = 30
    self.batchSize = 8
    self.shuffleIndices = true
    self.verbose = true
    self.solver = optim.adammy
    self.config = {}
    self.model = model
    self.criterion = criterion
    self.tensorType = 'torch.DoubleTensor'
    self.isoutput = true
    self.outputeveryNepoch = 1
    self.outputpath = './Models/'
    self.outputprefix = ''
    self.lossHistory = {}
    self.testIuHistory = {}
    self.valIuHistory = {}
    self.trainIuHistory = {}
    self.bestModelIdx = 0
    self.state = {} -- solver state
    self.evaluate = 0
    self.specParamJoint = nil
    self.datagen = gen_batch_input
end

-- trainset/testset/valset: several training/validation/test chunks, each corresponds to a structure with field (v: vertex functions, V: LB basis, D: LB energy, label: vertex label)
function MyTrainer:train(trainset,testset,valset)
    local batchSize = self.batchSize
    local model = self.model
    local criterion = self.criterion
    local params, gradParams = model:getParameters()
    local config = self.config
    local state = self.state
    local epoch = self.epoch
    
    local iter = 0
    for epoch = epoch, self.maxEpoch do
        local time = sys.clock()
        local ndata = 0
        for ii=1,trainset:size() do
            
            -- load current training chunk
            local dataset = trainset[ii]
            function dataset:size() 
                return #(dataset.v)
            end
            ndata = ndata+dataset:size()
            
            -- generate shuffle indices
            local shuffledIndices = torch.randperm(dataset:size())
            if not self.shuffleIndices then
                for t = 1,dataset:size() do
                    shuffledIndices[t] = t
                end
            end
            
            -- traverse each mini-batch
            for miniBatchStart = 1, dataset:size(), batchSize do
                
                -- generate input data for a mini-batch
                local curBatchSize = math.min(miniBatchStart+batchSize-1, dataset:size())-miniBatchStart+1
                batchInputs, batchLabels, batchSpecs = self.datagen(dataset,miniBatchStart,curBatchSize,shuffledIndices, self.tensorType,dataset,self.specParamJoint)

                if (#batchLabels)~=0 then
                    
                    -- set specParam
                    for mm=1,#(model.modules) do
                        if model.modules[mm].specParam~=nil then
                            model.modules[mm]:setSpecParam(batchSpecs)
                        end
                        if model.modules[mm].modules then
                            for nn=1,#(model.modules[mm].modules) do
                                if model.modules[mm].modules[nn].specParam~=nil then
                                    model.modules[mm].modules[nn]:setSpecParam(batchSpecs)
                                end
                            end
                        end
                    end

                    --create a func to evaluate df/dX where x are the model parameters at a given point
                    --and df/dx is the gradient of the loss wrt to thes parameters
                    local function feval(params)
                        gradParams:zero()
                        local outputs = model:forward(batchInputs)
                        local loss = criterion:forward(outputs, batchLabels)
                        local dloss_doutput = criterion:backward(outputs, batchLabels)
                        model:backward(batchInputs, dloss_doutput)
                        return loss,gradParams
                    end

                    --This function updates the 'params' variable (which is a view on the models parameters)
                    _,fx = self.solver(feval, params, config, state)
                    if self.verbose then
                        print(fx[1]) -- show loss
                    end
                    table.insert(self.lossHistory,fx)

                    if self.verbose then
                        xlua.progress(miniBatchStart, dataset:size()) --display progress
                    end
                    iter = iter+1
                end
            end
        end
        time = sys.clock() - time
        if self.verbose then
            print("time taken for 1 epoch = " .. (time) .. "s, time taken to learn 1 sample = " .. ((time/ndata)*1000) .. 'ms')
        end
        
        -- evalute the labeling quality through iu measurement
        if self.evaluate==1 then
            self:eva(trainset,testset,valset)
        end
        
        -- save model, solverstate, loss and evaluation results
        self.epoch = epoch
        if (self.epoch%self.outputeveryNepoch==0 and self.isoutput) or self.epoch==self.maxEpoch then
            self:save()
        end
    end
end

function MyTrainer:eva(trainset,testset,valset)
    local model = self.model
    model:evaluate()
    local batchSize = self.batchSize
    for iter=1,3 do
        
        -- choose training or test set
        iuAll = {}
        local dataset
        if iter==1 then
           dataset = trainset 
        elseif iter==2 then
           dataset = testset
        else
           dataset = valset
        end
        for ii=1,dataset:size() do
            local evaset = dataset[ii]
            function evaset:size() 
                return #(evaset.v)
            end
            local shuffledIndices = torch.randperm(evaset:size())
            for t = 1,evaset:size() do
                shuffledIndices[t] = t
            end
            iuAll[ii] = torch.zeros(evaset:size()):type(self.tensorType)
                    
            -- traverse mini-batch
            for miniBatchStart = 1, evaset:size(), batchSize do
                
                -- generate input data for a mini-batch
                local curBatchSize = math.min(miniBatchStart+batchSize-1, evaset:size())-miniBatchStart+1
                local batchInputs, batchLabels, batchSpecs = self.datagen(evaset,miniBatchStart,curBatchSize,shuffledIndices,self.tensorType,evaset,self.specParamJoint)
                for mm=1,#(model.modules) do
                    if model.modules[mm].specParam~=nil then
                        model.modules[mm]:setSpecParam(batchSpecs)
                    end
                    if model.modules[mm].modules then
                        for nn=1,#(model.modules[mm].modules) do
                            if model.modules[mm].modules[nn].specParam~=nil then
                                model.modules[mm].modules[nn]:setSpecParam(batchSpecs)
                            end
                        end
                    end
                end

                -- generate splitimx to split each shape for evaluation
                local splitimx = {}
                local nlabel = batchLabels:max()
                for i=1,#batchSpecs do
                    splitimx[i] = batchSpecs[i].V:size(1)
                end
                splitimx = torch.Tensor(splitimx)

                -- generate vertex label prediction
                outputs = model:forward(batchInputs)
                if torch.type(outputs)=="table" then
                    outputs = outputs[1]
                end    
                mm,preds = outputs:max(2)
                preds = preds:view(-1)
                preds = preds:typeAs(batchLabels)
                        
                -- evaluate iu for each shape and each label
                local count = 0
                for j=1,curBatchSize do
                    imx = preds[{{count+1,count+splitimx[j]}}]
                    gt = batchLabels[{{count+1,count+splitimx[j]}}]
                    local minlabel = gt:min()
                    local maxlabel = gt:max()
                    local labelcount = 0
                    for kk=minlabel,maxlabel do
                        local intersect = torch.add(gt:eq(kk),imx:eq(kk)):eq(2):sum()
                        local union = torch.add(gt:eq(kk),imx:eq(kk)):gt(0):sum()
                        if union==0 then
                            iuAll[ii][miniBatchStart+j-1] = iuAll[ii][miniBatchStart+j-1]+1
                            labelcount = labelcount+1
                        else
                             iuAll[ii][miniBatchStart+j-1] = iuAll[ii][miniBatchStart+j-1]+intersect/union
                             labelcount = labelcount+1
                        end
                    end
                    iuAll[ii][miniBatchStart+j-1] = iuAll[ii][miniBatchStart+j-1]/labelcount
                    count = count+splitimx[j]
                end
            end
        end
        iuAll = torch.cat(iuAll,1)
                
        if iter==1 then
            table.insert(self.trainIuHistory,iuAll:mean())
            print("train set iu:"..iuAll:mean())
        elseif iter==2 then
            table.insert(self.testIuHistory,iuAll:mean())
            print("test set iu:"..iuAll:mean())
        else
            table.insert(self.valIuHistory,iuAll:mean())
            print("val set iu:"..iuAll:mean())
            if self.bestModelIdx==0 then
                self.bestModelIdx = #(self.valIuHistory)
            elseif self.valIuHistory[self.bestModelIdx]<iuAll:mean() then
                self.bestModelIdx = #(self.valIuHistory)
            end
        end
    end
    if self.bestModelIdx>0 then
        print("test set iu of best model so far:"..self.testIuHistory[self.bestModelIdx])
    end
    model:training()
end

function MyTrainer:save()
    local model = self.model
    model:clearState()
    for mm=1,#(model.modules) do
        if model.modules[mm].specParam~=nil then
            model.modules[mm]:setSpecParam({})
        end
        if model.modules[mm].modules then
            for nn=1,#(model.modules[mm].modules) do
                if model.modules[mm].modules[nn].specParam~=nil then
                    model.modules[mm].modules[nn]:setSpecParam({})
                end
            end
        end
    end
    torch.save(self.outputpath..self.outputprefix..'_epoch'..self.epoch..'.dat',{self.lossHistory,self.model,self.state})
    if self.evaluate==1 then
        torch.save(self.outputpath..self.outputprefix..'_eva.dat',{self.testIuHistory,self.valIuHistory,self.trainIuHistory,self.bestModelIdx})
    end
end