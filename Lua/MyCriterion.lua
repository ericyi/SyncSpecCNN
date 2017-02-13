--[[ A combination of CrossEntropyCriterion and MSECriterion. CrossEntropyCriterion is used for vertex labeling. MSECriterion is used to regularize the functional map.

  nDim - Number of basis to be synchronized from each graph

]]--

require 'nn'
local MyCriterion, parent = torch.class('nn.MyCriterion', 'nn.Criterion')

function MyCriterion:__init(nDim)
    parent.__init(self)
    self.nDim = nDim or 12
    self.CrossEntropyCriterion = nn.CrossEntropyCriterion()
    self.MSECriterion = nn.MSECriterion()
end

function MyCriterion:updateOutput(input, target)
    self.output = self.CrossEntropyCriterion:updateOutput(input[1],target)
    
    -- enforce functional map to be orthonormal
    local ndim = self.nDim
    local ndim_sync = input[2]:size(2)/ndim
    self.transAll = torch.Tensor(input[2]:size(1),ndim,ndim_sync):typeAs(input[2]):fill(0)
    self.transxtrans = torch.Tensor(input[2]:size(1),ndim*ndim):typeAs(input[2]):fill(0)
    self.idtrans = torch.Tensor(input[2]:size(1),ndim*ndim):typeAs(input[2]):fill(0)
    for i=1,input[2]:size(1) do
        self.transAll[i] = input[2][i]:view(ndim,ndim_sync)
        self.transxtrans[i] = torch.mm(self.transAll[i],self.transAll[i]:t()):view(-1)
    end
    for i=1,ndim*ndim,ndim+1 do
        self.idtrans[{{},{i}}]:fill(1)
    end
    self.MSECriterion:updateOutput(self.transxtrans,self.idtrans)
    
    return self.output
end

function MyCriterion:updateGradInput(input, target)
    self.gradInput = {}
    self.gradInput[1] = self.CrossEntropyCriterion:updateGradInput(input[1],target)
    
    self.gradInput[2] = torch.Tensor():typeAs(input[2]):resizeAs(input[2]):fill(0)
    local gradTmp = self.MSECriterion:updateGradInput(self.transxtrans,self.idtrans):mul(1)
    local ndim = self.nDim
    local ndim_sync = input[2]:size(2)/ndim
    for i=1,input[2]:size(1) do
        self.gradInput[2][i] = torch.mm(torch.add(gradTmp[i]:view(ndim,ndim),gradTmp[i]:view(ndim,ndim):t()),self.transAll[i]):view(-1)
    end
    
    return self.gradInput
end

function MyCriterion:type(t)
    parent.type(self, t)
    self.CrossEntropyCriterion:type(t)
    self.MSECriterion:type(t)
    return self
end