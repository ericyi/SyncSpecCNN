--[[ Spectral modulation layer for low frequency end of the spectrum, which is the counter part for spatial convolution.

  nInputPlane    - Number of input channels
  nOutputPlane   - Number of output channels
  nDim           - Number of basis to be modulated
  input          - Input spectral data, of shape (N * nDim, nInputPlane), N is the batch size
  weight         - Learnable modulation parameters, of shape (nDim, nInputputPlane, nOutputPlane)

  output         - Output spectral data, of shape (N * nDim, nOutputPlane)
]]--

local SpecModulationCutDim, parent = torch.class('nn.SpecModulationCutDim', 'nn.Module')

function SpecModulationCutDim:__init(nInputPlane, nOutputPlane, nDim)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.nDim = nDim

   self.weight = torch.Tensor(nDim, nInputPlane, nOutputPlane)
   self.gradWeight = torch.Tensor(nDim, nInputPlane, nOutputPlane):fill(0)

   self:reset()
end


function SpecModulationCutDim:reset()
   self.weight = torch.randn(self.weight:size(1),self.weight:size(2),self.weight:size(3))
   self.weight:mul(math.sqrt(2)/math.sqrt(self.nInputPlane))
end

function SpecModulationCutDim:updateOutput(input)
   self.output = torch.Tensor(input:size(1),self.nOutputPlane):typeAs(input):fill(0)
   local nin = self.nInputPlane
   local nout = self.nOutputPlane
   local ndim = self.nDim
   local nbatch = input:size(1)/ndim
   
   for i=1,nbatch do
      self.output[{{(i-1)*ndim+1,i*ndim},{}}]:add(input[{{(i-1)*ndim+1,i*ndim},{}}]:clone():view(ndim,nin,1):repeatTensor(1,1,nout):cmul(self.weight):sum(2):squeeze(2))
   end
   return self.output
end

function SpecModulationCutDim:updateGradInput(input, gradOutput)
   local nin = self.nInputPlane
   local nout = self.nOutputPlane
   local ndim = self.nDim
   local nbatch = input:size(1)/ndim
   self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):fill(0)
    
   for i=1,nbatch do
      self.gradInput[{{(i-1)*ndim+1,i*ndim},{}}]:add(gradOutput[{{(i-1)*ndim+1,i*ndim},{}}]:clone():view(ndim,1,nout):repeatTensor(1,nin,1):cmul(self.weight):sum(3):squeeze(3))
   end
   return self.gradInput
end

function SpecModulationCutDim:accGradParameters(input, gradOutput, scale)
   local nin = self.nInputPlane
   local nout = self.nOutputPlane
   local ndim = self.nDim
   local nbatch = input:size(1)/ndim
    
   for i=1,nbatch do
      self.gradWeight:add(input[{{(i-1)*ndim+1,i*ndim},{}}]:clone():view(ndim,nin,1):repeatTensor(1,1,nout):cmul(gradOutput[{{(i-1)*ndim+1,i*ndim},{}}]:clone():view(ndim,1,nout):repeatTensor(1,nin,1)))
   end
end

function SpecModulationCutDim:clearState()
   return parent.clearState(self)
end