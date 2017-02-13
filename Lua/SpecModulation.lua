--[[ Spectral modulation layer, which is the counter part for spatial convolution.

  nInputPlane    - Number of input channels
  nOutputPlane   - Number of output channels
  gamma          - Dilation parameter
  nParam         - Number of kernel parameters
  specParam[i].D - Laplacian eigenvalues of graph_i, of shape (m_i, 1), m_i is the number of spectral basis used for graph_i, i=1,...,N, N is the batch size
  specParam[i].V - Laplacian eigenbasis of graph_i, of shape (x_i, m_i), with each column corresponding to one basis, x_i is the number of vertices for graph_i
  input          - Input spectral data, of shape (m_1 + m_2 + ... + m_N, nInputPlane)
  weight         - Learnable modulation parameters, of shape (nParam, nInputputPlane, nOutputPlane)
  bias           - Bias, of shape (nOutputPlane, )

  output         - Output spectral data, of shape (m_1 + m_2 + ... + m_N, nOutputPlane)

]]--

local SpecModulation, parent = torch.class('nn.SpecModulation', 'nn.Module')

function SpecModulation:__init(nInputPlane, nOutputPlane, gamma, nParam)
   parent.__init(self)
    
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.gamma = gamma
   self.nParam = nParam or 7
   self.specParam = {}

   self.weight = torch.Tensor(nParam, nInputPlane, nOutputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nParam, nInputPlane, nOutputPlane):fill(0)
   self.gradBias = torch.Tensor(nOutputPlane):fill(0)

   self:reset()
end

function SpecModulation:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function SpecModulation:setSpecParam(specParam)
   self.specParam = specParam
   return self
end

function SpecModulation:reset()
   self.weight = torch.randn(self.weight:size(1),self.weight:size(2),self.weight:size(3))
   self.weight:mul(math.sqrt(2)/math.sqrt(self.nInputPlane))
   if self.bias then
      self.bias = self.bias:fill(0)
   end 
end

function SpecModulation:updateOutput(input)
   -- split index for input and output (spectral representation)
   local splitimx = {}
   for i=1,#self.specParam do
      splitimx[i] = self.specParam[i].V:size(2)
   end
   splitimx = torch.Tensor(splitimx)
   self.splitimx = splitimx
   assert(splitimx:sum()==input:size(1),'Input and specParam do not match!')
    
   self.output = torch.Tensor(input:size(1),self.nOutputPlane):typeAs(input):fill(0)
   local nin = self.nInputPlane
   local nout = self.nOutputPlane
   local np = self.nParam
   local gamma = self.gamma
   local count=0
   assert(np%2==1,'Please set nParam an odd number!')
   
   self.W = {}
   for i=1,#self.specParam do
      -- generate spectral multipliers from weights and bias
      local m = self.specParam[i].D:size(1)
      local W = self.weight[{{1},{},{}}]:repeatTensor(m, 1, 1)
      for j=1,(np-1)/2 do
         W:add( self.specParam[i].D:view(m,1,1):clone():mul(-j*gamma):exp():cmul( self.specParam[i].D:view(m,1,1):clone():mul(j*3.14*gamma):sin() ):repeatTensor(1, nin, nout):cmul(self.weight[{{2*j},{},{}}]:repeatTensor(m, 1, 1)) )
         W:add( self.specParam[i].D:view(m,1,1):clone():mul(-j*gamma):exp():cmul( self.specParam[i].D:view(m,1,1):clone():mul(j*3.14*gamma):cos() ):repeatTensor(1, nin, nout):cmul(self.weight[{{2*j+1},{},{}}]:repeatTensor(m, 1, 1)) )
      end
      self.W[i] = W
      
      -- update output
      self.output[{{count+1,count+splitimx[i]},{}}] = torch.cmul( input[{{count+1,count+splitimx[i]},{}}]:clone():view(m,nin,1):expandAs(W), W ):sum(2):squeeze(2)
      if self.bias then
         local sgn = 1
         if self.specParam[i].V[{{},{1}}]:mean()<0 then
            sgn = -1
         end
         self.output[count+1]:add( torch.mul( self.bias:view(1,nout),math.sqrt(self.specParam[i].V:size(1))*sgn ) )
      end
      count = count+splitimx[i]
   end
   return self.output
end

function SpecModulation:updateGradInput(input, gradOutput)
   local splitimx = self.splitimx
   local nin = self.nInputPlane
   local nout = self.nOutputPlane
   local np = self.nParam
   local gamma = self.gamma
   local count=0
   self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):fill(0)
    
   -- update gradInput
   for i=1,#self.specParam do
      local m = self.specParam[i].D:size(1)
      self.gradInput[{{count+1,count+splitimx[i]},{}}] = gradOutput[{{count+1,count+splitimx[i]},{}}]:clone():view(m,1,nout):repeatTensor(1,nin,1):cmul(self.W[i]):sum(3):squeeze(3)
      count = count+splitimx[i]
   end
   return self.gradInput
end

function SpecModulation:accGradParameters(input, gradOutput, scale)
   local splitimx = self.splitimx
   local nin = self.nInputPlane
   local nout = self.nOutputPlane
   local np = self.nParam
   local gamma = self.gamma
   local count=0
    
   for i=1,#self.specParam do
      -- update gradient for multipliers
      local m = self.specParam[i].D:size(1)
      local dW = gradOutput[{{count+1,count+splitimx[i]},{}}]:clone():view(m,1,nout):repeatTensor(1,nin,1):cmul(input[{{count+1,count+splitimx[i]},{}}]:clone():view(m, nin, 1):expand(m, nin, nout))
      -- update gradWeight
      self.gradWeight[{{1},{},{}}]:add(dW:sum(1))
      for j=1,(np-1)/2 do
         self.gradWeight[{{2*j},{},{}}]:add( self.specParam[i].D:view(m,1,1):clone():mul(-j*gamma):exp():cmul( self.specParam[i].D:view(m,1,1):clone():mul(j*3.14*gamma):sin() ):repeatTensor(1, nin, nout):cmul(dW):sum(1) )
         self.gradWeight[{{2*j+1},{},{}}]:add( self.specParam[i].D:view(m,1,1):clone():mul(-j*gamma):exp():cmul( self.specParam[i].D:view(m,1,1):clone():mul(j*3.14*gamma):cos() ):repeatTensor(1, nin, nout):cmul(dW):sum(1) )
      end
      -- update gradBias
      if self.bias then
         local sgn = 1
         if self.specParam[i].V[{{},{1}}]:mean()<0 then
            sgn = -1
         end
         self.gradBias:add(torch.mul(gradOutput[{{count+1},{}}],math.sqrt(self.specParam[i].V:size(1))*sgn):view(-1))
      end
      count = count+splitimx[i]
   end
end

function SpecModulation:clearState()
   return parent.clearState(self)
end