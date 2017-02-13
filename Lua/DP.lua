--[[ Dual-Primal conversion layer. Convert spectral signals into their spatial representation.

  input          - Input spectral data, of shape (m_1 + m_2 + ... + m_N, c), m_i is the number of spectral basis used for graph_i, N is the batch size, c is the number of signal channels
  specParam[i].V - Laplacian eigenbasis of graph_i, of shape (x_i, m_i), with each column corresponding to one basis, i=1,...,N

  output         - Output spatial data, of shape (x_1 + x_2 + ... + x_N, c), x_i is the number of vertices for graph_i, N is the batch size, c is the number of signal channels
]]--

local DP, parent = torch.class('nn.DP', 'nn.Module')

function DP:__init()
   parent.__init(self)
   self.specParam = {}
   self:reset()
end

function DP:setSpecParam(specParam)
   self.specParam = specParam
   return self
end

function DP:updateOutput(input)
   -- split index for output (spatial representation)
   local splitimxSpa = {}
   for i=1,#self.specParam do
      splitimxSpa[i] = self.specParam[i].V:size(1)
   end
   splitimxSpa = torch.Tensor(splitimxSpa)
   self.splitimxSpa = splitimxSpa
   
   -- split index for input (spectral representation)
   local splitimxSpec = {}
   for i=1,#self.specParam do
      splitimxSpec[i] = self.specParam[i].V:size(2)
   end
   splitimxSpec = torch.Tensor(splitimxSpec)
   self.splitimxSpec = splitimxSpec
   assert(splitimxSpec:sum()==input:size(1),'Input and specParam do not match!')
   
   -- update output
   self.output = torch.Tensor(splitimxSpa:sum(),input:size(2)):typeAs(input):fill(0)
   local countSpa = 0
   local countSpec = 0
   for i=1,#self.specParam do
      self.output[{{countSpa+1,countSpa+splitimxSpa[i]},{}}] = torch.mm(self.specParam[i].V,input[{{countSpec+1,countSpec+splitimxSpec[i]},{}}])
      countSpa = countSpa+splitimxSpa[i]
      countSpec = countSpec+splitimxSpec[i]
   end
   
   return self.output
end

function DP:updateGradInput(input, gradOutput)
   local splitimxSpa = self.splitimxSpa
   local splitimxSpec = self.splitimxSpec
   
   local countSpa = 0
   local countSpec = 0
   self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):fill(0)
   
   -- update gradInput
   for i=1,#self.specParam do
      self.gradInput[{{countSpec+1,countSpec+splitimxSpec[i]},{}}] = torch.mm(self.specParam[i].V:t(),gradOutput[{{countSpa+1,countSpa+splitimxSpa[i]},{}}])
      countSpa = countSpa+splitimxSpa[i]
      countSpec = countSpec+splitimxSpec[i]
   end
   return self.gradInput
end

function DP:clearState()
   return parent.clearState(self)
end