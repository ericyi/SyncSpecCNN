--[[ Apply spectral synchronization, which synchronizes the low frequency end of different graph spectrums via a learnable functional map. It also conducts spectrum cutting/restoration for the favor of synchronization. It either cuts out the low frequency end of a spectrum, or restore the full spectrum from its low frequency end via zero padding.

  nDim           - Number of low frequency basis kept after cutting or before restoration
  mode           - Layer behavior controller, 0: spectrum cutting, 1: spectrum restoration
  specParam[i].D - Laplacian eigenvalues of graph_i, of shape (m_i, 1), m_i is the number of spectral basis for graph_i, i=1,...,N, N is the batch size
  input[1]       - Input spectral data, of shape (m_1 + m_2 + ... + m_N, c), c is the number of signal channels
  input[2]       - Functional map for spectral synchronization, of shape (N, nDim * nDimSync), nDimSync is the number of basis in the canonical domain, each spectrum with nDim basis cut out will be synchronized in a canonical domain with nDimSync basis

  output         - Output spectral data, of shape (N * nDimSync, c), c is the number of signal channels

]]--

local SpecSynchronization, parent = torch.class('nn.SpecSynchronization', 'nn.Module')

function SpecSynchronization:__init(nDim,mode)
   parent.__init(self)
   self.nDim = nDim
   self.mode = mode
   self.specParam = {}
end

function SpecSynchronization:setSpecParam(specParam)
   self.specParam = specParam
   return self
end

function SpecSynchronization:updateOutput(input)
   -- split index for input (spectral representation)
   local splitimx = {}
   for i=1,#self.specParam do
      splitimx[i] = self.specParam[i].D:size(1)
   end
   splitimx = torch.Tensor(splitimx)
   self.splitimx = splitimx
    
   local ndim = self.nDim
   local ndim_sync = input[2]:size(2)/ndim
   local count=0
   self.trans = input[2]:clone()
    
   if self.mode==0 then
      -- cutting spectrum
      self.output = torch.Tensor(#self.specParam*ndim_sync,input[1]:size(2)):typeAs(input[1]):fill(0)
      for i=1,#self.specParam do
         self.output[{{(i-1)*ndim_sync+1,i*ndim_sync},{}}] = torch.mm(self.trans[i]:view(ndim,ndim_sync):t(),input[1][{{count+1,count+ndim},{}}])
         count = count+splitimx[i]
      end
   elseif self.mode==1 then
      -- restoring spectrum
      self.output = torch.Tensor(splitimx:sum(),input[1]:size(2)):typeAs(input[1]):fill(0)
      for i=1,#self.specParam do
         self.output[{{count+1,count+ndim},{}}] = torch.mm(self.trans[i]:view(ndim,ndim_sync),input[1][{{(i-1)*ndim_sync+1,i*ndim_sync},{}}])
         count = count+splitimx[i]
      end
   end
   
   return self.output
end

function SpecSynchronization:updateGradInput(input, gradOutput)
   local splitimx = self.splitimx
   local ndim = self.nDim
   local ndim_sync = input[2]:size(2)/ndim
   local count=0
    
   self.gradInput = {}
   self.gradInput[1] = torch.Tensor():typeAs(input[1]):resizeAs(input[1]):fill(0)
   self.gradInput[2] = torch.Tensor():typeAs(input[2]):resizeAs(input[2]):fill(0)
   if self.mode==0 then
      -- cutting spectrum
      for i=1,#self.specParam do
         self.gradInput[1][{{count+1,count+ndim},{}}] = torch.mm(self.trans[i]:view(ndim,ndim_sync),gradOutput[{{(i-1)*ndim_sync+1,i*ndim_sync},{}}])
         self.gradInput[2][i] = torch.mm(gradOutput[{{(i-1)*ndim_sync+1,i*ndim_sync},{}}],input[1][{{count+1,count+ndim},{}}]:t()):view(1,input[2]:size(2))
         count = count+splitimx[i]
      end
   elseif self.mode==1 then
      -- restoring spectrum
      for i=1,#self.specParam do
         self.gradInput[1][{{(i-1)*ndim_sync+1,i*ndim_sync},{}}] = torch.mm(self.trans[i]:view(ndim,ndim_sync):t(),gradOutput[{{count+1,count+ndim},{}}])
         self.gradInput[2][i] = torch.mm(gradOutput[{{count+1,count+ndim},{}}],input[1][{{(i-1)*ndim_sync+1,i*ndim_sync},{}}]:t()):t():clone():view(1,input[2]:size(2))
         count = count+splitimx[i]
      end
   end   
   return self.gradInput
end

function SpecSynchronization:clearState()
   return parent.clearState(self)
end