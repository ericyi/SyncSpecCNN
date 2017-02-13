--[[ Reshape functional map.

  nDim           - Number of low frequency basis to be synchronized from each graph
  input          - Predicted functional map, of shape (N * nDim, nDimSync), nDimSync is the number of basis in the synchronized canonical domain

  output         - Reshaped functional map, of shape (N, nDim * nDimSync)

]]--

local FunctionalMapReshape, parent = torch.class('nn.FunctionalMapReshape', 'nn.Module')

function FunctionalMapReshape:__init(nDim)
   parent.__init(self)
   self.nDim = nDim or 30
end

function FunctionalMapReshape:updateOutput(input) 
   local ndim = self.nDim
   self.output = input:view(input:size(1)/ndim,ndim*input:size(2))
   return self.output
end

function FunctionalMapReshape:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput:viewAs(input)
   return self.gradInput
end

function FunctionalMapReshape:clearState()
   return parent.clearState(self)
end